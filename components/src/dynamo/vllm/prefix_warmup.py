# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode-worker prefix warmup before model registration.

Each decode/aggregated worker runs warmup requests through its own engine so
prefix/KV cache and attention-kernel JIT compilation happen per worker process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import Any
from urllib.parse import urlparse

from vllm.config import VllmConfig
from vllm.inputs import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo.common.utils.tokenizer_utils import normalize_prompt_token_ids

from .args import Config
from .constants import DisaggregationMode
from .handlers import build_sampling_params_openai, get_dp_range_for_worker

logger = logging.getLogger(__name__)


def _prefix_warmup_settings(
    config: Config,
) -> tuple[str | None, int | None, bool]:
    """Resolve prefix warmup flags from Dynamo decode/aggregated worker args."""
    return (
        config.prefix_warmup_file,
        config.prefix_warmup_count,
        config.prefix_warmup_parallel,
    )


def _load_prefix_warmup_bytes(location: str) -> bytes:
    """Load warmup JSON from a local path or http(s) URL."""
    scheme = urlparse(location).scheme.lower()

    if scheme in ("http", "https"):
        try:
            with urllib.request.urlopen(location, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            raise RuntimeError(
                f"Failed to download prefix warmup file from {location}: {e}"
            ) from e

    try:
        with open(location, "rb") as f:
            return f.read()
    except FileNotFoundError as e:
        raise RuntimeError(f"Prefix warmup file not found: {location}") from e


def _validate_prefix_warmup_entry(entry: object, idx: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"Prefix warmup entry #{idx} must be a JSON object, got "
            f"{type(entry).__name__}"
        )
    messages = entry.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError(
            f"Prefix warmup entry #{idx} must have a non-empty 'messages' list"
        )
    return entry


def _get_engine_tokenizer(engine_client: AsyncLLM) -> Any:
    tokenizer = getattr(engine_client, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer

    input_processor = getattr(engine_client, "input_processor", None)
    if input_processor is not None:
        get_tokenizer = getattr(input_processor, "get_tokenizer", None)
        if callable(get_tokenizer):
            return get_tokenizer()

    raise RuntimeError(
        "Prefix warmup requires a tokenizer on the vLLM engine client, but none "
        "was found."
    )


def _build_warmup_prompt(entry: dict[str, Any], tokenizer: Any) -> TokensPrompt:
    messages = entry["messages"]
    chat_template_kwargs = entry.get("chat_template_kwargs")
    tools = entry.get("tools")

    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if chat_template_kwargs is not None:
        template_kwargs["chat_template_kwargs"] = chat_template_kwargs
    if tools is not None:
        template_kwargs["tools"] = tools

    try:
        rendered = tokenizer.apply_chat_template(messages, **template_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to apply chat template for prefix warmup entry: {e}"
        ) from e

    try:
        token_ids = normalize_prompt_token_ids(rendered)
    except TypeError as e:
        raise RuntimeError(
            "Chat template for prefix warmup did not produce token ids"
        ) from e

    if not token_ids:
        raise RuntimeError(
            "Chat template for prefix warmup did not produce token ids"
        )

    return TokensPrompt(prompt_token_ids=token_ids)


async def _run_prefix_warmup_entry(
    engine_client: AsyncLLM,
    entry: dict[str, Any],
    idx: int,
    total: int,
    default_sampling_params: dict[str, Any],
    data_parallel_rank: int | None,
) -> None:
    tokenizer = _get_engine_tokenizer(engine_client)
    prompt = _build_warmup_prompt(entry, tokenizer)

    request = dict(entry)
    request.setdefault("max_tokens", 1)
    request["stream"] = False
    sampling_params = build_sampling_params_openai(request, default_sampling_params)
    sampling_params.max_tokens = request.get("max_tokens", 1)
    sampling_params.detokenize = False

    request_id = f"prefix-warmup-{idx}"
    gen_kwargs: dict[str, Any] = {}
    if data_parallel_rank is not None:
        gen_kwargs["data_parallel_rank"] = data_parallel_rank

    gen = engine_client.generate(
        prompt,
        sampling_params,
        request_id,
        **gen_kwargs,
    )

    prompt_tokens: int | str = "?"
    async for res in gen:
        if not res.outputs:
            raise RuntimeError(
                f"Prefix warmup request #{idx} returned no outputs from the engine"
            )
        for output in res.outputs:
            if output.finish_reason:
                usage = getattr(res, "usage", None) or getattr(res, "metrics", None)
                if usage is not None:
                    prompt_tokens = getattr(usage, "num_prompt_tokens", prompt_tokens)
                    if prompt_tokens == "?":
                        prompt_tokens = getattr(usage, "prompt_tokens", "?")
                logger.info(
                    "Prefix warmup request #%d/%d primed (prompt_tokens=%s)",
                    idx + 1,
                    total,
                    prompt_tokens,
                )
                return

    raise RuntimeError(
        f"Prefix warmup request #{idx} finished without a completion signal"
    )


async def run_decode_prefix_warmup(
    engine_client: AsyncLLM,
    vllm_config: VllmConfig,
    config: Config,
    default_sampling_params: dict[str, Any],
) -> None:
    """Prime prefix cache and JIT kernels on decode/aggregated workers."""
    if config.disaggregation_mode == DisaggregationMode.PREFILL:
        return

    warmup_file, warmup_count, warmup_parallel = _prefix_warmup_settings(config)
    if not warmup_file:
        return

    raw = _load_prefix_warmup_bytes(warmup_file)
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Prefix warmup file is not valid JSON ({warmup_file}): {e}"
        ) from e

    if not isinstance(entries, list):
        raise RuntimeError(
            f"Prefix warmup file must contain a JSON list, got "
            f"{type(entries).__name__}: {warmup_file}"
        )
    if not entries:
        logger.warning(
            "Prefix warmup file %s is empty; nothing to warm up.", warmup_file
        )
        return

    if warmup_count is not None:
        if warmup_count <= 0:
            logger.warning(
                "Prefix warmup count %d is not positive; nothing to warm up.",
                warmup_count,
            )
            return
        entries = entries[:warmup_count]

    validated = [
        _validate_prefix_warmup_entry(entry, idx) for idx, entry in enumerate(entries)
    ]
    total = len(validated)

    try:
        dp_start, _ = get_dp_range_for_worker(vllm_config)
        data_parallel_rank = dp_start
    except Exception:
        logger.warning(
            "Could not determine DP rank for prefix warmup; engine will choose",
            exc_info=True,
        )
        data_parallel_rank = None

    logger.info(
        "Running decode prefix warmup: %d request(s) from %s (%s)",
        total,
        warmup_file,
        "parallel" if warmup_parallel else "sequential",
    )

    if warmup_parallel:
        await asyncio.gather(
            *[
                _run_prefix_warmup_entry(
                    engine_client,
                    entry,
                    idx,
                    total,
                    default_sampling_params,
                    data_parallel_rank,
                )
                for idx, entry in enumerate(validated)
            ]
        )
    else:
        for idx, entry in enumerate(validated):
            await _run_prefix_warmup_entry(
                engine_client,
                entry,
                idx,
                total,
                default_sampling_params,
                data_parallel_rank,
            )

    logger.info("Decode prefix warmup complete: %d request(s) primed.", total)
