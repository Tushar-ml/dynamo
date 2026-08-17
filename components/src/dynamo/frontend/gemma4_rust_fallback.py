# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo Rust parser fallback for Gemma4 when vLLM parsers miss or leak markup."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from typing import Any

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)

from .gemma4_format import has_gemma4_tool_markup

logger = logging.getLogger(__name__)

_DYNAMO_CORE: Any | None = None
_DYNAMO_CORE_TRIED = False


def _load_dynamo_core() -> Any | None:
    global _DYNAMO_CORE, _DYNAMO_CORE_TRIED
    if _DYNAMO_CORE_TRIED:
        return _DYNAMO_CORE
    _DYNAMO_CORE_TRIED = True
    try:
        _DYNAMO_CORE = importlib.import_module("dynamo._core")
    except (ImportError, OSError):
        _DYNAMO_CORE = None
    return _DYNAMO_CORE


def vllm_extraction_needs_rust_fallback(
    text: str,
    *,
    tools_called: bool,
    content: str | None,
) -> bool:
    if tools_called:
        return False
    if not has_gemma4_tool_markup(text):
        return False
    if content and has_gemma4_tool_markup(content):
        return True
    return True


async def extract_gemma4_tool_calls_rust(
    text: str,
    tools: list[dict[str, Any]] | None,
) -> tuple[list[DeltaToolCall], str | None] | None:
    core = _load_dynamo_core()
    if core is None:
        return None

    tools_json = json.dumps(tools) if tools else None
    try:
        result_json: str = await core.parse_tool_calls_batch(
            "gemma4", text, tools_json
        )
        raw = json.loads(result_json)
    except Exception as exc:
        logger.warning("Gemma4 Rust fallback parse failed: %s", exc)
        return None

    calls_raw = raw.get("calls") or []
    if not calls_raw:
        return None

    tool_deltas: list[DeltaToolCall] = []
    for i, call in enumerate(calls_raw):
        fn = call.get("function") or {}
        name = fn.get("name")
        arguments = fn.get("arguments")
        if not name:
            continue
        tool_deltas.append(
            DeltaToolCall(
                index=i,
                type="function",
                id=call.get("id"),
                function=DeltaFunctionCall(
                    name=name,
                    arguments=arguments if arguments is not None else "",
                ),
            )
        )

    if not tool_deltas:
        return None

    normal_text = raw.get("normal_text")
    content = normal_text if normal_text else None
    return tool_deltas, content


def extract_gemma4_tool_calls_rust_sync(
    text: str,
    tools: list[dict[str, Any]] | None,
) -> tuple[list[DeltaToolCall], str | None] | None:
    """Run the async Rust parser from sync ``prepost`` code (may be inside a loop)."""
    import concurrent.futures

    def _run() -> tuple[list[DeltaToolCall], str | None] | None:
        return asyncio.run(extract_gemma4_tool_calls_rust(text, tools))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result(timeout=30)
    except Exception as exc:
        logger.warning("Gemma4 Rust fallback sync bridge failed: %s", exc)
        return None
