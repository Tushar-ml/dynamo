# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for decode-worker prefix warmup."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.vllm.constants import DisaggregationMode
from dynamo.common.utils.tokenizer_utils import normalize_prompt_token_ids
from dynamo.vllm.prefix_warmup import (
    _load_prefix_warmup_bytes,
    _prefix_warmup_settings,
    _validate_prefix_warmup_entry,
    run_decode_prefix_warmup,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _make_config(**overrides):
    defaults = {
        "disaggregation_mode": DisaggregationMode.DECODE,
        "prefix_warmup_file": None,
        "prefix_warmup_count": None,
        "prefix_warmup_parallel": False,
        "frontend_args": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_prefix_warmup_settings_from_dynamo_args():
    config = _make_config(
        prefix_warmup_file="/dynamo.json",
        prefix_warmup_count=2,
        prefix_warmup_parallel=True,
    )
    assert _prefix_warmup_settings(config) == ("/dynamo.json", 2, True)


def test_validate_prefix_warmup_entry_requires_messages():
    with pytest.raises(RuntimeError, match="non-empty 'messages'"):
        _validate_prefix_warmup_entry({"messages": []}, 0)


def test_load_prefix_warmup_bytes_local(tmp_path):
    path = tmp_path / "warmup.json"
    path.write_text("[]")
    assert _load_prefix_warmup_bytes(str(path)) == b"[]"


@pytest.mark.asyncio
async def test_run_decode_prefix_warmup_skips_without_file():
    engine_client = Mock()
    await run_decode_prefix_warmup(
        engine_client,
        Mock(),
        _make_config(),
        {},
    )
    engine_client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_run_decode_prefix_warmup_skips_on_prefill_worker(tmp_path):
    path = tmp_path / "warmup.json"
    path.write_text(
        json.dumps([{"messages": [{"role": "user", "content": "hi"}]}])
    )
    engine_client = Mock()
    config = _make_config(
        disaggregation_mode=DisaggregationMode.PREFILL,
        prefix_warmup_file=str(path),
    )
    await run_decode_prefix_warmup(engine_client, Mock(), config, {})
    engine_client.generate.assert_not_called()


def test_normalize_prompt_token_ids_batch_encoding():
    class BatchEncoding(dict):
        pass

    enc = BatchEncoding(input_ids=[10, 11, 12])
    assert normalize_prompt_token_ids(enc) == [10, 11, 12]


def test_normalize_prompt_token_ids_dict():
    assert normalize_prompt_token_ids({"input_ids": [1, 2]}) == [1, 2]


@pytest.mark.asyncio
async def test_run_decode_prefix_warmup_runs_engine_generate(tmp_path):
    path = tmp_path / "warmup.json"
    path.write_text(
        json.dumps(
            [
                {"messages": [{"role": "user", "content": "hello"}]},
                {"messages": [{"role": "user", "content": "world"}]},
            ]
        )
    )

    class _Output:
        finish_reason = "stop"

    class _Result:
        outputs = [_Output()]

        class _Usage:
            num_prompt_tokens = 12

        usage = _Usage()

    async def _fake_generate(*_args, **_kwargs):
        yield _Result()

    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = [1, 2, 3]

    engine_client = Mock()
    engine_client.tokenizer = tokenizer
    engine_client.generate = _fake_generate

    vllm_config = Mock()
    vllm_config.parallel_config.data_parallel_external_lb = True
    vllm_config.parallel_config.data_parallel_rank = 0

    config = _make_config(
        prefix_warmup_file=str(path),
        prefix_warmup_count=1,
    )

    await run_decode_prefix_warmup(engine_client, vllm_config, config, {})
