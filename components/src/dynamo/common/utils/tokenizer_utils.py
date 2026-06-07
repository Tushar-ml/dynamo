# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for tokenizer / chat-template output normalization."""

from __future__ import annotations

from typing import Any


def normalize_prompt_token_ids(prompt_token_ids: Any) -> list[int]:
    """Flatten ``apply_chat_template`` output to ``list[int]``.

    Transformers v5 returns a ``BatchEncoding`` for ``tokenize=True``; older
    paths may return a bare list or a batch of lists.
    """
    ids = getattr(prompt_token_ids, "input_ids", prompt_token_ids)
    if isinstance(ids, dict):
        ids = ids.get("input_ids", prompt_token_ids)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list):
        raise TypeError(
            f"Expected token ids as a list, got {type(prompt_token_ids).__name__}"
        )
    return [int(x) for x in ids]
