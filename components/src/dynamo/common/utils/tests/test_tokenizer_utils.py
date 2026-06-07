# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.utils.tokenizer_utils import normalize_prompt_token_ids

pytestmark = pytest.mark.unit


def test_normalize_prompt_token_ids_plain_list():
    assert normalize_prompt_token_ids([1, 2, 3]) == [1, 2, 3]


def test_normalize_prompt_token_ids_batch_encoding_dict():
    class BatchEncoding(dict):
        pass

    enc = BatchEncoding(input_ids=[10, 11, 12])
    assert normalize_prompt_token_ids(enc) == [10, 11, 12]


def test_normalize_prompt_token_ids_nested_dict():
    assert normalize_prompt_token_ids({"input_ids": [4, 5]}) == [4, 5]


def test_normalize_prompt_token_ids_batched_list():
    assert normalize_prompt_token_ids([[7, 8], [9, 10]]) == [7, 8]


def test_normalize_prompt_token_ids_rejects_invalid_type():
    with pytest.raises(TypeError, match="Expected token ids"):
        normalize_prompt_token_ids(42)
