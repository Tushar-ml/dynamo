# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.frontend.gemma4_format import (
    has_gemma4_tool_markup,
    strip_leaked_empty_thinking,
    strip_thought_shard_echoes,
)


def test_strip_leaked_empty_thinking():
    text = "Hello <|channel>thought\n<channel|>world"
    assert strip_leaked_empty_thinking(text) == "Hello world"


def test_strip_thought_shard_echoes():
    assert strip_thought_shard_echoes("thoughtthought") == ""
    assert strip_thought_shard_echoes("thoughtthoughtful") == "thoughtful"


def test_has_gemma4_tool_markup():
    assert has_gemma4_tool_markup("<|tool_call>call:f{a:1}<tool_call|>")
    assert has_gemma4_tool_markup("call:get_weather{location:<|\"|>NYC<|\"|>}")
    assert not has_gemma4_tool_markup("plain answer")
