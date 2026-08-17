# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.frontend.gemma4_format import (
    StreamingContentCleaner,
    has_gemma4_tool_markup,
    strip_leaked_empty_thinking,
    strip_thought_shard_echoes,
)


def test_strip_leaked_empty_thinking():
    text = "Hello <|channel>thought\n<channel|>world"
    assert strip_leaked_empty_thinking(text) == "Hello world"

    composite = "thought\n<|channel>thought\n<channel|>أعتذر جداً"
    assert strip_leaked_empty_thinking(composite) == "أعتذر جداً"

    orphan = "thought\n<channel|>أعتذر"
    assert strip_leaked_empty_thinking(orphan) == "أعتذر"

    assert (
        strip_leaked_empty_thinking("__thought\nAll getting sometime this")
        == "All getting sometime this"
    )
    assert (
        strip_leaked_empty_thinking(
            "__thought\n<|channel>thought\n<channel|>You're welcome."
        )
        == "You're welcome."
    )
    assert strip_leaked_empty_thinking("_________________\nHello there") == "Hello there"
    assert strip_leaked_empty_thinking("thought\n_________________\nHi") == "Hi"


def test_strip_thought_shard_echoes():
    assert strip_thought_shard_echoes("thoughtthought") == ""
    assert strip_thought_shard_echoes("thoughtthoughtful") == "thoughtthoughtful"
    assert strip_thought_shard_echoes("thoughtthoughtHello") == "thoughtthoughtHello"
    core = "thought" * 11 + "tho"
    assert strip_thought_shard_echoes(core).strip() == ""


def test_streaming_content_cleaner_prefix_diff():
    cleaner = StreamingContentCleaner()
    assert cleaner.pre_tool_content_delta("", "<|channel>", "<|channel>") is None
    out = cleaner.pre_tool_content_delta(
        "<|channel>",
        "<|channel>thought\n<channel|>Hi.",
        "thought\n<channel|>Hi.",
    )
    assert out == "Hi."


def test_has_gemma4_tool_markup():
    assert has_gemma4_tool_markup("<|tool_call>call:f{a:1}<tool_call|>")
    assert has_gemma4_tool_markup("call:get_weather{location:<|\"|>NYC<|\"|>}")
    assert not has_gemma4_tool_markup("plain answer")
