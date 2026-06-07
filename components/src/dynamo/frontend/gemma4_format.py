# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma4 content-cleaning helpers ported from vLLM gemma4_format.py."""

from __future__ import annotations

CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"
THOUGHT_PREFIX = "thought\n"
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'
CALL_PREFIX = "call:"

_EMPTY_THINKING_PATTERNS = (
    CHANNEL_START + THOUGHT_PREFIX + CHANNEL_END,
    CHANNEL_START + "thought\r\n" + CHANNEL_END,
)

_TH_WORD = "thought"
_LEN_TH = len(_TH_WORD)
_TH_PAIR = _TH_WORD * 2
_LEN_PAIR = len(_TH_PAIR)


def has_gemma4_tool_markup(text: str) -> bool:
    if TOOL_CALL_START in text or TOOL_CALL_END in text or STRING_DELIM in text:
        return True
    return _tool_call_markup_start(text) != -1


def _tool_call_markup_start(text: str) -> int:
    i = text.find(TOOL_CALL_START)
    if i != -1:
        return i
    search_from = 0
    channel_end = text.rfind(CHANNEL_END)
    if channel_end != -1:
        search_from = channel_end + len(CHANNEL_END)
    slice_text = text[search_from:]
    call_rel = slice_text.find(CALL_PREFIX)
    if call_rel == -1:
        return -1
    call_abs = search_from + call_rel
    if text.find("{", call_abs + len(CALL_PREFIX)) == -1:
        return -1
    return call_abs


def _compact_cf_no_ws(core: str) -> str:
    return "".join(core.casefold().split())


def _strip_one_thought_shard_line(core: str, nl: str) -> str:
    compact = _compact_cf_no_ws(core)
    if not compact:
        return core + nl

    limit = len(compact)
    i = 0
    while i + _LEN_PAIR <= limit and compact.startswith(_TH_PAIR, i):
        i += _LEN_PAIR
    c = compact[i:]
    lc = len(c)

    if lc == 0:
        return nl
    if c == _TH_WORD:
        return nl
    if lc < _LEN_TH:
        return nl if _TH_WORD.startswith(c) else core + nl
    if lc <= 5 and _TH_WORD.endswith(c) and c != _TH_WORD:
        return core + nl

    tail = c[_LEN_TH:]
    if (
        lc > _LEN_TH
        and tail
        and tail != _TH_WORD
        and c.startswith(_TH_WORD)
        and _TH_WORD.startswith(tail)
    ):
        return core + nl
    return core + nl


def strip_thought_shard_echoes(text: str) -> str:
    if not text:
        return text
    if "thought" not in text.casefold():
        return text
    if "\n" not in text and "\r" not in text:
        return _strip_one_thought_shard_line(text, "")

    rebuilt: list[str] = []
    for raw_line in text.splitlines(keepends=True):
        nl = ""
        core = raw_line
        if raw_line.endswith("\n"):
            nl = "\n"
            core = raw_line[:-1]
        rebuilt.append(_strip_one_thought_shard_line(core, nl))
    return "".join(rebuilt)


def strip_leaked_empty_thinking(text: str) -> str:
    if not text:
        return text
    if (
        CHANNEL_START not in text
        and CHANNEL_END not in text
        and "thought" not in text.casefold()
    ):
        return text
    s = text
    for pattern in _EMPTY_THINKING_PATTERNS:
        if pattern in s:
            s = s.replace(pattern, "")
    if CHANNEL_START in s or CHANNEL_END in s:
        while True:
            old = s
            s = s.replace(CHANNEL_START, "").replace(CHANNEL_END, "")
            s = s.strip()
            if s == old:
                break
    return strip_thought_shard_echoes(s)
