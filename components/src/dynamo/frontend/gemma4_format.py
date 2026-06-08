# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma4 content-cleaning helpers ported from vLLM gemma4_format.py."""

from __future__ import annotations

from dataclasses import dataclass

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

_INCOMPLETE_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        {
            tok[:i]
            for tok in (
                CHANNEL_START,
                CHANNEL_END,
                CHANNEL_START + THOUGHT_PREFIX,
                TOOL_CALL_START,
                TOOL_CALL_END,
                STRING_DELIM,
            )
            for i in range(1, len(tok))
        },
        key=len,
        reverse=True,
    )
)


@dataclass
class ReasoningSnapshot:
    reasoning: str | None = None
    content: str | None = None


class StreamingContentCleaner:
    """Stateful prefix-diff cleaner for pre-tool streaming content."""

    def __init__(self) -> None:
        self._clean_content_prefix = ""
        self._clean_content_previous_text: str | None = None

    def pre_tool_content_delta(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> str | None:
        same_stream = (
            self._clean_content_previous_text is not None
            and previous_text == self._clean_content_previous_text
        )
        if same_stream and previous_text:
            clean_prev = self._clean_content_prefix
        else:
            clean_prev = strip_leaked_empty_thinking(previous_text) or ""

        clean_curr = strip_tool_call_suffix(
            strip_leaked_empty_thinking(current_text) or ""
        )
        clean_prev = strip_tool_call_suffix(clean_prev)
        self._clean_content_prefix = clean_curr
        self._clean_content_previous_text = current_text

        if clean_curr.startswith(clean_prev):
            out = clean_curr[len(clean_prev) :]
            return out if out else None

        cleaned = strip_tool_call_suffix(strip_leaked_empty_thinking(delta_text))
        return cleaned if cleaned else None


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


def strip_tool_call_suffix(text: str) -> str:
    idx = _tool_call_markup_start(text)
    if idx != -1:
        text = text[:idx]
    return text.rstrip()


def extract_tool_handoff_text(text: str) -> str:
    idx = _tool_call_markup_start(text)
    return text[idx:] if idx != -1 else ""


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
        return nl

    tail = c[_LEN_TH:]
    if (
        lc > _LEN_TH
        and tail
        and tail != _TH_WORD
        and c.startswith(_TH_WORD)
        and _TH_WORD.startswith(tail)
    ):
        return nl
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


def _may_contain_gemma4_control_leak(text: str) -> bool:
    cf = text.casefold()
    return (
        CHANNEL_START in text
        or CHANNEL_END in text
        or "thought" in cf
        or TOOL_CALL_START in text
        or TOOL_CALL_END in text
        or STRING_DELIM in text
    )


def _strip_leaked_tool_grammar(text: str) -> str:
    s = text
    for tok in (TOOL_CALL_END, STRING_DELIM):
        if tok in s:
            s = s.replace(tok, "")
    return s


def strip_leaked_empty_thinking(text: str) -> str:
    """Remove echoed empty thinking channels and orphan control-token prefixes."""
    if not text:
        return text
    if not _may_contain_gemma4_control_leak(text):
        return text
    s = text

    composite = THOUGHT_PREFIX + CHANNEL_START + THOUGHT_PREFIX + CHANNEL_END
    if composite in s:
        s = s.replace(composite, "")

    for pattern in _EMPTY_THINKING_PATTERNS:
        if pattern in s:
            s = s.replace(pattern, "")

    while s.startswith(THOUGHT_PREFIX) and (
        CHANNEL_START in s[len(THOUGHT_PREFIX) :]
        or CHANNEL_END in s[len(THOUGHT_PREFIX) :]
    ):
        s = s[len(THOUGHT_PREFIX) :]

    orphan_close = THOUGHT_PREFIX + CHANNEL_END
    if s.startswith(orphan_close):
        s = s[len(orphan_close) :]

    if CHANNEL_START in s or CHANNEL_END in s:
        while True:
            old = s
            s = s.replace(CHANNEL_START, "").replace(CHANNEL_END, "")
            s = s.strip()
            if s == old:
                break

    s = _strip_leaked_tool_grammar(s)
    s = strip_thought_shard_echoes(s)
    return s.lstrip("\n").strip()


def strip_trailing_incomplete_token(text: str) -> str:
    if not text:
        return text
    longest = 0
    for pref in _INCOMPLETE_SUFFIXES:
        if text.endswith(pref) and len(pref) > longest:
            longest = len(pref)
    return text[:-longest] if longest else text


def _finalize_client_content(text: str | None) -> str | None:
    after = strip_leaked_empty_thinking(text) if text is not None else None
    if not after:
        return None
    after = strip_tool_call_suffix(after)
    return after if after else None


def strip_thought_label(text: str) -> str:
    if text.startswith(THOUGHT_PREFIX):
        return text[len(THOUGHT_PREFIX) :]
    return text


def extract_reasoning_non_streaming(model_output: str) -> ReasoningSnapshot:
    if CHANNEL_START not in model_output:
        return ReasoningSnapshot(content=_finalize_client_content(model_output))

    before_channel, _, after_channel = model_output.partition(CHANNEL_START)
    rest = after_channel
    if rest.startswith(THOUGHT_PREFIX):
        rest = rest[len(THOUGHT_PREFIX) :]

    if CHANNEL_END not in rest:
        if TOOL_CALL_START in rest:
            tool_idx = rest.find(TOOL_CALL_START)
            reasoning_text = rest[:tool_idx].strip()
            normal_text = (before_channel + rest[tool_idx:]).strip()
            return ReasoningSnapshot(
                reasoning=reasoning_text if reasoning_text else None,
                content=_finalize_client_content(normal_text),
            )
        reasoning_text = rest.strip()
        return ReasoningSnapshot(
            reasoning=reasoning_text if reasoning_text else None,
            content=None,
        )

    reason_part, _, content_part = rest.partition(CHANNEL_END)
    reasoning_text = reason_part.strip()
    merged_content = before_channel + content_part
    if not content_part:
        return ReasoningSnapshot(
            reasoning=reasoning_text if reasoning_text else "",
            content=None,
        )
    merged_trimmed = merged_content.strip()
    return ReasoningSnapshot(
        reasoning=reasoning_text if reasoning_text else "",
        content=_finalize_client_content(merged_trimmed if merged_trimmed else None),
    )


def diff_reasoning_streaming_snapshots(
    curr: ReasoningSnapshot,
    prev: ReasoningSnapshot,
) -> tuple[str, str]:
    nr = curr.reasoning or ""
    nc = curr.content or ""
    nr_prev = prev.reasoning or ""
    nc_prev = prev.content or ""
    dr = nr[len(nr_prev) :] if nr.startswith(nr_prev) else nr
    dc = nc[len(nc_prev) :] if nc.startswith(nc_prev) else nc
    return dr, dc
