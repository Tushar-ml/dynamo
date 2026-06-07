// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Content-cleaning helpers ported from vLLM's gemma4_format.py (SGLang-aligned).

use super::parser::{CALL_PREFIX, STRING_DELIM, TOOL_CALL_END, TOOL_CALL_START};

pub(crate) const CHANNEL_START: &str = "<|channel>";
pub(crate) const CHANNEL_END: &str = "<channel|>";
pub(crate) const THOUGHT_PREFIX: &str = "thought\n";

const TH_WORD: &str = "thought";
const LEN_TH: usize = TH_WORD.len();
const TH_PAIR: &str = "thoughtthought";
const LEN_PAIR: usize = TH_PAIR.len();

const EMPTY_THINKING_PATTERNS: &[&str] = &[
    "<|channel>thought\n<channel|>",
    "<|channel>thought\r\n<channel|>",
];

/// Index where Gemma4 tool-call markup begins, or `None`.
pub(crate) fn tool_call_markup_start(text: &str) -> Option<usize> {
    if let Some(i) = text.find(TOOL_CALL_START) {
        return Some(i);
    }
    let search_from = text
        .rfind(CHANNEL_END)
        .map(|i| i + CHANNEL_END.len())
        .unwrap_or(0);
    let slice_text = &text[search_from..];
    let call_rel = slice_text.find(CALL_PREFIX)?;
    let call_abs = search_from + call_rel;
    text[call_abs + CALL_PREFIX.len()..].find('{').map(|_| call_abs)
}

/// True if *text* contains Gemma4 tool-call delimiters or a bare `call:fn{` block.
pub(crate) fn has_tool_call_markup(text: &str) -> bool {
    text.contains(TOOL_CALL_START)
        || text.contains(TOOL_CALL_END)
        || text.contains(STRING_DELIM)
        || tool_call_markup_start(text).is_some()
}

/// Remove trailing tool-call markup so it is not emitted as client `content`.
pub(crate) fn strip_tool_call_suffix(text: &str) -> String {
    if let Some(idx) = tool_call_markup_start(text) {
        text[..idx].trim_end().to_string()
    } else {
        text.trim_end().to_string()
    }
}

fn compact_cf_no_ws(core: &str) -> String {
    core.to_lowercase().split_whitespace().collect()
}

fn strip_one_thought_shard_line(core: &str, nl: &str) -> String {
    let compact = compact_cf_no_ws(core);
    if compact.is_empty() {
        return format!("{core}{nl}");
    }

    let mut i = 0usize;
    let limit = compact.len();
    while i + LEN_PAIR <= limit && compact[i..].starts_with(TH_PAIR) {
        i += LEN_PAIR;
    }
    let c = &compact[i..];
    let lc = c.len();

    if lc == 0 {
        return nl.to_string();
    }
    if c == TH_WORD {
        return nl.to_string();
    }
    if lc < LEN_TH {
        return if TH_WORD.starts_with(c) {
            nl.to_string()
        } else {
            format!("{core}{nl}")
        };
    }
    if lc <= 5 && TH_WORD.ends_with(c) && c != TH_WORD {
        return format!("{core}{nl}");
    }

    let tail = &c[LEN_TH..];
    if lc > LEN_TH
        && !tail.is_empty()
        && tail != TH_WORD
        && c.starts_with(TH_WORD)
        && TH_WORD.starts_with(tail)
    {
        return format!("{core}{nl}");
    }

    format!("{core}{nl}")
}

/// Strip glued `thought` shards from plain `content` (streaming cut artifacts).
pub(crate) fn strip_thought_shard_echoes(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let cf_full = text.to_lowercase();
    if !cf_full.contains("thought") {
        return text.to_string();
    }

    if !text.contains('\n') && !text.contains('\r') {
        return strip_one_thought_shard_line(text, "");
    }

    let mut rebuilt = String::new();
    for raw_line in text.split_inclusive('\n') {
        let (core, nl) = if let Some(stripped) = raw_line.strip_suffix('\n') {
            (stripped, "\n")
        } else {
            (raw_line, "")
        };
        rebuilt.push_str(&strip_one_thought_shard_line(core, nl));
    }
    rebuilt
}

/// Remove echoed empty thinking channels (Gemma4 "suppress CoT" pattern).
pub(crate) fn strip_leaked_empty_thinking(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    if !text.contains(CHANNEL_START)
        && !text.contains(CHANNEL_END)
        && !text.to_lowercase().contains("thought")
    {
        return text.to_string();
    }

    let mut s = text.to_string();
    for pattern in EMPTY_THINKING_PATTERNS {
        if s.contains(pattern) {
            s = s.replace(pattern, "");
        }
    }

    if s.contains(CHANNEL_START) || s.contains(CHANNEL_END) {
        loop {
            let old = s.clone();
            s = s.replace(CHANNEL_START, "").replace(CHANNEL_END, "");
            s = s.trim().to_string();
            if s == old {
                break;
            }
        }
    }

    strip_thought_shard_echoes(&s)
}

/// Client-visible content pipeline: empty-thinking suppression + tool-suffix strip.
pub(crate) fn finalize_client_content(text: &str) -> Option<String> {
    let after = strip_leaked_empty_thinking(text);
    if after.is_empty() {
        return None;
    }
    let after = strip_tool_call_suffix(&after);
    if after.is_empty() {
        None
    } else {
        Some(after)
    }
}

/// Clean prose prefix without stripping downstream tool-call markup.
pub(crate) fn clean_visible_prefix(text: &str) -> String {
    strip_leaked_empty_thinking(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_leaked_empty_thinking_removes_echoed_channel() {
        let input = "Hello <|channel>thought\n<channel|>world";
        assert_eq!(strip_leaked_empty_thinking(input), "Hello world");
    }

    #[test]
    fn strip_thought_shard_echoes_removes_glued_shards() {
        assert_eq!(strip_thought_shard_echoes("thoughtthought"), "");
        // Genuine English prefix survives (vLLM parity).
        assert_eq!(
            strip_thought_shard_echoes("thoughtthoughtful"),
            "thoughtthoughtful"
        );
    }

    #[test]
    fn strip_tool_call_suffix_keeps_prefix() {
        let input = "I'll help!<|tool_call>call:f{}<tool_call|>";
        assert_eq!(strip_tool_call_suffix(input), "I'll help!");
    }

    #[test]
    fn finalize_client_content_strips_empty_thinking_and_tool_suffix() {
        let input = "Hi <|channel>thought\n<channel|>there<|tool_call>call:f{}<tool_call|>";
        assert_eq!(finalize_client_content(input).as_deref(), Some("Hi there"));
    }
}
