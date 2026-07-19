// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

use crate::engines::ValidateRequest;
use crate::preprocessor::media::MediaDecoder;

use super::{
    OpenAIOutputOptionsProvider, OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider,
    common_ext::{CommonExt, CommonExtProvider},
    nvext::NvExt,
    nvext::NvExtProvider,
    tools, validate,
};

pub mod aggregator;
mod delta;
pub mod jail;

pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

/// A request structure for creating a chat completion, extending OpenAI's
/// `CreateChatCompletionRequest` with [`NvExt`] extensions and common fields.
///
/// # Fields
/// - `inner`: The base OpenAI chat completion request, embedded using `serde(flatten)`.
/// - `common`: Common extension fields (ignore_eos, min_tokens) at root level, embedded using `serde(flatten)`.
/// - `nvext`: The optional NVIDIA extension field. See [`NvExt`] for more details.
///   Note: If ignore_eos is specified in both common and nvext, the common (root-level) value takes precedence.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionRequest {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::CreateChatCompletionRequest,

    #[serde(flatten, default)]
    pub common: CommonExt,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,

    /// Extra args to pass to the chat template rendering context
    /// Also accepts "chat_template_kwargs" as an alias for compatibility
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "chat_template_kwargs"
    )]
    pub chat_template_args: Option<std::collections::HashMap<String, serde_json::Value>>,

    /// Runtime media decoding parameters.
    /// When provided, these override the MDC defaults
    /// Example: `{"video": {"num_frames": 16}}`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_io_kwargs: Option<MediaDecoder>,

    /// When true, logprob token fields are returned as "token_id:<id>" instead
    /// of decoded text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_tokens_as_token_ids: Option<bool>,

    /// Catch-all for unsupported fields - checked during validation
    #[serde(flatten, default, skip_serializing)]
    pub unsupported_fields: std::collections::HashMap<String, serde_json::Value>,
}

/// A response structure for unary chat completion responses, embedding OpenAI's
/// `CreateChatCompletionResponse` with optional NVIDIA extension metadata.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NvCreateChatCompletionResponse {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateChatCompletionResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<serde_json::Value>,
}

/// A response structure for streamed chat completions, embedding OpenAI's
/// `CreateChatCompletionStreamResponse` with optional NVIDIA extension metadata.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NvCreateChatCompletionStreamResponse {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateChatCompletionStreamResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<serde_json::Value>,
}

/// Implements `NvExtProvider` for `NvCreateChatCompletionRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateChatCompletionRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    /// Returns `None`, as raw prompt extraction is not implemented.
    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

/// Implements `AnnotationsProvider` for `NvCreateChatCompletionRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

/// Implements `OpenAISamplingOptionsProvider` for `NvCreateChatCompletionRequest`,
/// exposing OpenAI's sampling parameters for chat completion.
impl OpenAISamplingOptionsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the temperature parameter for sampling, if set.
    fn get_temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    /// Retrieves the top-p (nucleus sampling) parameter, if set.
    fn get_top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    /// Retrieves the frequency penalty parameter, if set.
    fn get_frequency_penalty(&self) -> Option<f32> {
        self.inner.frequency_penalty
    }

    /// Retrieves the presence penalty parameter, if set.
    fn get_presence_penalty(&self) -> Option<f32> {
        self.inner.presence_penalty
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
    /// Retrieves the seed value for random number generation, if set.
    fn get_seed(&self) -> Option<i64> {
        self.inner.seed
    }

    /// Retrieves the number of completions to generate for each prompt, if set.
    fn get_n(&self) -> Option<u8> {
        self.inner.n
    }

    /// Retrieves the best_of parameter, if set.
    fn get_best_of(&self) -> Option<u8> {
        None // Not supported in chat completions
    }
}

/// Gemma-4 native tool-call argument grammar (GBNF/EBNF), transcribed from the
/// recursive-descent parser in `lib/parsers/src/tool_calling/gemma4/parser.rs`
/// (informal grammar at parser.rs:488-499). It constrains the bytes *between* the
/// `<|tool_call>call:NAME{` and `}<tool_call|>` markers: bare keys, `<|"|>`-delimited
/// strings, numbers, booleans, null/none/nil, and nested objects/arrays. It does not
/// enforce any individual tool's `parameters` schema (a JSON-Schema→EBNF transpiler
/// would be needed for that) — argument correctness is model-driven, exactly as in the
/// plain tool-calling path. The `string` rule forbids the `<|"|>` delimiter from
/// appearing in a body by disallowing `<` immediately followed by `|`.
const GEMMA4_ARGS_GRAMMAR: &str = r#"root    ::= (entry ("," entry)*)?
entry   ::= key ":" value
key     ::= [a-zA-Z0-9_.\-]+
value   ::= string | number | bool | null | object | array
string  ::= "<|\"|>" char* "<|\"|>"
char    ::= [^<] | "<" [^|]
number  ::= "-"? [0-9]+ ("." [0-9]+)?
bool    ::= "true" | "false"
null    ::= "null" | "none" | "nil"
object  ::= "{" (entry ("," entry)*)? "}"
array   ::= "[" (value ("," value)*)? "]""#;

/// Builds the xgrammar structural tag described on `get_structural_tag`: a top-level
/// `OrFormat` whose two alternatives are the `response_format` schema (as a
/// `json_schema` element) and a `tags_with_separator` of one native-tool-call tag per
/// declared tool. With no `any_text` element anywhere, the schema branch stays a hard
/// guarantee. Shape verified to compile and match correctly against xgrammar 0.2.0.
fn build_gemma4_structural_tag(
    response_schema: &serde_json::Value,
    tools: &[dynamo_protocols::types::ChatCompletionTool],
) -> serde_json::Value {
    let tags: Vec<serde_json::Value> = tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "tag",
                "begin": format!("<|tool_call>call:{}{{", tool.function.name),
                "content": {"type": "grammar", "grammar": GEMMA4_ARGS_GRAMMAR},
                "end": "}<tool_call|>",
            })
        })
        .collect();

    serde_json::json!({
        "type": "structural_tag",
        "format": {
            "type": "or",
            "elements": [
                {"type": "json_schema", "json_schema": response_schema},
                {
                    "type": "tags_with_separator",
                    "separator": "",
                    "at_least_one": true,
                    "tags": tags,
                },
            ],
        },
    })
}

impl NvCreateChatCompletionRequest {
    /// Returns the `response_format` schema to enforce *iff* this request is the
    /// structural-tag combo: non-empty `tools`, a non-forced `tool_choice`
    /// (auto/none/unset — not required/named), and a JSON `response_format`. `None`
    /// otherwise. Drives both `get_guided_json` (returns `None` to yield to the tag)
    /// and `get_structural_tag` (builds the tag), so the two can never both fire.
    fn structural_tag_response_schema(&self) -> Option<serde_json::Value> {
        use dynamo_protocols::types::{ChatCompletionToolChoiceOption, ResponseFormat};

        let has_tools = self
            .inner
            .tools
            .as_deref()
            .is_some_and(|tools| !tools.is_empty());
        if !has_tools {
            return None;
        }
        if matches!(
            self.inner.tool_choice,
            Some(ChatCompletionToolChoiceOption::Required)
                | Some(ChatCompletionToolChoiceOption::Named(_))
        ) {
            return None;
        }

        match self.inner.response_format.as_ref()? {
            ResponseFormat::Text => None,
            ResponseFormat::JsonObject => Some(serde_json::json!({"type": "object"})),
            ResponseFormat::JsonSchema { json_schema } => json_schema.schema.clone(),
        }
    }
}

/// Implements `CommonExtProvider` for `NvCreateChatCompletionRequest`,
/// providing access to common extension fields.
impl CommonExtProvider for NvCreateChatCompletionRequest {
    /// Returns a reference to the CommonExt struct.
    fn common_ext(&self) -> Option<&CommonExt> {
        Some(&self.common)
    }

    /// Guided Decoding Options
    fn get_guided_json(&self) -> Option<serde_json::Value> {
        if let Some(value) = self.common.guided_json.clone() {
            return Some(value);
        }

        // 1) Tool-call guided decoding (highest precedence after explicit guided_json)
        if let (Some(tool_choice), Some(tools)) =
            (self.inner.tool_choice.as_ref(), self.inner.tools.as_deref())
        {
            match tools::get_json_schema_from_tools(Some(tool_choice), Some(tools)) {
                Ok(Some(schema)) => return Some(schema),
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "failed to derive guided_json from tool_choice"
                    );
                }
            }
        }

        // 1b) `tools` offered with a non-forced tool_choice (auto/none/unset) plus a
        // `response_format` schema is handled by a structural tag (see
        // `get_structural_tag`), not by `guided_json`. That tag hard-constrains
        // decoding to *either* a native Gemma-4 tool call *or* schema-valid content,
        // keeping both guarantees. Return `None` here so `guided_json` stays unset —
        // it is mutually exclusive with `structural_tag`
        // (`common::GuidedDecodingOptions::validate`), and the response schema below
        // (step 2) would otherwise re-foreclose tool calls (the original bug).
        if self.structural_tag_response_schema().is_some() {
            return None;
        }

        // 2) OpenAI `response_format` (applies to assistant content, not tool calls)
        if let Some(response_format) = self.inner.response_format.as_ref() {
            use dynamo_protocols::types::ResponseFormat;
            match response_format {
                ResponseFormat::Text => {}
                ResponseFormat::JsonObject => {
                    // Minimal JSON Schema for "any JSON object"
                    return Some(serde_json::json!({
                        "type": "object"
                    }));
                }
                ResponseFormat::JsonSchema { json_schema } => {
                    // validate_response_format ensures schema is present when type=json_schema
                    if let Some(schema) = json_schema.schema.clone() {
                        return Some(schema);
                    }
                }
            }
        }

        None
    }

    fn get_guided_regex(&self) -> Option<String> {
        self.common.guided_regex.clone()
    }

    fn get_guided_grammar(&self) -> Option<String> {
        self.common.guided_grammar.clone()
    }

    fn get_guided_choice(&self) -> Option<Vec<String>> {
        self.common.guided_choice.clone()
    }

    /// Builds an xgrammar structural tag for the `tools` + non-forced `tool_choice`
    /// + `response_format` combo, so decoding is hard-constrained to *either* a
    /// native Gemma-4 tool call *or* schema-valid content — never free text, and
    /// never one foreclosing the other. Returns `None` for every other request, in
    /// which case the ordinary `guided_json`/`guided_grammar` path applies.
    fn get_structural_tag(&self) -> Option<serde_json::Value> {
        let response_schema = self.structural_tag_response_schema()?;
        // `structural_tag_response_schema` already guaranteed non-empty tools.
        let tools = self.inner.tools.as_deref()?;
        Some(build_gemma4_structural_tag(&response_schema, tools))
    }

    fn get_guided_decoding_backend(&self) -> Option<String> {
        self.common.guided_decoding_backend.clone()
    }

    fn get_guided_whitespace_pattern(&self) -> Option<String> {
        self.common.guided_whitespace_pattern.clone()
    }

    fn get_top_k(&self) -> Option<i32> {
        self.common.top_k
    }

    fn get_min_p(&self) -> Option<f32> {
        self.common.min_p
    }

    fn get_repetition_penalty(&self) -> Option<f32> {
        self.common.repetition_penalty
    }

    fn get_include_stop_str_in_output(&self) -> Option<bool> {
        self.common.include_stop_str_in_output
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        self.common.skip_special_tokens
    }
}

/// Implements `OpenAIStopConditionsProvider` for `NvCreateChatCompletionRequest`,
/// providing access to stop conditions that control chat completion behavior.
impl OpenAIStopConditionsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the maximum number of tokens allowed in the response.
    #[allow(deprecated)]
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner.max_completion_tokens.or(self.inner.max_tokens)
    }

    /// Retrieves the minimum number of tokens required in the response.
    /// Returns `min_tokens` Value
    /// `min_tokens` is not an OpenAI-supported parameter.
    fn get_min_tokens(&self) -> Option<u32> {
        self.common.min_tokens
    }

    /// Retrieves the stop conditions that terminate the chat completion response.
    ///
    /// Converts OpenAI's `Stop` enum to a `Vec<String>`, normalizing the representation.
    ///
    /// # Returns
    /// * `Some(Vec<String>)` if stop conditions are set.
    /// * `None` if no stop conditions are defined.
    fn get_stop(&self) -> Option<Vec<String>> {
        self.inner.stop.as_ref().and_then(|stop| stop.strings())
    }

    fn get_stop_token_ids(&self) -> Option<Vec<crate::types::TokenIdType>> {
        self.inner.stop.as_ref().and_then(|stop| stop.token_ids())
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    /// Get ignore_eos from CommonExt.
    fn get_common_ignore_eos(&self) -> Option<bool> {
        self.common.ignore_eos
    }

    /// Get the effective ignore_eos value from CommonExt.
    fn get_ignore_eos(&self) -> Option<bool> {
        self.common.ignore_eos
    }
}

impl OpenAIOutputOptionsProvider for NvCreateChatCompletionRequest {
    fn get_logprobs(&self) -> Option<u32> {
        match self.inner.logprobs {
            Some(true) => match self.inner.top_logprobs {
                Some(top_logprobs) => Some(top_logprobs as u32),
                None => Some(1_u32),
            },
            Some(false) => None,
            None => None,
        }
    }

    fn get_prompt_logprobs(&self) -> Option<u32> {
        None
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        CommonExtProvider::get_skip_special_tokens(self)
    }

    fn get_formatted_prompt(&self) -> Option<bool> {
        None
    }

    fn get_return_tokens_as_token_ids(&self) -> Option<bool> {
        self.return_tokens_as_token_ids
    }
}

/// Implements `ValidateRequest` for `NvCreateChatCompletionRequest`,
/// allowing us to validate the data.
impl ValidateRequest for NvCreateChatCompletionRequest {
    fn validate(&self) -> Result<(), anyhow::Error> {
        validate::validate_no_unsupported_fields(&self.unsupported_fields)?;
        validate::validate_messages(&self.inner.messages)?;
        validate::validate_model(&self.inner.model)?;
        // none for store
        validate::validate_reasoning_effort(&self.inner.reasoning_effort)?;
        // none for metadata
        validate::validate_frequency_penalty(self.inner.frequency_penalty)?;
        validate::validate_logit_bias(&self.inner.logit_bias)?;
        // none for logprobs
        validate::validate_top_logprobs(self.inner.top_logprobs)?;
        // validate::validate_max_tokens(self.inner.max_tokens)?; // warning depricated field
        validate::validate_max_completion_tokens(self.inner.max_completion_tokens)?;
        validate::validate_n(self.inner.n)?;
        // none for modalities
        // none for prediction
        // none for audio
        validate::validate_presence_penalty(self.inner.presence_penalty)?;
        validate::validate_response_format(&self.inner.response_format)?;
        // none for seed
        validate::validate_service_tier(&self.inner.service_tier)?;
        validate::validate_stop(&self.inner.stop)?;
        // none for stream
        // none for stream_options
        validate::validate_temperature(self.inner.temperature)?;
        validate::validate_top_p(self.inner.top_p)?;
        validate::validate_tools(&self.inner.tools.as_deref())?;
        // none for tool_choice
        // none for parallel_tool_calls
        validate::validate_user(self.inner.user.as_deref())?;
        // none for function call
        // none for functions
        // Common Ext
        validate::validate_repetition_penalty(self.get_repetition_penalty())?;
        validate::validate_min_p(self.get_min_p())?;
        validate::validate_top_k(self.get_top_k())?;
        // Cross-field validation
        validate::validate_n_with_temperature(self.inner.n, self.inner.temperature)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::ValidateRequest;
    use crate::protocols::common::{OutputOptionsProvider, StopConditionsProvider};
    use serde_json::json;

    #[test]
    fn test_skip_special_tokens_none() {
        let json_str = json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        assert_eq!(request.common.skip_special_tokens, None);

        let output_options = request
            .extract_output_options()
            .expect("Failed to extract output options");

        assert_eq!(output_options.skip_special_tokens, None);
    }

    #[test]
    fn test_skip_special_tokens_propagates() {
        for skip_value in [true, false] {
            let json_str = json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "skip_special_tokens": skip_value
            });

            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(json_str).expect("Failed to deserialize request");

            let output_options = request
                .extract_output_options()
                .expect("Failed to extract output options");

            assert_eq!(output_options.skip_special_tokens, Some(skip_value));
        }
    }

    #[test]
    fn test_stop_contract() {
        let one_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": " The"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(one_stop).expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec![" The".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let many_stops = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["A", "B"]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(many_stops).expect("Failed to deserialize request");
        assert_eq!(
            request.get_stop(),
            Some(vec!["A".to_string(), "B".to_string()])
        );
        assert_eq!(request.get_stop_token_ids(), None);

        let token_id_stops = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": [32, 34]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(token_id_stops).expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), None);
        assert_eq!(request.get_stop_token_ids(), Some(vec![32, 34]));

        let stop_conditions = request
            .extract_stop_conditions()
            .expect("extract stop conditions");
        assert_eq!(stop_conditions.stop, None);
        assert_eq!(stop_conditions.stop_token_ids, Some(vec![32, 34]));

        let token_id_display_string_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": "token_id:576"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(token_id_display_string_stop)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec!["token_id:576".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let token_id_display_string_array_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["token_id:576"]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(token_id_display_string_array_stop)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec!["token_id:576".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let scalar_token_id_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": 576
        });
        let result: Result<NvCreateChatCompletionRequest, _> =
            serde_json::from_value(scalar_token_id_stop);
        assert!(result.is_err());

        let unsupported_stop_token_ids = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop_token_ids": [576]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(unsupported_stop_token_ids)
                .expect("Failed to deserialize request");
        assert!(ValidateRequest::validate(&request).is_err());
    }

    // --- structural tag for tools + response_format (task4 / SquadStack) ---

    fn weather_tool() -> serde_json::Value {
        json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        })
    }

    fn response_format_schema() -> serde_json::Value {
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": "reply",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        })
    }

    fn request_from(value: serde_json::Value) -> NvCreateChatCompletionRequest {
        serde_json::from_value(value).expect("Failed to deserialize request")
    }

    #[test]
    fn structural_tag_built_for_tools_plus_response_format() {
        let req = request_from(json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": "weather in Bengaluru?"}],
            "tools": [weather_tool()],
            "response_format": response_format_schema(),
        }));

        // The combo yields a structural tag, and NOT a guided_json (mutually
        // exclusive — see common::GuidedDecodingOptions::validate).
        assert!(req.get_guided_json().is_none());
        let tag = req.get_structural_tag().expect("structural tag expected");
        assert_eq!(tag["type"], "structural_tag");
        assert_eq!(tag["format"]["type"], "or");
        let elements = tag["format"]["elements"].as_array().unwrap();
        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0]["type"], "json_schema");
        assert_eq!(elements[1]["type"], "tags_with_separator");
        assert_eq!(elements[1]["at_least_one"], true);
        let first_tag = &elements[1]["tags"][0];
        assert_eq!(first_tag["begin"], "<|tool_call>call:get_weather{");
        assert_eq!(first_tag["end"], "}<tool_call|>");
        assert_eq!(first_tag["content"]["type"], "grammar");
    }

    #[test]
    fn no_structural_tag_without_response_format() {
        let req = request_from(json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [weather_tool()],
        }));
        assert!(req.get_structural_tag().is_none());
    }

    #[test]
    fn no_structural_tag_without_tools() {
        let req = request_from(json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": response_format_schema(),
        }));
        assert!(req.get_structural_tag().is_none());
        // Plain response_format still drives guided_json as before.
        assert!(req.get_guided_json().is_some());
    }

    #[test]
    fn no_structural_tag_for_required_tool_choice() {
        let req = request_from(json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [weather_tool()],
            "tool_choice": "required",
            "response_format": response_format_schema(),
        }));
        // Forced tool_choice keeps its own tool-schema guided_json; no structural tag.
        assert!(req.get_structural_tag().is_none());
        assert!(req.get_guided_json().is_some());
    }
}
