# Chat templates

## `gemma4_tool_gated.jinja`

`google/gemma-4-*-it`'s own chat template with one addition: a tool-use policy emitted
immediately before the tool declarations, for requests that carry `tools`.

### Why

A bare tool list reads to the model as a menu it is expected to use. With a
`json_schema` `response_format` in play this was masked — a tool call was impossible, so
every turn produced the JSON object. Once the structural-tag fix makes tool calls
reachable, the model's real preference shows: on `gemma-4-31B-it` it called
`fetch_seller_details` on an intro turn that should simply have replied, 3 of 4 runs.

Stating the default fixes it. Measured with `--custom-jinja-template`, 31B, TP=2,
streamed, 3 runs of a 3-scenario acceptance script:

| | non-tool turn | tool turn A | tool turn B (2-call) |
|---|---|---|---|
| stock dynamo | 3/3 | 0/3 | 0/3 |
| structural-tag fix | 0/3 | 3/3 | 3/3 |
| fix + this template | **3/3** | **3/3** | **3/3** |

### Use

```bash
python3 -m dynamo.vllm --model <gemma-4> \
    --dyn-tool-call-parser gemma4 --dyn-reasoning-parser gemma4 \
    --custom-jinja-template /workspace/chat_templates/gemma4_tool_gated.jinja
```

### Scope

This is persuasion, not enforcement: it improves the model's judgment on turns the
server cannot reason about. Where a tool call is *definitionally* wrong for a turn, send
`tool_choice: "none"` for that request — the server then compiles no tool branch at all,
which is a hard guarantee. The two are complementary: template for everything,
`tool_choice` for the turns the client knows about.

Regenerate after a model template update by re-applying the same insertion to the new
`chat_template.jinja`, immediately before the `{%- for tool in tools %}` loop.
