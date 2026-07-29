# Deploying Gemma 4 on this image

Serves `google/gemma-4-*-it` through Dynamo + vLLM with tool calling that works alongside a
`json_schema` response format.

## Quick start

```bash
export MODEL_DIR=/workspace/gemma4          # plain directory holding the model files
docker compose up -d
curl localhost:8000/v1/models
```

Then the usual OpenAI calls against `http://localhost:8000/v1`.

## Direct docker run

```bash
# worker
docker run -d --name gemma4-worker --gpus all --network host --ipc host --shm-size 16g \
  --user 0:0 \
  -v /workspace/gemma4:/model:ro \
  -e ETCD_ENDPOINTS=http://127.0.0.1:2379 -e NATS_SERVER=nats://127.0.0.1:4222 \
  --entrypoint python3 <image> -m dynamo.vllm \
    --model /model --served-model-name gemma4 \
    --dyn-tool-call-parser gemma4 --dyn-reasoning-parser gemma4 \
    --custom-jinja-template /opt/dynamo/chat_templates/gemma4_tool_gated.jinja \
    --tensor-parallel-size 2 --max-model-len 64000 --max-num-seqs 8 \
    --gpu-memory-utilization 0.95

# frontend
docker run -d --name gemma4-frontend --network host --user 0:0 \
  -v /workspace/gemma4:/model:ro \
  -e ETCD_ENDPOINTS=http://127.0.0.1:2379 -e NATS_SERVER=nats://127.0.0.1:4222 \
  --entrypoint python3 <image> -m dynamo.frontend --http-port 8000
```

etcd and NATS must be reachable at those endpoints; the compose file starts them for you.

## Four things that will bite you otherwise

Each of these cost real debugging time, so they are worth knowing up front.

1. **`--user 0:0`.** The image runs as uid 1000. A model directory owned by any other uid
   gives `Permission denied` on the mount, and the failure surfaces as an unrelated
   "Failed to fetch model ... from HuggingFace" error.
2. **Mount a plain directory, not a HuggingFace cache snapshot path.** Discovery rejects
   the latter: `Model snapshots for '/root/.cache/huggingface/hub/models--…/snapshots/<rev>'
   not found in cache`. If you only have the cache layout, mount the whole
   `models--org--name` directory and pass `/model/snapshots/<revision>` — its relative
   symlinks into `../../blobs` then still resolve.
3. **The frontend needs the model mount too**, not just the worker: it renders the chat
   template. Without it you get an HTTP 404 on `/v1/chat/completions` while
   `/v1/models` looks fine.
4. **Clear stale etcd registrations between runs.** A worker killed without unregistering
   leaves a model card behind, and the frontend then flaps — registering the model and
   immediately removing it on the stale event, leaving `/v1/models` empty:

   ```bash
   docker compose down && docker volume rm <project>_etcd-data
   ```

## Tool calling with a JSON response format

The point of this image. A request may carry both `response_format`
(`json_schema` or `json_object`) and `tools`:

| request | behaviour |
|---|---|
| `tool_choice` absent/`auto` | schema-conforming object, a tool call, or both — the model chooses |
| `tool_choice: "none"` | schema-conforming object, guaranteed: no tool branch is compiled at all |
| `tool_choice: "required"` | exactly one native tool call |
| `tool_choice: {"type":"function",…}` | exactly one call, to that tool |

Two notes on using this well:

- **Speech-only turns.** With the tool channel reachable, the model may call a tool on a
  turn that should just speak. `--custom-jinja-template
  /opt/dynamo/chat_templates/gemma4_tool_gated.jinja` fixes that for all traffic by stating
  that not calling a tool is the normal outcome. For turns where a tool call is
  *definitionally* wrong, send `tool_choice: "none"`: that is enforced by the grammar
  rather than by persuasion, and it is the only hard guarantee.
- **The follow-up request.** After executing a tool, the request that carries
  `function_call_output` exists to turn the result into the response object. Send it with
  `tool_choice: "none"` so the model cannot chain another call.

## Speculative decoding

Add the MTP draft (`google/gemma-4-31B-it-assistant` — a draft model, not a servable
target):

```bash
  --speculative-config.model /workspace/draft_extra \
  --speculative-config.num_speculative_tokens 4
```

Verified working together with the structural-tag constraint.

## Verifying a deployment

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma4",
  "messages": [{"role": "user", "content": "hang up politely, we are done"}],
  "tools": [{"type": "function", "function": {
      "name": "hangup_call", "description": "End the call",
      "parameters": {"type": "object", "properties": {"message": {"type": "string"}},
                     "required": ["message"]}}}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "reply",
      "schema": {"type": "object", "properties": {"assistant_reply": {"type": "string"}},
                 "required": ["assistant_reply"]}, "strict": false}}
}' | jq '.choices[0].message | {content, tool_calls}'
```

A tool call in `tool_calls` (with `content` empty) is the fix working. Before it, the tool
call was swallowed into the JSON string and the object never terminated.
