# Gemma 4 flavour of the Dynamo vLLM runtime image.
#
# The base image is model-agnostic, so serving Gemma 4 correctly means passing the same
# three flags on every launch. Dynamo reads each of them from an environment variable, so
# they can be baked in instead: a deployment then only needs `--model`.
#
#   docker build -f deploy/gemma4/Dockerfile \
#       --build-arg BASE_IMAGE=dynamo:gemma4-fix \
#       -t simplismart/simplismart-inference-server-base:higgs-vllm-gemma4 .
#
#   docker run --gpus all --network host --ipc host --shm-size 16g --user 0:0 \
#       -v /workspace/gemma4:/model:ro \
#       <image> python3 -m dynamo.vllm --model /model --served-model-name gemma4

ARG BASE_IMAGE=dynamo:gemma4-fix
FROM ${BASE_IMAGE}

# Gemma 4 emits `<|tool_call>call:name{...}<tool_call|>` and reasoning in a
# `<|channel>thought` block; these parsers are what turn that into OpenAI-shaped
# tool_calls and reasoning fields.
ENV DYN_TOOL_CALL_PARSER=gemma4 \
    DYN_REASONING_PARSER=gemma4

# Tool-use gating. With a JSON response format the model can now reach the tool channel,
# and a bare tool list reads to it as a menu: without this it calls a tool on turns that
# should simply speak (measured 3 of 4 runs on an intro turn). Stating that not calling a
# tool is the normal outcome fixed that 3/3.
# See ../../container/chat_templates/README.md. Override with DYN_CUSTOM_JINJA_TEMPLATE=""
# to fall back to the model's own template.
ENV DYN_CUSTOM_JINJA_TEMPLATE=/opt/dynamo/chat_templates/gemma4_tool_gated.jinja

# Gemma 4's card recommends temperature=1.0, top_p=0.95, top_k=64. Dynamo has no env var
# for those, and the model's own generation_config.json already carries them, so they are
# left to the model config rather than pinned here.

# VLLM_USE_PRECOMPILED is a *build-time* switch: it selects how vLLM was installed while
# this image was built and does nothing at runtime. Cleared so it cannot be mistaken for a
# runtime knob.
ENV VLLM_USE_PRECOMPILED=

LABEL org.opencontainers.image.description="Dynamo + vLLM runtime for Gemma 4, with \
structured output (json_schema/json_object) and native tool calling working in the same \
request. Tool-call parser, reasoning parser and the tool-gated chat template are baked in."
LABEL com.simplismart.gemma4.tool-gating="DYN_CUSTOM_JINJA_TEMPLATE=/opt/dynamo/chat_templates/gemma4_tool_gated.jinja"
