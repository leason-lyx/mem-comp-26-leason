#!/usr/bin/env bash
set -euo pipefail

# Quick health check llm endpoint + tool/parallel_tool_calls capability probe.

BASE_URL=${QWEN_BASE_URL:-"http://100.124.83.129:46400/v1"}
API_KEY=${QWEN_API_KEY:-"api_key"}
MODEL=${QWEN_MODEL:-"Qwen3-Coder-Next"}

# BASE_URL=${QWEN_BASE_URL:-"https://api.deepseek.com"}
# API_KEY=${QWEN_API_KEY:-"sk-9926cbaed4664f43be23befb17a11a9f"}
# MODEL=${QWEN_MODEL:-"deepseek-chat"}

echo "POST ${BASE_URL}/chat/completions with model=${MODEL}"

curl_json() {
  local payload="$1"
  curl -sS --fail \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "${BASE_URL}/chat/completions"
}

require_jq() {
  if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq not found. Please install jq for capability probing."
    exit 1
  fi
}

# -----------------------------
# 0) Basic ping
# -----------------------------
payload_ping=$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [
    {"role": "user", "content": "ping"}
  ],
  "max_tokens": 128
}
EOF
)

resp_ping=$(curl_json "$payload_ping")

if command -v jq >/dev/null 2>&1; then
  echo "$resp_ping" | jq -e '.choices[0].message.content' >/dev/null
  echo "Qwen3 response:" "$(echo "$resp_ping" | jq -r '.choices[0].message.content')"
else
  echo "Raw response (jq not found):"
  echo "$resp_ping"
fi

# -----------------------------
# Tool definitions for probing
# -----------------------------
# Two dummy tools; we only care whether the model returns tool_calls in response.
tools_json=$(cat <<'EOF'
[
  {
    "type": "function",
    "function": {
      "name": "tool_a",
      "description": "Dummy tool A for capability probing. Do not execute anything.",
      "parameters": {
        "type": "object",
        "properties": {
          "x": {"type": "string", "description": "Any string"}
        },
        "required": ["x"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tool_b",
      "description": "Dummy tool B for capability probing. Do not execute anything.",
      "parameters": {
        "type": "object",
        "properties": {
          "y": {"type": "string", "description": "Any string"}
        },
        "required": ["y"]
      }
    }
  }
]
EOF
)

require_jq

# Helper: detect tool_calls count in first assistant message
tool_calls_count() {
  local resp="$1"
  # New format: message.tool_calls (array)
  # Fallback: message.function_call (object) => count as 1
  echo "$resp" | jq -r '
    ( .choices[0].message.tool_calls // empty ) as $tc
    | if ($tc|type) == "array" then ($tc|length)
      else ( .choices[0].message.function_call // empty ) as $fc
      | if ($fc|type) == "object" then 1 else 0 end
      end
  '
}

# Helper: list tool names (for debug)
tool_calls_names() {
  local resp="$1"
  echo "$resp" | jq -r '
    if (.choices[0].message.tool_calls|type)=="array" then
      .choices[0].message.tool_calls[].function.name
    elif (.choices[0].message.function_call|type)=="object" then
      .choices[0].message.function_call.name
    else
      empty
    end
  '
}

echo ""
echo "== Tool calling probe =="

# -----------------------------
# 1) Tool-calling capability probe
# -----------------------------
payload_tool_single=$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [
    {
      "role": "user",
      "content": "You MUST call tool_a exactly once. Do not answer normally. Return a tool call."
    }
  ],
  "tools": $tools_json,
  "tool_choice": {"type": "function", "function": {"name": "tool_a"}},
  "max_tokens": 128
}
EOF
)

resp_tool_single=$(curl_json "$payload_tool_single")
cnt_single=$(tool_calls_count "$resp_tool_single")

if [[ "$cnt_single" -ge 1 ]]; then
  echo "✅ Tool calling supported (detected ${cnt_single} call(s)):"
  echo "$resp_tool_single" | jq -c '.choices[0].message.tool_calls // .choices[0].message.function_call'
else
  echo "❌ Tool calling NOT detected. Full response:"
  echo "$resp_tool_single" | jq .
fi

echo ""
echo "== parallel_tool_calls probe =="

# -----------------------------
# 2) parallel_tool_calls capability probe
# -----------------------------
# We ask for two tool calls in ONE assistant message.
payload_parallel=$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [
    {
      "role": "user",
      "content": "Call tool_a and tool_b in parallel in a SINGLE assistant message. Do NOT call one after the other; output BOTH tool calls at once. Do not provide normal text."
    }
  ],
  "tools": $tools_json,
  "parallel_tool_calls": true,
  "tool_choice": "auto",
  "max_tokens": 128
}
EOF
)

resp_parallel=$(curl_json "$payload_parallel")
cnt_parallel=$(tool_calls_count "$resp_parallel")

if [[ "$cnt_parallel" -ge 2 ]]; then
  echo "✅ parallel_tool_calls appears supported (detected ${cnt_parallel} tool calls in one message)."
  echo "Tools called:"
  tool_calls_names "$resp_parallel" | sed 's/^/  - /'
  echo "Raw tool_calls:"
  echo "$resp_parallel" | jq -c '.choices[0].message.tool_calls'
else
  echo "⚠️ parallel_tool_calls not detected (tool_calls count=${cnt_parallel})."
  echo "Tools detected (if any):"
  tool_calls_names "$resp_parallel" | sed 's/^/  - /' || true
  echo "Full response:"
  echo "$resp_parallel" | jq .
fi