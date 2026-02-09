#!/usr/bin/env bash
set -euo pipefail

# Quick health check llm endpoint.

BASE_URL=${QWEN_BASE_URL:-"http://100.124.83.129:46400/v1"}
API_KEY=${QWEN_API_KEY:-"api_key"}
MODEL=${QWEN_MODEL:-"Qwen3-Coder-Next"}

payload=$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [
    {"role": "user", "content": "ping"}
  ],
  "max_tokens": 128
}
EOF
)

echo "POST ${BASE_URL}/chat/completions with model=${MODEL}"
resp=$(curl -sS --fail \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$payload" \
  "${BASE_URL}/chat/completions")

if command -v jq >/dev/null 2>&1; then
  echo "$resp" | jq -e '.choices[0].message.content' >/dev/null
  echo "Qwen3 response:" "$(echo "$resp" | jq -r '.choices[0].message.content')"
else
  echo "Raw response (jq not found):"
  echo "$resp"
fi
