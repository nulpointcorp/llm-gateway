#!/usr/bin/env bash
# nulpoint Gateway — curl examples
#
# Prerequisites:
#   1. Gateway running locally:  make run
#   2. At least one provider key set in environment (e.g. OPENAI_API_KEY)
#
# Usage:
#   chmod +x quickstart.sh && ./quickstart.sh

GATEWAY="http://localhost:8080"

echo "=== 1. Health check ==="
curl -s "$GATEWAY/health" | jq .

echo ""
echo "=== 2. Basic chat completion (GPT-4o) ==="
curl -s "$GATEWAY/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ]
  }' | jq .choices[0].message.content

echo ""
echo "=== 3. Chat with Anthropic Claude ==="
curl -s "$GATEWAY/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user",   "content": "What is 2 + 2?"}
    ],
    "max_tokens": 64
  }' | jq .choices[0].message.content

echo ""
echo "=== 4. Streaming response (SSE) ==="
curl -s -N "$GATEWAY/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "stream": true
  }'

echo ""
echo "=== 5. Cache headers — run twice to see HIT ==="
for i in 1 2; do
  echo "  Request $i:"
  curl -si "$GATEWAY/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "gpt-4o",
      "messages": [{"role": "user", "content": "What colour is the sky?"}]
    }' | grep -E "(X-Cache|HTTP)"
done

echo ""
echo "=== 6. Embeddings — single input ==="
curl -s "$GATEWAY/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-3-small",
    "input": "The quick brown fox jumps over the lazy dog."
  }' | jq '{
    dims:   (.data[0].embedding | length),
    first4: .data[0].embedding[:4],
    tokens: .usage.total_tokens
  }'

echo ""
echo "=== 7. Embeddings — batch input ==="
curl -s "$GATEWAY/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-3-small",
    "input": ["Hello world", "How are you?", "Embeddings are useful."]
  }' | jq '{
    vectors: (.data | length),
    dims:    (.data[0].embedding | length),
    tokens:  .usage.total_tokens
  }'

echo ""
echo "=== 8. Prometheus metrics ==="
curl -s "$GATEWAY/metrics" | grep gateway_requests_total | head -5
