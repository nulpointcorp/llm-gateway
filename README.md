# nulpoint LLM-Gateway

**OpenAI-compatible LLM proxy with < 2 ms overhead, automatic failover, and response caching.**

Point any OpenAI SDK at `http://localhost:8080` and gain multi-provider routing, a circuit breaker, exact-match caching, and Prometheus metrics — without touching your application code.

```
Your App  →  nulpoint Gateway  →  OpenAI / Anthropic / Gemini / Mistral
              (< 2 ms overhead)      automatic failover, cache, rate limit
```

---

## Features

| Feature | Details                                                      |
|---|--------------------------------------------------------------|
| **OpenAI-compatible API** | Drop-in replacement — change `base_url`, zero code changes   |
| **Multi-provider routing** | Route by model name; add providers without restarting        |
| **Automatic failover** | Circuit breaker per provider; configurable retries           |
| **Response cache** | Exact-match SHA-256 cache; in-memory or Redis                |
| **Streaming (SSE)** | Full pass-through for streaming responses                    |
| **Embeddings** | `/v1/embeddings` — OpenAI, Mistral, Gemini                   |
| **Prometheus metrics** | `/metrics` endpoint; requests, latency, cache, circuit state |
| **Bring-your-own keys** | Optional client `Authorization` passthrough with fallback |
| **Zero required deps** | Runs with `CACHE_MODE=memory` — no Redis, no DB              |
| **Structured logs** | Full analysis of your requests                               |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/nulpoint/gateway.git
cd gateway

# 2. Set at least one provider key
export OPENAI_API_KEY=sk-...

# 3. Run (uses in-process cache — no Redis needed)
make run
```

The gateway is now listening on `http://localhost:8080`.

**Use it with any OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(
    api_key="any-string",         # optional when the gateway holds the upstream keys
    base_url="http://localhost:8080/v1",
)

# Chat completion
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Embeddings
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox jumps over the lazy dog.",
)
print(f"dims: {len(response.data[0].embedding)}")
```

More examples: [`examples/`](examples/)

---

## Docker

```bash
cp .env.example .env
# fill OPENAI_API_KEY (or any other provider key)

# In-memory cache — no external deps
docker compose up

# Redis-backed cache (for multi-replica setups)
docker compose --profile redis up
```

---

## Configuration

All configuration is via environment variables (or `config.yaml` in the working directory).

### Provider Keys

At least one key is required. The gateway enables only the providers with non-empty keys.

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `MISTRAL_API_KEY` | Mistral AI API key |

### Server

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8080` | HTTP listen port |
| `LOG_LEVEL` | `info` | Log level: `debug` / `info` / `warn` / `error` |
| `ALLOW_CLIENT_API_KEYS` | `false` | Forward `Authorization` headers from clients; fall back to config values when missing |

> **Client-supplied tokens:** With `ALLOW_CLIENT_API_KEYS=true` the gateway uses the caller's
> `Authorization: Bearer …` header (when present) and falls back to the configured key only if the
> header is missing. Cache entries are automatically namespaced per client key.

### Cache

| Variable | Default | Description |
|---|---|---|
| `CACHE_MODE` | `memory` | `memory` · `redis` · `none` |
| `CACHE_TTL` | `1h` | Default TTL for cached responses |
| `REDIS_URL` | — | Required when `CACHE_MODE=redis`. e.g. `redis://localhost:6379` |
| `CACHE_EXCLUDE_EXACT` | — | Comma-separated model names to never cache |
| `CACHE_EXCLUDE_PATTERNS` | — | Comma-separated Go regexes matched against model names |

> **In-memory vs Redis:** Use `memory` for single-instance deployments and local dev.
> Use `redis` when running multiple gateway replicas so they share a cache.

### Circuit Breaker

| Variable | Default | Description |
|---|---|---|
| `CB_ERROR_THRESHOLD` | `5` | Failures within the window that trip the breaker |
| `CB_TIME_WINDOW` | `60s` | Rolling window for counting failures |
| `CB_HALF_OPEN_TIMEOUT` | `30s` | How long the breaker stays open before a probe |

### Failover

| Variable | Default | Description |
|---|---|---|
| `MAX_RETRIES` | `3` | Max provider attempts per request (including first) |
| `PROVIDER_TIMEOUT` | `30s` | Per-provider HTTP timeout |

### Rate Limiting

| Variable | Default | Description |
|---|---|---|
| `RPM_LIMIT` | `0` (off) | Global requests-per-minute. Requires `CACHE_MODE=redis` |

### CORS / Other

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `APP_BASE_URL` | — | Base URL for absolute links in callbacks |

---

## API Reference

The gateway implements a subset of the OpenAI API.

### Proxy Endpoints

```
POST /v1/chat/completions    Main chat endpoint (streaming supported)
POST /v1/completions         Legacy completions (aliases chat/completions)
POST /v1/embeddings          Embeddings (OpenAI, Mistral, Gemini)
```

### Health & Metrics

```
GET /health      Full health snapshot (providers, cache, uptime)
GET /readiness   Liveness probe for Kubernetes (200 OK or 503)
GET /metrics     Prometheus metrics
```

### Model → Provider Routing

The gateway resolves the provider from the `model` field:

**Chat / completions:**

| Models | Provider |
|---|---|
| `gpt-4`, `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo` | OpenAI |
| `claude-3-5-sonnet`, `claude-3-opus`, `claude-3-haiku` | Anthropic |
| `gemini-pro`, `gemini-1.5-pro`, `gemini-1.5-flash` | Google Gemini |
| `mistral-large`, `mistral-medium`, `mixtral-8x7b` | Mistral |
| *(anything else)* | Falls back to OpenAI |

**Embeddings (`POST /v1/embeddings`):**

| Models | Provider |
|---|---|
| `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` | OpenAI |
| `mistral-embed` | Mistral |
| `text-embedding-004`, `embedding-001` | Google Gemini |

### Embeddings

`POST /v1/embeddings` accepts a single string or an array of strings and returns
an OpenAI-compatible response. Anthropic does not support embeddings; requests
targeting a Claude model return `400`.

**Request:**

```json
{
  "model": "text-embedding-3-small",
  "input": ["Hello world", "How are you?"]
}
```

`input` may also be a plain string for single-text requests.

**Response:**

```json
{
  "object": "list",
  "model": "text-embedding-3-small",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.0023, -0.0094, ...] },
    { "object": "embedding", "index": 1, "embedding": [-0.0412,  0.0071, ...] }
  ],
  "usage": { "prompt_tokens": 8, "total_tokens": 8 }
}
```

### Error Format

Errors use the OpenAI error envelope so existing SDK error handling works:

```json
{
  "error": {
    "message": "Rate limit exceeded",
    "type": "rate_limit_error",
    "code": "rate_limit_exceeded"
  }
}
```

HTTP status mapping:

| Situation | Status |
|---|---|
| Provider 429 | `429` + `Retry-After: 60` |
| Provider 5xx | `502 Bad Gateway` |
| Timeout | `504 Gateway Timeout` |
| Auth failed | `401 Unauthorized` |
| Bad request | `400 Bad Request` |

---

## Architecture

### Key Design Decisions

- **fasthttp** over `net/http` for minimal per-request overhead (~5× faster at high concurrency).
- **No global state** — all dependencies injected via struct fields; trivially testable.
- **Async logger** — request logging is channel-buffered and never blocks the hot path.
- **Graceful cache degradation** — if Redis is down, cache ops are no-ops; the proxy continues.
- **Circuit breaker per provider** — a flaky OpenAI doesn't block Anthropic fallback.

---

## Development

```bash
# Run tests
make mock

# Run with race detector
make mock-race

# Run only fast tests (skip latency SLA benchmark)
make mock-short

# Benchmark proxy overhead (target: P50 < 2 ms)
make bench

# Generate coverage report
make mock-cover

# Lint
make lint
```


---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/my-feature`.
3. Write tests for new behaviour — `make test-race` must pass.
4. Open a pull request with a clear description of the change.

### Adding a Provider

1. Create `internal/providers/<name>/` implementing `providers.Provider`.
2. Add model aliases to `providers.ModelAliases` in `internal/providers/provider.go`.
3. Wire the new provider in `internal/app/app.go` (`buildProviders`).
4. Add the provider to `providers.DefaultFallbackOrder`.

To support embeddings, additionally implement the `providers.EmbeddingProvider` interface
and add embedding model aliases to `providers.EmbeddingModelAliases`.

---

## License

MIT — see [LICENSE](LICENSE).
