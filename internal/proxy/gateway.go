// Package proxy is the core LLM request dispatcher.
//
// The Gateway receives an incoming OpenAI-compatible request, resolves the
// target provider, checks the cache, applies rate limiting, and forwards the
// request to the selected provider — falling back to alternatives when the
// primary is unavailable.
//
// Key design constraints:
//   - Proxy overhead < 2 ms P50 (SLA). No blocking I/O on the hot path.
//   - Logger, cache, and rate limiter are optional and nil-safe.
//   - All I/O uses context.Context so timeouts propagate correctly.
//   - Streaming responses are pass-through (SSE); they are never cached.
package proxy

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/nulpointcorp/llm-gateway/internal/cache"
	"github.com/nulpointcorp/llm-gateway/internal/logger"
	"github.com/nulpointcorp/llm-gateway/internal/metrics"
	"github.com/nulpointcorp/llm-gateway/internal/providers"
	"github.com/nulpointcorp/llm-gateway/internal/ratelimit"
	"github.com/nulpointcorp/llm-gateway/pkg/apierr"
	"github.com/valyala/fasthttp"
)

const (
	xCacheHIT  = "HIT"
	xCacheMISS = "MISS"

	// defaultTPMLimit is a conservative fallback used when no per-workspace plan
	// information is available in the request context. Real limits are enforced
	// by the billing layer; this prevents runaway token consumption.
	defaultTPMLimit = 2_000_000
)

// GatewayOptions holds optional tuning parameters for a Gateway. All fields
// have sensible defaults and can be omitted.
type GatewayOptions struct {
	// Logger is the structured logger used for request events and failover
	// diagnostics. Defaults to a no-op logger when nil.
	Logger *slog.Logger

	// MaxRetries is the maximum number of provider attempts per request
	// (including the first). Must be ≥ 1. Default: providers.MaxRetries (3).
	MaxRetries int

	// ProviderTimeout is the per-provider HTTP request timeout.
	// Default: providers.ProviderTimeout (30s).
	ProviderTimeout time.Duration

	// CBConfig configures the per-provider circuit breaker thresholds.
	// Zero values use the package-level defaults.
	CBConfig CBConfig

	// AllowClientAPIKeys enables forwarding Authorization headers from clients
	// directly to upstream providers. When false, client headers are ignored and
	// only configured keys are used.
	AllowClientAPIKeys bool

	// Metrics enables Prometheus metrics collection. When nil, metrics are disabled.
	Metrics *metrics.Registry

	// CacheTTL controls the default TTL for cached responses.
	// Default: 1h.
	CacheTTL time.Duration
}

// Gateway is the main proxy — all dependencies are injected via the constructor
// so they can be replaced with mock doubles in unit tests.
type Gateway struct {
	providers map[string]providers.Provider
	cache     cache.Cache
	cb        *CircuitBreaker
	health    *HealthChecker
	baseCtx   context.Context
	log       *slog.Logger
	metrics   *metrics.Registry

	// Configurable failover parameters (set from GatewayOptions).
	maxRetries      int
	providerTimeout time.Duration
	cacheTTL        time.Duration

	// Optional dependencies — nil-safe when not configured.
	rpmLimiter      *ratelimit.RPMLimiter
	reqLogger       *logger.Logger
	cacheExclusions *cache.ExclusionList

	// CORS allowed origins. Empty slice means deny all; ["*"] means allow all.
	corsOrigins []string

	allowClientAPIKeys bool
}

// SetCORSOrigins configures the allowed CORS origins for the gateway.
func (g *Gateway) SetCORSOrigins(origins []string) {
	g.corsOrigins = origins
}

// NewGateway creates a Gateway with default settings.
func NewGateway(ctx context.Context, provs map[string]providers.Provider, c cache.Cache) *Gateway {
	return NewGatewayWithOptions(ctx, provs, c, nil, GatewayOptions{})
}

// NewGatewayWithProbes creates a Gateway with an explicit readiness probe for
// the cache backend (used by GET /readiness for Kubernetes liveness checks).
func NewGatewayWithProbes(
	baseCtx context.Context,
	provs map[string]providers.Provider,
	c cache.Cache,
	cacheReady func() bool,
) *Gateway {
	return NewGatewayWithOptions(baseCtx, provs, c, cacheReady, GatewayOptions{})
}

// NewGatewayWithOptions creates a fully configured Gateway. Use this when you
// need to customise the logger, circuit breaker thresholds, or failover limits.
func NewGatewayWithOptions(
	baseCtx context.Context,
	provs map[string]providers.Provider,
	c cache.Cache,
	cacheReady func() bool,
	opts GatewayOptions,
) *Gateway {
	if baseCtx == nil {
		panic("gateway: context must not be nil")
	}

	log := opts.Logger
	if log == nil {
		log = slog.Default()
	}

	maxRetries := opts.MaxRetries
	if maxRetries < 1 {
		maxRetries = providers.MaxRetries
	}

	providerTimeout := opts.ProviderTimeout
	if providerTimeout <= 0 {
		providerTimeout = providers.ProviderTimeout
	}

	cacheTTL := opts.CacheTTL
	if cacheTTL <= 0 {
		cacheTTL = time.Hour
	}

	gw := &Gateway{
		providers:          provs,
		cache:              c,
		cb:                 NewCircuitBreakerWithConfig(opts.CBConfig),
		baseCtx:            baseCtx,
		log:                log,
		maxRetries:         maxRetries,
		providerTimeout:    providerTimeout,
		cacheTTL:           cacheTTL,
		metrics:            opts.Metrics,
		allowClientAPIKeys: opts.AllowClientAPIKeys,
	}

	// Initialise circuit breaker gauges (closed) for known providers.
	if gw.metrics != nil && gw.cb != nil {
		for _, name := range providers.DefaultFallbackOrder {
			gw.metrics.SetCircuitBreaker(name, int64(gw.cb.State(name)))
		}
	}

	if len(provs) > 0 {
		gw.health = NewHealthChecker(baseCtx, provs, cacheReady, gw.metrics)
	}

	return gw
}

// SetRateLimiters injects the RPM rate limiter.
func (g *Gateway) SetRateLimiters(rpm *ratelimit.RPMLimiter) {
	g.rpmLimiter = rpm
}

// SetLogger injects the async request logger (e.g. for ClickHouse or stdout).
func (g *Gateway) SetLogger(l *logger.Logger) {
	g.reqLogger = l
}

// SetCacheExclusions injects the cache exclusion list.
// Requests whose model name matches any rule skip both cache GET and SET.
func (g *Gateway) SetCacheExclusions(el *cache.ExclusionList) {
	g.cacheExclusions = el
}

// ── Internal request / response types ─────────────────────────────────────────

type (
	// inboundEmbeddingRequest mirrors the OpenAI POST /v1/embeddings body.
	// The "input" field accepts a string or array of strings; we normalise
	// to []string via a custom unmarshal in parseEmbeddingInput.
	inboundEmbeddingRequest struct {
		Model          string          `json:"model"`
		Input          json.RawMessage `json:"input"`
		EncodingFormat string          `json:"encoding_format"`
	}

	outboundEmbeddingData struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	}

	outboundEmbeddingUsage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	}

	outboundEmbeddingResponse struct {
		Object string                  `json:"object"`
		Data   []outboundEmbeddingData `json:"data"`
		Model  string                  `json:"model"`
		Usage  outboundEmbeddingUsage  `json:"usage"`
	}
)

// parseEmbeddingInput converts the raw JSON "input" field into []string.
// The OpenAI API accepts either a bare string or an array of strings.
func parseEmbeddingInput(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("'input' is required")
	}
	// Try array first.
	var arr []string
	if err := json.Unmarshal(raw, &arr); err == nil {
		if len(arr) == 0 {
			return nil, fmt.Errorf("'input' must not be empty")
		}
		return arr, nil
	}
	// Try bare string.
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if s == "" {
			return nil, fmt.Errorf("'input' must not be empty")
		}
		return []string{s}, nil
	}
	return nil, fmt.Errorf("'input' must be a string or array of strings")
}

// dispatchEmbeddings handles POST /v1/embeddings.
// It resolves the provider from the model name, delegates to the provider's
// Embed method, and returns an OpenAI-compatible response envelope.
func (g *Gateway) dispatchEmbeddings(ctx *fasthttp.RequestCtx) {
	start := time.Now()
	route := "embeddings"
	reqBytes := len(ctx.PostBody())
	servedProvider := "unknown"
	cacheLabel := "bypass"
	inputTokens, outputTokens := 0, 0
	cached := false
	respBytes := -1

	if g.metrics != nil {
		g.metrics.IncInFlight()
	}
	defer func() {
		if g.metrics == nil {
			return
		}
		g.metrics.DecInFlight()
		status := ctx.Response.StatusCode()
		dur := time.Since(start)
		if respBytes < 0 {
			respBytes = len(ctx.Response.Body())
		}
		g.metrics.ObserveHTTP(route, status, dur, reqBytes, respBytes)
		g.metrics.RecordRequest(servedProvider, status, dur.Milliseconds())
		g.metrics.ObserveGatewayRequest(servedProvider, route, cacheLabel, dur)
		g.metrics.AddTokens(servedProvider, route, inputTokens, outputTokens, cached)
	}()

	reqID, _ := ctx.UserValue("request_id").(string)
	clientKey, clientKeyID := g.extractClientAPIKey(ctx)

	// 1. Parse request.
	var req inboundEmbeddingRequest
	if err := json.Unmarshal(ctx.PostBody(), &req); err != nil {
		apierr.Write(ctx, fasthttp.StatusBadRequest,
			fmt.Sprintf("invalid JSON: %s", err.Error()),
			apierr.TypeInvalidRequest, apierr.CodeInvalidRequest)
		return
	}

	if req.Model == "" {
		apierr.Write(ctx, fasthttp.StatusBadRequest,
			"field 'model' is required",
			apierr.TypeInvalidRequest, apierr.CodeInvalidRequest)
		return
	}

	inputs, err := parseEmbeddingInput(req.Input)
	if err != nil {
		apierr.Write(ctx, fasthttp.StatusBadRequest,
			err.Error(), apierr.TypeInvalidRequest, apierr.CodeInvalidRequest)
		return
	}

	// 2. Resolve provider.
	providerName := resolveEmbeddingProvider(req.Model)
	servedProvider = providerName

	g.log.InfoContext(ctx, "embedding_request",
		slog.String("request_id", reqID),
		slog.String("model", req.Model),
		slog.String("provider", providerName),
		slog.Int("inputs", len(inputs)),
	)

	if len(g.providers) == 0 {
		apierr.Write(ctx, fasthttp.StatusBadGateway,
			"no providers configured",
			apierr.TypeProviderError, apierr.CodeProviderError)
		return
	}

	// 3. Find a provider that implements EmbeddingProvider.
	prov, ok := g.providers[providerName]
	if !ok {
		// Try the first available provider.
		for _, p := range g.providers {
			prov = p
			break
		}
	}
	if prov != nil {
		servedProvider = prov.Name()
	}

	embedder, ok := prov.(providers.EmbeddingProvider)
	if !ok {
		apierr.Write(ctx, fasthttp.StatusBadRequest,
			fmt.Sprintf("provider %q does not support embeddings", prov.Name()),
			apierr.TypeInvalidRequest, apierr.CodeInvalidRequest)
		return
	}

	// 4. Call the provider.
	provCtx, cancel := context.WithTimeout(ctx, g.providerTimeout)
	defer cancel()

	embReq := &providers.EmbeddingRequest{
		Input:     inputs,
		Model:     req.Model,
		RequestID: reqID,
		APIKey:    clientKey,
		APIKeyID:  clientKeyID,
	}

	upStart := time.Now()
	embResp, err := embedder.Embed(provCtx, embReq)
	upDur := time.Since(upStart)
	if err != nil {
		if g.metrics != nil {
			reason := classifyError(err)
			g.metrics.ObserveUpstreamAttempt(servedProvider, route, reason, upDur)
			g.metrics.RecordError(servedProvider, reason)
		}
		g.log.ErrorContext(ctx, "embedding_error",
			slog.String("request_id", reqID),
			slog.String("provider", providerName),
			slog.String("error", err.Error()),
			slog.Duration("elapsed", time.Since(start)),
		)
		handleProviderError(ctx, err)
		return
	}
	if g.metrics != nil {
		g.metrics.ObserveUpstreamAttempt(servedProvider, route, "success", upDur)
	}

	// 5. Build OpenAI-compatible response.
	outData := make([]outboundEmbeddingData, len(embResp.Data))
	for i, d := range embResp.Data {
		outData[i] = outboundEmbeddingData{
			Object:    "embedding",
			Index:     d.Index,
			Embedding: d.Embedding,
		}
	}

	out := outboundEmbeddingResponse{
		Object: "list",
		Data:   outData,
		Model:  embResp.Model,
		Usage: outboundEmbeddingUsage{
			PromptTokens: embResp.Usage.InputTokens,
			TotalTokens:  embResp.Usage.InputTokens,
		},
	}
	inputTokens = embResp.Usage.InputTokens

	body, err := json.Marshal(out)
	if err != nil {
		apierr.Write(ctx, fasthttp.StatusInternalServerError,
			"failed to serialize response", apierr.TypeServerError, apierr.CodeInternalError)
		return
	}

	g.log.DebugContext(ctx, "embedding_ok",
		slog.String("request_id", reqID),
		slog.String("provider", prov.Name()),
		slog.String("model", embResp.Model),
		slog.Int("vectors", len(embResp.Data)),
		slog.Int("input_tokens", embResp.Usage.InputTokens),
		slog.Duration("elapsed", time.Since(start)),
	)

	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentType("application/json")
	ctx.SetBody(body)
	respBytes = len(body)
}

// extractClientAPIKey returns the Authorization bearer token (if allowed and present)
// and a deterministic SHA-256 hash suitable for cache partitioning.
func (g *Gateway) extractClientAPIKey(ctx *fasthttp.RequestCtx) (token string, tokenID string) {
	if !g.allowClientAPIKeys {
		return "", ""
	}
	raw := strings.TrimSpace(string(ctx.Request.Header.Peek("Authorization")))
	if raw == "" {
		return "", ""
	}
	token = parseBearerToken(raw)
	if token == "" {
		return "", ""
	}
	sum := sha256.Sum256([]byte(token))
	return token, hex.EncodeToString(sum[:])
}

func parseBearerToken(header string) string {
	if header == "" {
		return ""
	}
	parts := strings.SplitN(header, " ", 2)
	if len(parts) != 2 {
		return ""
	}
	if !strings.EqualFold(parts[0], "Bearer") {
		return ""
	}
	token := strings.TrimSpace(parts[1])
	if token == "" {
		return ""
	}
	return token
}

type (
	inboundMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	inboundRequest struct {
		Model       string           `json:"model"`
		Messages    []inboundMessage `json:"messages"`
		Stream      bool             `json:"stream"`
		Temperature float64          `json:"temperature"`
		MaxTokens   int              `json:"max_tokens"`
	}

	outboundUsage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	}

	outboundMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	outboundChoice struct {
		Index        int             `json:"index"`
		Message      outboundMessage `json:"message"`
		FinishReason string          `json:"finish_reason"`
	}

	outboundResponse struct {
		ID      string           `json:"id"`
		Object  string           `json:"object"`
		Created int64            `json:"created"`
		Model   string           `json:"model"`
		Choices []outboundChoice `json:"choices"`
		Usage   outboundUsage    `json:"usage"`
	}
)

// dispatchChat is the core handler for /v1/chat/completions and /v1/completions.
func (g *Gateway) dispatchChat(ctx *fasthttp.RequestCtx) {
	start := time.Now()
	path := string(ctx.Path())
	route := "chat_completions"
	if path == "/v1/completions" {
		route = "completions"
	}
	reqBytes := len(ctx.PostBody())
	servedProvider := "unknown"
	cacheLabel := "bypass" // hit|miss|bypass
	inputTokens, outputTokens := 0, 0
	cached := false
	streaming := false
	respBytes := -1

	if g.metrics != nil {
		g.metrics.IncInFlight()
	}
	defer func() {
		if g.metrics == nil {
			return
		}
		if streaming {
			return // finalised by the stream writer
		}
		g.metrics.DecInFlight()
		status := ctx.Response.StatusCode()
		dur := time.Since(start)
		if respBytes < 0 {
			respBytes = len(ctx.Response.Body())
		}
		g.metrics.ObserveHTTP(route, status, dur, reqBytes, respBytes)
		g.metrics.RecordRequest(servedProvider, status, dur.Milliseconds())
		g.metrics.ObserveGatewayRequest(servedProvider, route, cacheLabel, dur)
		g.metrics.AddTokens(servedProvider, route, inputTokens, outputTokens, cached)
	}()

	reqID, _ := ctx.UserValue("request_id").(string)
	clientKey, clientKeyID := g.extractClientAPIKey(ctx)

	// 1. Parse request body.
	var req inboundRequest
	if err := json.Unmarshal(ctx.PostBody(), &req); err != nil {
		apierr.Write(ctx, fasthttp.StatusBadRequest,
			fmt.Sprintf("invalid JSON: %s", err.Error()),
			apierr.TypeInvalidRequest, apierr.CodeInvalidRequest)
		return
	}

	if req.Model == "" {
		apierr.Write(ctx, fasthttp.StatusBadRequest,
			"field 'model' is required",
			apierr.TypeInvalidRequest, apierr.CodeInvalidRequest)
		return
	}

	// 2. Route to provider based on model name.
	providerName := resolveProvider(req.Model)
	servedProvider = providerName

	g.log.InfoContext(ctx, "request",
		slog.String("request_id", reqID),
		slog.String("model", req.Model),
		slog.String("provider", providerName),
		slog.Bool("stream", req.Stream),
	)

	if len(g.providers) == 0 {
		apierr.Write(ctx, fasthttp.StatusBadGateway,
			"no providers configured",
			apierr.TypeProviderError, apierr.CodeProviderError)
		return
	}

	// 3. Rate limit check (RPM).
	if g.rpmLimiter != nil {
		allowed, err := g.rpmLimiter.Allow(ctx)
		if err == nil && !allowed {
			if g.metrics != nil {
				g.metrics.RecordRateLimit("blocked")
			}
			g.log.WarnContext(ctx, "rate_limit_exceeded",
				slog.String("request_id", reqID),
				slog.String("provider", providerName),
			)
			apierr.WriteRateLimit(ctx)
			return
		}
		if g.metrics != nil {
			if err != nil {
				g.metrics.RecordRateLimit("error")
			} else {
				g.metrics.RecordRateLimit("allowed")
			}
		}
	}

	// 4. Build the normalized ProxyRequest.
	msgs := make([]providers.Message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = providers.Message{Role: m.Role, Content: m.Content}
	}

	proxyReq := &providers.ProxyRequest{
		Model:       req.Model,
		Messages:    msgs,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		RequestID:   reqID,
		APIKey:      clientKey,
		APIKeyID:    clientKeyID,
	}

	// 5. Cache lookup — non-streaming only; skip excluded models.
	cacheEligible := !req.Stream && g.cache != nil && (g.cacheExclusions == nil || !g.cacheExclusions.Matches(req.Model))
	if g.metrics != nil && !cacheEligible {
		g.metrics.CacheGetBypass()
	}
	if cacheEligible {
		cacheKey := buildCacheKey(proxyReq)
		if cachedBody, ok := g.cache.Get(ctx, cacheKey); ok {
			cacheLabel = "hit"
			cached = true
			respBytes = len(cachedBody)
			if g.metrics != nil {
				g.metrics.CacheGetHit()
			}
			g.log.DebugContext(ctx, "cache_hit",
				slog.String("request_id", reqID),
				slog.String("model", req.Model),
			)
			ctx.Response.Header.Set("X-Cache", xCacheHIT)
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody(cachedBody)

			// Best-effort token extraction from cached payload.
			var cu struct {
				Model string `json:"model"`
				Usage struct {
					PromptTokens     int `json:"prompt_tokens"`
					CompletionTokens int `json:"completion_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(cachedBody, &cu); err == nil {
				inputTokens = cu.Usage.PromptTokens
				outputTokens = cu.Usage.CompletionTokens
			}

			g.logRequest(reqID, providerName, req.Model,
				inputTokens, outputTokens, time.Since(start), fasthttp.StatusOK, true)
			return
		}
		cacheLabel = "miss"
		if g.metrics != nil {
			g.metrics.CacheGetMiss()
		}
	}

	// 6. Call provider with automatic failover.
	provCtx, cancel := context.WithTimeout(ctx, g.providerTimeout)
	defer cancel()

	resp, usedProvider, err := g.requestWithFailover(provCtx, proxyReq, providerName, route)
	if err != nil {
		g.log.ErrorContext(ctx, "provider_error",
			slog.String("request_id", reqID),
			slog.String("primary_provider", providerName),
			slog.String("error", err.Error()),
			slog.Duration("elapsed", time.Since(start)),
		)
		handleProviderError(ctx, err)
		g.logRequest(reqID, providerName, req.Model,
			0, 0, time.Since(start), fasthttp.StatusBadGateway, false)
		return
	}
	servedProvider = usedProvider

	// 7a. Streaming — SSE pass-through. Responses are never cached for streams.
	if req.Stream && resp.Stream != nil {
		streaming = true
		capturedStart := start
		capturedReqBytes := reqBytes
		capturedRoute := route
		capturedProvider := usedProvider
		writeSSE(ctx, resp, func(outputTokens int) {
			g.logRequest(reqID, usedProvider, resp.Model,
				0, outputTokens, time.Since(capturedStart), fasthttp.StatusOK, false)
			if g.metrics != nil {
				// End-to-end duration is measured until stream drain.
				dur := time.Since(capturedStart)
				g.metrics.ObserveHTTP(capturedRoute, fasthttp.StatusOK, dur, capturedReqBytes, -1)
				g.metrics.RecordRequest(capturedProvider, fasthttp.StatusOK, dur.Milliseconds())
				g.metrics.ObserveGatewayRequest(capturedProvider, capturedRoute, "bypass", dur)
				g.metrics.AddTokens(capturedProvider, capturedRoute, 0, outputTokens, false)
				g.metrics.DecInFlight()
			}
		})
		return
	}

	// 7b. Non-streaming — build an OpenAI-compatible response envelope.
	out := outboundResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   resp.Model,
		Choices: []outboundChoice{
			{
				Index:        0,
				Message:      outboundMessage{Role: "assistant", Content: resp.Content},
				FinishReason: "stop",
			},
		},
		Usage: outboundUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}

	body, err := json.Marshal(out)
	if err != nil {
		apierr.Write(ctx, fasthttp.StatusInternalServerError,
			"failed to serialize response", apierr.TypeServerError, apierr.CodeInternalError)
		return
	}

	// 8. Populate cache for future identical requests.
	if cacheEligible {
		cacheKey := buildCacheKey(proxyReq)
		if err := g.cache.Set(ctx, cacheKey, body, g.cacheTTL); err != nil {
			if g.metrics != nil {
				g.metrics.CacheSetError()
			}
		} else {
			if g.metrics != nil {
				g.metrics.CacheSetOK()
			}
		}
	}

	// 9. Emit request log entry asynchronously.
	g.logRequest(reqID, usedProvider, resp.Model,
		resp.Usage.InputTokens, resp.Usage.OutputTokens,
		time.Since(start), fasthttp.StatusOK, false)
	inputTokens = resp.Usage.InputTokens
	outputTokens = resp.Usage.OutputTokens
	if cacheEligible {
		cacheLabel = "miss"
	} else {
		cacheLabel = "bypass"
	}

	g.log.DebugContext(ctx, "response_ok",
		slog.String("request_id", reqID),
		slog.String("used_provider", usedProvider),
		slog.String("model", resp.Model),
		slog.Int("input_tokens", resp.Usage.InputTokens),
		slog.Int("output_tokens", resp.Usage.OutputTokens),
		slog.Duration("elapsed", time.Since(start)),
	)

	ctx.Response.Header.Set("X-Cache", xCacheMISS)
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentType("application/json")
	ctx.SetBody(body)
	respBytes = len(body)
}

// logRequest enqueues a RequestLog entry to the async logger. Never blocks.
func (g *Gateway) logRequest(
	requestID, provider, model string,
	inputTokens, outputTokens int,
	latency time.Duration,
	status int,
	isCached bool,
) {
	if g.reqLogger == nil {
		return
	}

	reqUUID, _ := uuid.Parse(requestID)

	// Clamp to uint16 max so we don't overflow the field.
	latencyMs := uint16(latency.Milliseconds())
	if latency.Milliseconds() > 65535 {
		latencyMs = 65535
	}

	g.reqLogger.Log(logger.RequestLog{
		ID:           reqUUID,
		Provider:     provider,
		Model:        model,
		InputTokens:  uint32(inputTokens),
		OutputTokens: uint32(outputTokens),
		LatencyMs:    latencyMs,
		Status:       uint16(status),
		Cached:       isCached,
		CreatedAt:    time.Now(),
	})
}

// buildCacheKey returns a deterministic SHA-256 cache key for the request.
// The provider name is included to prevent cross-provider key collisions when
// two providers share a model name.
func buildCacheKey(req *providers.ProxyRequest) string {
	type msg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	msgs := make([]msg, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = msg{Role: m.Role, Content: m.Content}
	}
	data, _ := json.Marshal(struct {
		W    string `json:"w"`
		K    string `json:"k"`
		P    string `json:"p"`
		M    string `json:"m"`
		T    string `json:"t"`
		MT   int    `json:"mt"`
		Msgs []msg  `json:"msgs"`
	}{
		req.WorkspaceID,
		req.APIKeyID,
		resolveProvider(req.Model),
		req.Model,
		fmt.Sprintf("%.2f", req.Temperature),
		req.MaxTokens,
		msgs,
	})
	h := sha256.Sum256(data)
	return "cache:" + hex.EncodeToString(h[:])
}

// handleProviderError maps provider errors to the appropriate HTTP response.
//
//	statusCoder (providers that return HTTP codes) → passed through with remapping
//	context.DeadlineExceeded                       → 504 Gateway Timeout
//	all other errors                               → 502 Bad Gateway
func handleProviderError(ctx *fasthttp.RequestCtx, err error) {
	type statusCoder interface{ HTTPStatus() int }

	if sc, ok := err.(statusCoder); ok {
		apierr.WriteProviderError(ctx, sc.HTTPStatus(), err.Error())
		return
	}
	if errors.Is(err, context.DeadlineExceeded) {
		apierr.WriteTimeout(ctx)
		return
	}

	apierr.Write(ctx, fasthttp.StatusBadGateway,
		err.Error(), apierr.TypeProviderError, apierr.CodeProviderError)
}

// writeSSE streams response chunks from the provider as Server-Sent Events.
// onComplete is called once the stream drains with an estimated output token
// count (≈ chars/4), enabling async logging for streaming requests.
func writeSSE(ctx *fasthttp.RequestCtx, resp *providers.ProxyResponse, onComplete func(outputTokens int)) {
	ctx.SetContentType("text/event-stream")
	ctx.Response.Header.Set("Cache-Control", "no-cache")
	ctx.Response.Header.Set("Connection", "keep-alive")
	ctx.SetStatusCode(fasthttp.StatusOK)

	ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
		defer func() { recover() }() //nolint:errcheck // panic recovery in stream writer

		var sb strings.Builder
		for chunk := range resp.Stream {
			sb.WriteString(chunk.Content)

			delta := map[string]any{
				"id":      "chatcmpl-stream",
				"object":  "chat.completion.chunk",
				"created": time.Now().Unix(),
				"choices": []map[string]any{
					{
						"index": 0,
						"delta": map[string]string{"content": chunk.Content},
						"finish_reason": func() any {
							if chunk.FinishReason != "" {
								return chunk.FinishReason
							}
							return nil
						}(),
					},
				},
			}
			data, _ := json.Marshal(delta)
			fmt.Fprintf(w, "data: %s\n\n", data)
			w.Flush() //nolint:errcheck
		}

		fmt.Fprint(w, "data: [DONE]\n\n")
		w.Flush() //nolint:errcheck

		// Estimate output tokens: ~4 characters per token (GPT-style heuristic).
		estimated := sb.Len() / 4
		if estimated == 0 {
			estimated = 1
		}
		if onComplete != nil {
			onComplete(estimated)
		}
	})
}
