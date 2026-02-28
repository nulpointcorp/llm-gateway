package proxy

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/nulpointcorp/llm-gateway/internal/cache"
	"github.com/nulpointcorp/llm-gateway/internal/providers"
	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttputil"
)

// --- helpers ----------------------------------------------------------------

// stubCache is a simple in-memory cache for tests.
type stubCache struct {
	store map[string][]byte
}

func newStubCache() *stubCache {
	return &stubCache{store: make(map[string][]byte)}
}

func (c *stubCache) Get(_ context.Context, key string) ([]byte, bool) {
	v, ok := c.store[key]
	return v, ok
}

func (c *stubCache) Set(_ context.Context, key string, value []byte, _ time.Duration) error {
	c.store[key] = value
	return nil
}

func (c *stubCache) Delete(_ context.Context, key string) error {
	delete(c.store, key)
	return nil
}

// okProvider always returns a successful response.
func okProvider(name string) *funcProvider {
	return &funcProvider{
		name: name,
		requestFn: func(_ context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			return &providers.ProxyResponse{
				ID:      "resp-" + req.RequestID,
				Model:   req.Model,
				Content: "hello from " + name,
				Usage:   providers.Usage{InputTokens: 10, OutputTokens: 5},
			}, nil
		},
	}
}

// serveGateway starts a fasthttp server on an in-memory listener with the
// gateway's full middleware pipeline. Returns an HTTP client that routes to it,
// and a cleanup function.
func serveGateway(t *testing.T, gw *Gateway) (*http.Client, func()) {
	t.Helper()
	ln := fasthttputil.NewInmemoryListener()

	handler := applyMiddleware(
		func(ctx *fasthttp.RequestCtx) {
			switch string(ctx.Path()) {
			case "/v1/chat/completions", "/v1/completions":
				gw.dispatchChat(ctx)
			default:
				ctx.SetStatusCode(404)
			}
		},
		recovery,
		requestID,
		timing,
	)

	go func() {
		_ = fasthttp.Serve(ln, handler)
	}()

	client := &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				return ln.Dial()
			},
		},
	}

	return client, func() { ln.Close() }
}

// doPost sends a POST request via the in-memory listener client.
func doPost(t *testing.T, client *http.Client, path string, body []byte) *http.Response {
	t.Helper()
	req, err := http.NewRequest("POST", "http://test"+path, readerFromBytes(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

// readBody reads and returns the full response body.
func readBody(t *testing.T, resp *http.Response) []byte {
	t.Helper()
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// --- NewGateway tests -------------------------------------------------------

func TestNewGateway_PanicsOnNilContext(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for nil context")
		}
	}()
	NewGateway(nil, nil, nil)
}

func TestNewGateway_NilProvidersAndCache(t *testing.T) {
	gw := NewGateway(context.Background(), nil, nil)
	if gw == nil {
		t.Fatal("expected non-nil gateway")
	}
	if gw.health != nil {
		t.Error("health checker should be nil when no providers")
	}
}

func TestNewGateway_WithProviders(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai": okProvider("openai"),
	}
	gw := NewGateway(context.Background(), provs, nil)
	if gw.health == nil {
		t.Error("health checker should be created when providers exist")
	}
	gw.health.Close()
}

func TestNewGatewayWithProbes_CacheReadyProbe(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai": okProvider("openai"),
	}
	gw := NewGatewayWithProbes(context.Background(), provs, nil, func() bool { return true })
	if gw == nil {
		t.Fatal("expected non-nil gateway")
	}
	gw.health.Close()
}

// --- SetRateLimiters / SetLogger / SetCacheExclusions -----------------------

func TestGateway_Setters(t *testing.T) {
	gw := NewGateway(context.Background(), nil, nil)

	gw.SetRateLimiters(nil)
	if gw.rpmLimiter != nil {
		t.Error("expected nil rpm limiter")
	}

	gw.SetLogger(nil)
	if gw.reqLogger != nil {
		t.Error("expected nil logger")
	}

	gw.SetCacheExclusions(nil)
	if gw.cacheExclusions != nil {
		t.Error("expected nil exclusions")
	}

	gw.SetCORSOrigins([]string{"https://example.com"})
	if len(gw.corsOrigins) != 1 || gw.corsOrigins[0] != "https://example.com" {
		t.Error("CORS origins not set correctly")
	}
}

// --- dispatchChat tests (via in-memory HTTP server) -------------------------

// Tests that return early before context.WithTimeout can use bare RequestCtx.

func TestDispatchChat_InvalidJSON(t *testing.T) {
	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": okProvider("openai"),
	}, nil)

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.SetBody([]byte(`{invalid`))
	ctx.SetUserValue("request_id", "mock-1")

	gw.dispatchChat(ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusBadRequest {
		t.Errorf("expected 400, got %d", ctx.Response.StatusCode())
	}

	var errResp struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(ctx.Response.Body(), &errResp); err != nil {
		t.Fatalf("failed to parse error response: %v", err)
	}
	if errResp.Error.Code != "invalid_request" {
		t.Errorf("expected code=invalid_request, got %s", errResp.Error.Code)
	}
}

func TestDispatchChat_MissingModel(t *testing.T) {
	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": okProvider("openai"),
	}, nil)

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.SetBody([]byte(`{"messages":[{"role":"user","content":"hi"}]}`))
	ctx.SetUserValue("request_id", "mock-2")

	gw.dispatchChat(ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusBadRequest {
		t.Errorf("expected 400, got %d", ctx.Response.StatusCode())
	}
	body := string(ctx.Response.Body())
	if !contains(body, "model") {
		t.Errorf("error should mention 'model', got: %s", body)
	}
}

func TestDispatchChat_NoProviders(t *testing.T) {
	gw := NewGateway(context.Background(), map[string]providers.Provider{}, nil)

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.SetBody([]byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
	ctx.SetUserValue("request_id", "mock-3")

	gw.dispatchChat(ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusBadGateway {
		t.Errorf("expected 502, got %d", ctx.Response.StatusCode())
	}
}

func TestDispatchChat_ClientAPIKeyForwarding(t *testing.T) {
	var capturedKey, capturedID string
	prov := &funcProvider{
		name: "openai",
		requestFn: func(_ context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			capturedKey = req.APIKey
			capturedID = req.APIKeyID
			return &providers.ProxyResponse{
				ID:      "resp-" + req.RequestID,
				Model:   req.Model,
				Content: "ok",
			}, nil
		},
	}
	gw := NewGatewayWithOptions(context.Background(), map[string]providers.Provider{
		"openai": prov,
	}, nil, nil, GatewayOptions{AllowClientAPIKeys: true})
	client, closeFn := serveGateway(t, gw)
	defer closeFn()

	req, err := http.NewRequest("POST", "http://test/v1/chat/completions",
		readerFromBytes([]byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`)))
	if err != nil {
		t.Fatalf("failed to build request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-forward-me")

	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, body)
	}
	if capturedKey != "sk-forward-me" {
		t.Fatalf("expected API key to be forwarded, got %q", capturedKey)
	}
	sum := sha256.Sum256([]byte("sk-forward-me"))
	if capturedID != hex.EncodeToString(sum[:]) {
		t.Fatalf("expected APIKeyID hash, got %q", capturedID)
	}
}

func TestDispatchChat_ClientAPIKeyIgnoredWhenDisabled(t *testing.T) {
	var capturedKey, capturedID string
	prov := &funcProvider{
		name: "openai",
		requestFn: func(_ context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			capturedKey = req.APIKey
			capturedID = req.APIKeyID
			return &providers.ProxyResponse{
				ID:      "resp-" + req.RequestID,
				Model:   req.Model,
				Content: "ok",
			}, nil
		},
	}
	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": prov,
	}, nil)

	client, closeFn := serveGateway(t, gw)
	defer closeFn()

	req, err := http.NewRequest("POST", "http://test/v1/chat/completions",
		readerFromBytes([]byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`)))
	if err != nil {
		t.Fatalf("failed to build request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-ignored")

	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if capturedKey != "" || capturedID != "" {
		t.Fatalf("expected API key to be ignored when disabled, got key=%q id=%q", capturedKey, capturedID)
	}
}

// Tests that reach provider calls need a real fasthttp server context.

func TestDispatchChat_Success(t *testing.T) {
	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": okProvider("openai"),
	}, nil)

	client, cleanup := serveGateway(t, gw)
	defer cleanup()

	resp := doPost(t, client, "/v1/chat/completions",
		[]byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`))
	body := readBody(t, resp)

	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", resp.StatusCode, body)
	}

	var out outboundResponse
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if out.Object != "chat.completion" {
		t.Errorf("expected object=chat.completion, got %s", out.Object)
	}
	if len(out.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(out.Choices))
	}
	if out.Choices[0].FinishReason != "stop" {
		t.Errorf("expected finish_reason=stop, got %s", out.Choices[0].FinishReason)
	}
	if out.Usage.TotalTokens != 15 {
		t.Errorf("expected total_tokens=15, got %d", out.Usage.TotalTokens)
	}
	if resp.Header.Get("X-Cache") != xCacheMISS {
		t.Errorf("expected X-Cache=MISS on first request")
	}
}

func TestDispatchChat_CacheHit(t *testing.T) {
	sc := newStubCache()
	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": okProvider("openai"),
	}, sc)

	client, cleanup := serveGateway(t, gw)
	defer cleanup()

	reqBody := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"cached"}]}`)

	// First request — cache miss.
	resp1 := doPost(t, client, "/v1/chat/completions", reqBody)
	readBody(t, resp1)

	if resp1.Header.Get("X-Cache") != xCacheMISS {
		t.Error("first request should be a cache MISS")
	}

	// Second request — cache hit.
	resp2 := doPost(t, client, "/v1/chat/completions", reqBody)
	readBody(t, resp2)

	if resp2.Header.Get("X-Cache") != xCacheHIT {
		t.Error("second request should be a cache HIT")
	}
	if resp2.StatusCode != http.StatusOK {
		t.Errorf("expected 200 on cache hit, got %d", resp2.StatusCode)
	}
}

func TestDispatchChat_CacheExcludedModel(t *testing.T) {
	sc := newStubCache()
	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": okProvider("openai"),
	}, sc)

	el, err := cache.NewExclusionList([]string{"gpt-4o"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	gw.SetCacheExclusions(el)

	client, cleanup := serveGateway(t, gw)
	defer cleanup()

	reqBody := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"no-cache"}]}`)

	// First request.
	resp1 := doPost(t, client, "/v1/chat/completions", reqBody)
	readBody(t, resp1)
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp1.StatusCode)
	}

	// Second request — should NOT be a cache hit because model is excluded.
	resp2 := doPost(t, client, "/v1/chat/completions", reqBody)
	readBody(t, resp2)

	xCache := resp2.Header.Get("X-Cache")
	if xCache == xCacheHIT {
		t.Error("excluded model should never produce a cache HIT")
	}
}

func TestDispatchChat_ProviderError(t *testing.T) {
	failing := &funcProvider{
		name: "openai",
		requestFn: func(_ context.Context, _ *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			return nil, &providerError{status: 503, msg: "service unavailable"}
		},
	}
	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": failing,
	}, nil)

	client, cleanup := serveGateway(t, gw)
	defer cleanup()

	resp := doPost(t, client, "/v1/chat/completions",
		[]byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"fail"}]}`))
	readBody(t, resp)

	if resp.StatusCode == http.StatusOK {
		t.Error("expected non-200 status when provider fails")
	}
}

func TestDispatchChat_StreamingResponse(t *testing.T) {
	streamProv := &funcProvider{
		name: "openai",
		requestFn: func(_ context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			ch := make(chan providers.StreamChunk, 3)
			ch <- providers.StreamChunk{Content: "hello "}
			ch <- providers.StreamChunk{Content: "world"}
			ch <- providers.StreamChunk{Content: "", FinishReason: "stop"}
			close(ch)
			return &providers.ProxyResponse{
				ID:     "stream-resp",
				Model:  req.Model,
				Stream: ch,
			}, nil
		},
	}
	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": streamProv,
	}, nil)

	client, cleanup := serveGateway(t, gw)
	defer cleanup()

	resp := doPost(t, client, "/v1/chat/completions",
		[]byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"stream"}],"stream":true}`))
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, body)
	}

	ct := resp.Header.Get("Content-Type")
	if !contains(ct, "text/event-stream") {
		t.Errorf("expected text/event-stream content type, got %s", ct)
	}

	// Read SSE lines.
	scanner := bufio.NewScanner(resp.Body)
	var dataLines []string
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) > 5 && line[:5] == "data:" {
			dataLines = append(dataLines, line[6:])
		}
	}

	if len(dataLines) == 0 {
		t.Fatal("expected at least one data line in SSE stream")
	}

	// Last data line should be [DONE].
	last := dataLines[len(dataLines)-1]
	if last != "[DONE]" {
		t.Errorf("expected last SSE line to be [DONE], got %q", last)
	}
}

// --- buildCacheKey tests ----------------------------------------------------

func TestBuildCacheKey_Deterministic(t *testing.T) {
	req := &providers.ProxyRequest{
		Model:       "gpt-4o",
		Messages:    []providers.Message{{Role: "user", Content: "hello"}},
		Temperature: 0.7,
		MaxTokens:   100,
		WorkspaceID: "ws-1",
	}

	key1 := buildCacheKey(req)
	key2 := buildCacheKey(req)

	if key1 != key2 {
		t.Errorf("cache key should be deterministic: %s != %s", key1, key2)
	}
	if !contains(key1, "cache:") {
		t.Errorf("cache key should have prefix 'cache:', got %s", key1)
	}
}

func TestBuildCacheKey_DifferentModels(t *testing.T) {
	req1 := &providers.ProxyRequest{
		Model:       "gpt-4o",
		Messages:    []providers.Message{{Role: "user", Content: "hi"}},
		Temperature: 0.5,
	}
	req2 := &providers.ProxyRequest{
		Model:       "claude-3-opus",
		Messages:    []providers.Message{{Role: "user", Content: "hi"}},
		Temperature: 0.5,
	}

	if buildCacheKey(req1) == buildCacheKey(req2) {
		t.Error("different models should produce different cache keys")
	}
}

func TestBuildCacheKey_DifferentMessages(t *testing.T) {
	req1 := &providers.ProxyRequest{
		Model:    "gpt-4o",
		Messages: []providers.Message{{Role: "user", Content: "hello"}},
	}
	req2 := &providers.ProxyRequest{
		Model:    "gpt-4o",
		Messages: []providers.Message{{Role: "user", Content: "world"}},
	}

	if buildCacheKey(req1) == buildCacheKey(req2) {
		t.Error("different messages should produce different cache keys")
	}
}

func TestBuildCacheKey_DifferentWorkspaces(t *testing.T) {
	req1 := &providers.ProxyRequest{
		Model:       "gpt-4o",
		Messages:    []providers.Message{{Role: "user", Content: "hi"}},
		WorkspaceID: "ws-1",
	}
	req2 := &providers.ProxyRequest{
		Model:       "gpt-4o",
		Messages:    []providers.Message{{Role: "user", Content: "hi"}},
		WorkspaceID: "ws-2",
	}

	if buildCacheKey(req1) == buildCacheKey(req2) {
		t.Error("different workspace IDs should produce different cache keys")
	}
}

func TestBuildCacheKey_DifferentTemperatures(t *testing.T) {
	req1 := &providers.ProxyRequest{
		Model:       "gpt-4o",
		Messages:    []providers.Message{{Role: "user", Content: "hi"}},
		Temperature: 0.0,
	}
	req2 := &providers.ProxyRequest{
		Model:       "gpt-4o",
		Messages:    []providers.Message{{Role: "user", Content: "hi"}},
		Temperature: 1.0,
	}

	if buildCacheKey(req1) == buildCacheKey(req2) {
		t.Error("different temperatures should produce different cache keys")
	}
}

func TestBuildCacheKey_DifferentAPIKeys(t *testing.T) {
	req1 := &providers.ProxyRequest{
		Model:    "gpt-4o",
		Messages: []providers.Message{{Role: "user", Content: "hi"}},
		APIKeyID: "hash-a",
	}
	req2 := &providers.ProxyRequest{
		Model:    "gpt-4o",
		Messages: []providers.Message{{Role: "user", Content: "hi"}},
		APIKeyID: "hash-b",
	}

	if buildCacheKey(req1) == buildCacheKey(req2) {
		t.Error("different API key hashes should produce different cache keys")
	}
}

func TestBuildCacheKey_DifferentMaxTokens(t *testing.T) {
	req1 := &providers.ProxyRequest{
		Model:     "gpt-4o",
		Messages:  []providers.Message{{Role: "user", Content: "hi"}},
		MaxTokens: 100,
	}
	req2 := &providers.ProxyRequest{
		Model:     "gpt-4o",
		Messages:  []providers.Message{{Role: "user", Content: "hi"}},
		MaxTokens: 200,
	}

	if buildCacheKey(req1) == buildCacheKey(req2) {
		t.Error("different max_tokens should produce different cache keys")
	}
}

// --- handleProviderError tests ----------------------------------------------

func TestHandleProviderError_StatusCoder(t *testing.T) {
	tests := []struct {
		name       string
		err        error
		wantStatus int
	}{
		{"429 rate limit", &providerError{status: 429, msg: "rate limited"}, 429},
		{"503 service unavailable", &providerError{status: 503, msg: "unavailable"}, 502},
		{"500 internal", &providerError{status: 500, msg: "internal"}, 502},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &fasthttp.RequestCtx{}
			handleProviderError(ctx, tt.err)
			if ctx.Response.StatusCode() != tt.wantStatus {
				t.Errorf("expected %d, got %d", tt.wantStatus, ctx.Response.StatusCode())
			}
		})
	}
}

func TestHandleProviderError_Timeout(t *testing.T) {
	ctx := &fasthttp.RequestCtx{}
	handleProviderError(ctx, context.DeadlineExceeded)
	if ctx.Response.StatusCode() != fasthttp.StatusGatewayTimeout {
		t.Errorf("expected 504, got %d", ctx.Response.StatusCode())
	}
}

func TestHandleProviderError_GenericError(t *testing.T) {
	ctx := &fasthttp.RequestCtx{}
	handleProviderError(ctx, context.Canceled)
	if ctx.Response.StatusCode() != fasthttp.StatusBadGateway {
		t.Errorf("expected 502, got %d", ctx.Response.StatusCode())
	}
}

// --- logRequest nil-safe mock -----------------------------------------------

func TestLogRequest_NilLogger(t *testing.T) {
	gw := NewGateway(context.Background(), nil, nil)
	// Should not panic when logger is nil.
	gw.logRequest("req-1", "openai", "gpt-4o", 10, 5, time.Millisecond, 200, false)
}

// --- helpers ----------------------------------------------------------------

func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// readerFromBytes wraps a byte slice in a reader for http.NewRequest.
func readerFromBytes(b []byte) io.Reader {
	return io.NopCloser(bReader(b))
}

type byteReader struct {
	data []byte
	pos  int
}

func bReader(b []byte) *byteReader { return &byteReader{data: b} }

func (r *byteReader) Read(p []byte) (n int, err error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	n = copy(p, r.data[r.pos:])
	r.pos += n
	return
}
