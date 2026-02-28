package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"testing"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttputil"
)

// serveRouter starts the full router (with all routes) on an in-memory
// listener and returns an HTTP client + cleanup.
func serveRouter(t *testing.T, gw *Gateway) (*http.Client, func()) {
	t.Helper()
	ln := fasthttputil.NewInmemoryListener()

	handler := applyMiddleware(
		func(ctx *fasthttp.RequestCtx) {
			switch string(ctx.Path()) {
			case "/v1/chat/completions":
				gw.handleChatCompletions(ctx)
			case "/v1/completions":
				gw.handleCompletions(ctx)
			case "/v1/embeddings":
				gw.handleEmbeddings(ctx)
			case "/health":
				gw.handleHealth(ctx)
			case "/readiness":
				gw.handleReadiness(ctx)
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

// --- handleHealth -----------------------------------------------------------

func TestHandleHealth_NoHealthChecker(t *testing.T) {
	gw := NewGateway(context.Background(), nil, nil)

	ctx := &fasthttp.RequestCtx{}
	gw.handleHealth(ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200, got %d", ctx.Response.StatusCode())
	}

	var resp map[string]any
	if err := json.Unmarshal(ctx.Response.Body(), &resp); err != nil {
		t.Fatalf("failed to parse health response: %v", err)
	}
	if resp["status"] != "ok" {
		t.Errorf("expected status=ok, got %v", resp["status"])
	}
}

func TestHandleHealth_WithProviders(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai": okProvider("openai"),
	}
	gw := NewGateway(context.Background(), provs, nil)
	defer gw.health.Close()

	ctx := &fasthttp.RequestCtx{}
	gw.handleHealth(ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200, got %d", ctx.Response.StatusCode())
	}

	var snap HealthSnapshot
	if err := json.Unmarshal(ctx.Response.Body(), &snap); err != nil {
		t.Fatalf("failed to parse health snapshot: %v", err)
	}
	if snap.Status != "ok" {
		t.Errorf("expected status=ok, got %s", snap.Status)
	}
	if _, ok := snap.Providers["openai"]; !ok {
		t.Error("expected openai in providers map")
	}
}

// --- handleReadiness --------------------------------------------------------

func TestHandleReadiness_NoHealthChecker(t *testing.T) {
	gw := NewGateway(context.Background(), nil, nil)

	ctx := &fasthttp.RequestCtx{}
	gw.handleReadiness(ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200, got %d", ctx.Response.StatusCode())
	}
}

func TestHandleReadiness_Healthy(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai": okProvider("openai"),
	}
	gw := NewGateway(context.Background(), provs, nil)
	defer gw.health.Close()

	ctx := &fasthttp.RequestCtx{}
	gw.handleReadiness(ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200, got %d", ctx.Response.StatusCode())
	}

	var resp map[string]string
	if err := json.Unmarshal(ctx.Response.Body(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp["status"] != "ok" {
		t.Errorf("expected status=ok, got %s", resp["status"])
	}
}

// --- handleEmbeddings -------------------------------------------------------

func TestHandleEmbeddings_NoModel(t *testing.T) {
	// Embeddings dispatcher now returns 400 when 'model' is missing.
	gw := NewGateway(context.Background(), nil, nil)
	client, cleanup := serveRouter(t, gw)
	defer cleanup()

	req, _ := http.NewRequest("POST", "http://test/v1/embeddings",
		bReader([]byte(`{"input":"hello"}`)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}
}

func TestHandleEmbeddings_NoProviders(t *testing.T) {
	// When no providers are configured, dispatcher returns 502.
	gw := NewGateway(context.Background(), nil, nil)
	client, cleanup := serveRouter(t, gw)
	defer cleanup()

	req, _ := http.NewRequest("POST", "http://test/v1/embeddings",
		bReader([]byte(`{"model":"text-embedding-3-small","input":"hello"}`)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("expected 502, got %d", resp.StatusCode)
	}
}

// --- handleChatCompletions / handleCompletions (via in-memory server) --------

func TestHandleChatCompletions_DelegatesToDispatch(t *testing.T) {
	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": okProvider("openai"),
	}, nil)

	client, cleanup := serveRouter(t, gw)
	defer cleanup()

	req, _ := http.NewRequest("POST", "http://test/v1/chat/completions",
		bReader([]byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"mock"}]}`)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

func TestHandleCompletions_DelegatesToDispatch(t *testing.T) {
	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": okProvider("openai"),
	}, nil)

	client, cleanup := serveRouter(t, gw)
	defer cleanup()

	req, _ := http.NewRequest("POST", "http://test/v1/completions",
		bReader([]byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"mock"}]}`)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

// --- writeJSON --------------------------------------------------------------

func TestWriteJSON(t *testing.T) {
	ctx := &fasthttp.RequestCtx{}
	writeJSON(ctx, map[string]string{"key": "value"})

	if string(ctx.Response.Header.ContentType()) != "application/json" {
		t.Errorf("expected application/json, got %s", string(ctx.Response.Header.ContentType()))
	}

	var resp map[string]string
	if err := json.Unmarshal(ctx.Response.Body(), &resp); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}
	if resp["key"] != "value" {
		t.Errorf("expected key=value, got %v", resp["key"])
	}
}
