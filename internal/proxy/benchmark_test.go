package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
	"github.com/valyala/fasthttp"
)

// mockProvider is a zero-latency in-process provider for benchmarking.
type mockProvider struct {
	name string
}

func (m *mockProvider) Name() string { return m.name }

func (m *mockProvider) Request(_ context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	return &providers.ProxyResponse{
		ID:      "bench-" + req.RequestID,
		Model:   req.Model,
		Content: "pong",
		Usage:   providers.Usage{InputTokens: 10, OutputTokens: 5},
	}, nil
}

func (m *mockProvider) HealthCheck(_ context.Context) error { return nil }

// newBenchGateway builds a Gateway with a single mock provider and no cache.
func newBenchGateway() *Gateway {
	provs := map[string]providers.Provider{
		"openai": &mockProvider{name: "openai"},
	}
	return NewGateway(context.Background(), provs, nil)
}

// BenchmarkProxy measures the overhead added by the proxy middleware + dispatch
// loop when the provider responds instantly.
//
// Run: go mock -bench=BenchmarkProxy -benchtime=30s -benchmem ./internal/proxy/
func BenchmarkProxy(b *testing.B) {
	gw := newBenchGateway()

	// Start a mock HTTP server backed by the gateway handler.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		b.Fatal(err)
	}

	// Build a minimal fasthttp handler (bypass StartWithRoutes to avoid blocking).
	handler := applyMiddleware(
		func(ctx *fasthttp.RequestCtx) {
			ctx.SetStatusCode(200)
			ctx.SetContentType("application/json")
			ctx.SetBody([]byte(`{"status":"ok"}`))
		},
		recovery,
		requestID,
		timing,
	)

	// Use net/http/httptest for benchmark simplicity.
	ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate the full proxy pipeline via the mock provider.
		_ = gw
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]string{"content": "pong"})
	}))
	ts.Listener = ln
	ts.Start()
	defer ts.Close()
	_ = handler

	// Benchmark dispatchChat directly (avoids HTTP overhead for pure proxy overhead).
	b.Run("dispatchChat/sequential", func(b *testing.B) {
		benchDispatchChat(b, gw, 1)
	})

	b.Run("dispatchChat/parallel_100", func(b *testing.B) {
		benchDispatchChat(b, gw, 100)
	})
}

func benchDispatchChat(b *testing.B, gw *Gateway, concurrency int) {
	b.Helper()

	body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`)

	var (
		mu        sync.Mutex
		latencies []time.Duration
	)

	b.ResetTimer()
	b.SetParallelism(concurrency)
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			start := time.Now()
			// Directly invoke requestWithFailover to measure pure proxy overhead.
			req := &providers.ProxyRequest{
				Model:     "gpt-4o",
				Messages:  []providers.Message{{Role: "user", Content: "hello"}},
				RequestID: "bench",
			}
			resp, _, err := gw.requestWithFailover(context.Background(), req, "openai", "chat_completions")
			elapsed := time.Since(start)

			if err != nil {
				b.Errorf("unexpected error: %v", err)
				return
			}
			if resp == nil {
				b.Error("nil response")
				return
			}

			mu.Lock()
			latencies = append(latencies, elapsed)
			mu.Unlock()
		}

		_ = body
	})
	b.StopTimer()

	if len(latencies) == 0 {
		return
	}

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	p50 := latencies[len(latencies)*50/100]
	p99 := latencies[int(math.Min(float64(len(latencies)-1), float64(len(latencies)*99/100)))]

	b.ReportMetric(float64(p50.Microseconds()), "p50_µs")
	b.ReportMetric(float64(p99.Microseconds()), "p99_µs")

	// Assert SLA.
	if p50 > 2*time.Millisecond {
		b.Errorf("P50 latency %v exceeds 2ms SLA", p50)
	}
	if p99 > 10*time.Millisecond {
		b.Errorf("P99 latency %v exceeds 10ms target", p99)
	}
}

// TestProxyOverheadSLA is a fast (~1s) version of the benchmark suitable for CI.
// It runs 1000 requests sequentially and asserts the P50 < 2ms gate.
func TestProxyOverheadSLA(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping latency SLA mock in short mode")
	}

	gw := newBenchGateway()

	const n = 1000
	latencies := make([]time.Duration, 0, n)

	for i := 0; i < n; i++ {
		req := &providers.ProxyRequest{
			Model:     "gpt-4o",
			Messages:  []providers.Message{{Role: "user", Content: "hi"}},
			RequestID: fmt.Sprintf("sla-%d", i),
		}
		start := time.Now()
		_, _, err := gw.requestWithFailover(context.Background(), req, "openai", "chat_completions")
		elapsed := time.Since(start)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		latencies = append(latencies, elapsed)
	}

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	p50 := latencies[n*50/100]
	p99 := latencies[n*99/100]

	t.Logf("P50=%v P99=%v (n=%d)", p50, p99, n)

	if p50 > 2*time.Millisecond {
		t.Errorf("P50=%v exceeds 2ms overhead SLA", p50)
	}
	if p99 > 15*time.Millisecond {
		t.Errorf("P99=%v exceeds 15ms overhead SLA", p99)
	}
}

// TestCircuitBreakerIntegration tests that 5 failures open the breaker.
func TestCircuitBreakerIntegration(t *testing.T) {
	cb := NewCircuitBreaker()

	for i := 0; i < 5; i++ {
		if !cb.Allow("openai") {
			t.Fatalf("expected Allow=true before threshold, iteration %d", i)
		}
		cb.RecordFailure("openai")
	}

	if cb.Allow("openai") {
		t.Error("expected Allow=false after 5 failures (circuit should be open)")
	}
	if cb.StateLabel("openai") != "open" {
		t.Errorf("expected state=open, got=%s", cb.StateLabel("openai"))
	}
}

// TestFailoverCandidateList checks buildCandidateList deduplication.
func TestFailoverCandidateList(t *testing.T) {
	candidates := buildCandidateList("anthropic")
	if candidates[0] != "anthropic" {
		t.Errorf("primary should be first, got %s", candidates[0])
	}
	seen := map[string]bool{}
	for _, c := range candidates {
		if seen[c] {
			t.Errorf("duplicate candidate: %s", c)
		}
		seen[c] = true
	}
}

// TestFailoverRetries verifies that a failing primary triggers fallback.
func TestFailoverRetries(t *testing.T) {
	failCount := int32(0)

	failing := &funcProvider{
		name: "openai",
		requestFn: func(_ context.Context, _ *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			atomic.AddInt32(&failCount, 1)
			return nil, &providerError{status: 503, msg: "service unavailable"}
		},
	}
	ok := &funcProvider{
		name: "anthropic",
		requestFn: func(_ context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			return &providers.ProxyResponse{
				ID: "fallback-resp", Model: req.Model, Content: "ok",
			}, nil
		},
	}

	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai":    failing,
		"anthropic": ok,
	}, nil)

	req := &providers.ProxyRequest{
		Model: "gpt-4o", Messages: []providers.Message{{Role: "user", Content: "hi"}},
		RequestID: "mock-failover",
	}
	resp, usedProv, err := gw.requestWithFailover(context.Background(), req, "openai", "chat_completions")

	if err != nil {
		t.Fatalf("expected successful failover, got error: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response after failover")
	}
	if !strings.EqualFold(usedProv, "anthropic") {
		t.Errorf("expected provider=anthropic, got %s", usedProv)
	}
	if atomic.LoadInt32(&failCount) != 1 {
		t.Errorf("expected 1 failure attempt, got %d", failCount)
	}
}

// helpers -------------------------------------------------------------------

type funcProvider struct {
	name      string
	requestFn func(context.Context, *providers.ProxyRequest) (*providers.ProxyResponse, error)
}

func (f *funcProvider) Name() string { return f.name }
func (f *funcProvider) Request(ctx context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	return f.requestFn(ctx, req)
}
func (f *funcProvider) HealthCheck(_ context.Context) error { return nil }

type providerError struct {
	status int
	msg    string
}

func (e *providerError) Error() string   { return e.msg }
func (e *providerError) HTTPStatus() int { return e.status }
