package proxy

// load_bench_test.go — end-to-end throughput and latency benchmarks.
//
// These benchmarks measure the full HTTP pipeline through the gateway:
// TCP accept → middleware → dispatch → provider → serialise → write response.
// An in-memory listener is used so network I/O is not a factor.
//
// Usage:
//
//	# Full suite (30s per benchmark)
//	go mock -bench=. -benchtime=30s -benchmem ./internal/proxy/
//
//	# Quick run (10s)
//	go mock -bench=. -benchtime=10s -benchmem ./internal/proxy/
//
//	# Specific benchmark
//	go mock -bench=BenchmarkGateway_CacheHit -benchtime=30s -benchmem ./internal/proxy/

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	npCache "github.com/nulpointcorp/llm-gateway/internal/cache"
	"github.com/nulpointcorp/llm-gateway/internal/providers"
	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttputil"
)

// ── Helpers ──────────────────────────────────────────────────────────────────

// dialTransport satisfies http.RoundTripper by dialling the in-memory listener.
// A new connection is dialled per request so the benchmark reflects raw
// per-request overhead without persistent-connection amortisation.
type dialTransport struct {
	ln *fasthttputil.InmemoryListener
}

func (t *dialTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	conn, err := t.ln.Dial()
	if err != nil {
		return nil, err
	}
	tr := &http.Transport{
		DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
			return conn, nil
		},
	}
	return tr.RoundTrip(req)
}

// benchPayload is a minimal valid chat-completion request body.
var benchPayload = []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`)

// doRequest sends one POST /v1/chat/completions and discards the response body.
func doRequest(client *http.Client) error {
	req, err := http.NewRequest(http.MethodPost, "http://bench/v1/chat/completions",
		bytes.NewReader(benchPayload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	io.Copy(io.Discard, resp.Body) //nolint:errcheck
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}
	return nil
}

// latencyStats computes P50/P95/P99 from a slice of durations.
func latencyStats(d []time.Duration) (p50, p95, p99 time.Duration) {
	if len(d) == 0 {
		return
	}
	sort.Slice(d, func(i, j int) bool { return d[i] < d[j] })
	n := len(d)
	p50 = d[n*50/100]
	p95 = d[int(math.Min(float64(n-1), float64(n*95/100)))]
	p99 = d[int(math.Min(float64(n-1), float64(n*99/100)))]
	return
}

// ── Baseline: raw fasthttp handler, zero gateway logic ───────────────────────

// BenchmarkBaseline_RawHandler measures a minimal fasthttp handler:
// parse request → write JSON. This is the theoretical floor — what you'd get
// with no proxy logic at all.
func BenchmarkBaseline_RawHandler(b *testing.B) {
	for _, concurrency := range []int{1, 50, 200} {
		concurrency := concurrency
		b.Run(fmt.Sprintf("c%d", concurrency), func(b *testing.B) {
			ln := fasthttputil.NewInmemoryListener()
			rawResp := []byte(`{"id":"base","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"pong"},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
			srv := &fasthttp.Server{
				Handler: func(ctx *fasthttp.RequestCtx) {
					ctx.SetStatusCode(200)
					ctx.SetContentType("application/json")
					ctx.SetBody(rawResp)
				},
			}
			go srv.Serve(ln) //nolint:errcheck
			defer ln.Close()

			client := &http.Client{Transport: &dialTransport{ln: ln}}

			var (
				mu        sync.Mutex
				latencies = make([]time.Duration, 0, b.N)
				errCount  int64
			)

			b.SetParallelism(concurrency)
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					start := time.Now()
					if err := doRequest(client); err != nil {
						atomic.AddInt64(&errCount, 1)
					}
					d := time.Since(start)
					mu.Lock()
					latencies = append(latencies, d)
					mu.Unlock()
				}
			})
			b.StopTimer()

			p50, p95, p99 := latencyStats(latencies)
			b.ReportMetric(float64(p50.Microseconds()), "p50_µs")
			b.ReportMetric(float64(p95.Microseconds()), "p95_µs")
			b.ReportMetric(float64(p99.Microseconds()), "p99_µs")
		})
	}
}

// ── Gateway benchmarks ────────────────────────────────────────────────────────

// BenchmarkGateway_CacheMiss measures the full proxy pipeline when the
// provider must be called (cache cold). Provider is an instant in-process mock.
func BenchmarkGateway_CacheMiss(b *testing.B) {
	for _, concurrency := range []int{1, 50, 200} {
		concurrency := concurrency
		b.Run(fmt.Sprintf("c%d", concurrency), func(b *testing.B) {
			gw := NewGateway(context.Background(),
				map[string]providers.Provider{"openai": &mockProvider{name: "openai"}},
				nil, // no cache
			)
			ln := fasthttputil.NewInmemoryListener()
			srv := &fasthttp.Server{
				Handler: applyMiddleware(gw.handleChatCompletions, recovery, requestID, timing),
			}
			go srv.Serve(ln) //nolint:errcheck
			defer ln.Close()

			client := &http.Client{Transport: &dialTransport{ln: ln}}

			var (
				mu        sync.Mutex
				latencies = make([]time.Duration, 0, b.N)
				errCount  int64
			)

			b.SetParallelism(concurrency)
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					start := time.Now()
					if err := doRequest(client); err != nil {
						atomic.AddInt64(&errCount, 1)
					}
					d := time.Since(start)
					mu.Lock()
					latencies = append(latencies, d)
					mu.Unlock()
				}
			})
			b.StopTimer()

			p50, p95, p99 := latencyStats(latencies)
			b.ReportMetric(float64(p50.Microseconds()), "p50_µs")
			b.ReportMetric(float64(p95.Microseconds()), "p95_µs")
			b.ReportMetric(float64(p99.Microseconds()), "p99_µs")
			if errCount > 0 {
				b.Logf("errors: %d", errCount)
			}
		})
	}
}

// BenchmarkGateway_CacheHit measures the proxy pipeline when the response is
// served from the in-memory cache — no provider call, pure serialisation + I/O.
func BenchmarkGateway_CacheHit(b *testing.B) {
	for _, concurrency := range []int{1, 50, 200} {
		concurrency := concurrency
		b.Run(fmt.Sprintf("c%d", concurrency), func(b *testing.B) {
			ctx := context.Background()
			mc := npCache.NewMemoryCache(ctx)
			defer mc.Close()

			gw := NewGateway(ctx,
				map[string]providers.Provider{"openai": &mockProvider{name: "openai"}},
				mc,
			)
			ln := fasthttputil.NewInmemoryListener()
			srv := &fasthttp.Server{
				Handler: applyMiddleware(gw.handleChatCompletions, recovery, requestID, timing),
			}
			go srv.Serve(ln) //nolint:errcheck
			defer ln.Close()

			client := &http.Client{Transport: &dialTransport{ln: ln}}

			// Warm the cache with one request.
			if err := doRequest(client); err != nil {
				b.Fatalf("warmup: %v", err)
			}

			var (
				mu        sync.Mutex
				latencies = make([]time.Duration, 0, b.N)
				errCount  int64
			)

			b.SetParallelism(concurrency)
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					start := time.Now()
					if err := doRequest(client); err != nil {
						atomic.AddInt64(&errCount, 1)
					}
					d := time.Since(start)
					mu.Lock()
					latencies = append(latencies, d)
					mu.Unlock()
				}
			})
			b.StopTimer()

			p50, p95, p99 := latencyStats(latencies)
			b.ReportMetric(float64(p50.Microseconds()), "p50_µs")
			b.ReportMetric(float64(p95.Microseconds()), "p95_µs")
			b.ReportMetric(float64(p99.Microseconds()), "p99_µs")
			if errCount > 0 {
				b.Logf("errors: %d", errCount)
			}
		})
	}
}

// BenchmarkGateway_OverheadVsBaseline runs both the raw handler and the full
// gateway back-to-back at the same concurrency so the numbers are directly
// comparable in one pass.
func BenchmarkGateway_OverheadVsBaseline(b *testing.B) {
	rawResp := []byte(`{"id":"base","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"pong"},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)

	run := func(b *testing.B, client *http.Client) (p50, p95, p99 time.Duration) {
		b.Helper()
		var (
			mu        sync.Mutex
			latencies = make([]time.Duration, 0, b.N)
		)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				start := time.Now()
				doRequest(client) //nolint:errcheck
				mu.Lock()
				latencies = append(latencies, time.Since(start))
				mu.Unlock()
			}
		})
		b.StopTimer()
		return latencyStats(latencies)
	}

	for _, concurrency := range []int{1, 50, 200} {
		concurrency := concurrency
		b.Run(fmt.Sprintf("c%d", concurrency), func(b *testing.B) {
			// ── Baseline ──────────────────────────────────────────────────────
			b.Run("baseline", func(b *testing.B) {
				ln := fasthttputil.NewInmemoryListener()
				srv := &fasthttp.Server{
					Handler: func(ctx *fasthttp.RequestCtx) {
						ctx.SetStatusCode(200)
						ctx.SetContentType("application/json")
						ctx.SetBody(rawResp)
					},
				}
				go srv.Serve(ln) //nolint:errcheck
				defer ln.Close()

				b.SetParallelism(concurrency)
				p50, p95, p99 := run(b, &http.Client{Transport: &dialTransport{ln: ln}})
				b.ReportMetric(float64(p50.Microseconds()), "p50_µs")
				b.ReportMetric(float64(p95.Microseconds()), "p95_µs")
				b.ReportMetric(float64(p99.Microseconds()), "p99_µs")
			})

			// ── Full gateway (cache miss) ──────────────────────────────────
			b.Run("gateway_cold", func(b *testing.B) {
				gw := NewGateway(context.Background(),
					map[string]providers.Provider{"openai": &mockProvider{name: "openai"}},
					nil,
				)
				ln := fasthttputil.NewInmemoryListener()
				srv := &fasthttp.Server{
					Handler: applyMiddleware(gw.handleChatCompletions, recovery, requestID, timing),
				}
				go srv.Serve(ln) //nolint:errcheck
				defer ln.Close()

				b.SetParallelism(concurrency)
				p50, p95, p99 := run(b, &http.Client{Transport: &dialTransport{ln: ln}})
				b.ReportMetric(float64(p50.Microseconds()), "p50_µs")
				b.ReportMetric(float64(p95.Microseconds()), "p95_µs")
				b.ReportMetric(float64(p99.Microseconds()), "p99_µs")
			})

			// ── Full gateway (cache warm) ──────────────────────────────────
			b.Run("gateway_warm", func(b *testing.B) {
				ctx := context.Background()
				mc := npCache.NewMemoryCache(ctx)
				defer mc.Close()

				gw := NewGateway(ctx,
					map[string]providers.Provider{"openai": &mockProvider{name: "openai"}},
					mc,
				)
				ln := fasthttputil.NewInmemoryListener()
				srv := &fasthttp.Server{
					Handler: applyMiddleware(gw.handleChatCompletions, recovery, requestID, timing),
				}
				go srv.Serve(ln) //nolint:errcheck
				defer ln.Close()

				client := &http.Client{Transport: &dialTransport{ln: ln}}
				doRequest(client) //nolint:errcheck // warm cache

				b.SetParallelism(concurrency)
				p50, p95, p99 := run(b, client)
				b.ReportMetric(float64(p50.Microseconds()), "p50_µs")
				b.ReportMetric(float64(p95.Microseconds()), "p95_µs")
				b.ReportMetric(float64(p99.Microseconds()), "p99_µs")
			})
		})
	}
}

// BenchmarkGateway_Throughput measures maximum sustained requests per second
// using a fixed number of goroutines saturating the gateway.
func BenchmarkGateway_Throughput(b *testing.B) {
	for _, concurrency := range []int{1, 10, 50, 100, 200, 500} {
		concurrency := concurrency
		b.Run(fmt.Sprintf("c%d", concurrency), func(b *testing.B) {
			gw := NewGateway(context.Background(),
				map[string]providers.Provider{"openai": &mockProvider{name: "openai"}},
				nil,
			)
			ln := fasthttputil.NewInmemoryListener()
			srv := &fasthttp.Server{
				Handler: applyMiddleware(gw.handleChatCompletions, recovery, requestID, timing),
			}
			go srv.Serve(ln) //nolint:errcheck
			defer ln.Close()

			client := &http.Client{Transport: &dialTransport{ln: ln}}

			var total int64
			b.SetParallelism(concurrency)
			b.ResetTimer()
			start := time.Now()

			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					doRequest(client) //nolint:errcheck
					atomic.AddInt64(&total, 1)
				}
			})

			elapsed := time.Since(start)
			rps := float64(atomic.LoadInt64(&total)) / elapsed.Seconds()
			b.ReportMetric(rps, "req/s")
		})
	}
}
