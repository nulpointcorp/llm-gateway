package proxy

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

func TestBuildCandidateList_PrimaryFirst(t *testing.T) {
	candidates := buildCandidateList("anthropic")
	if candidates[0] != "anthropic" {
		t.Errorf("expected primary first, got %s", candidates[0])
	}
}

func TestBuildCandidateList_NoDuplicates(t *testing.T) {
	for _, primary := range []string{"openai", "anthropic", "gemini", "mistral"} {
		t.Run(primary, func(t *testing.T) {
			candidates := buildCandidateList(primary)
			seen := make(map[string]bool)
			for _, c := range candidates {
				if seen[c] {
					t.Errorf("duplicate candidate: %s", c)
				}
				seen[c] = true
			}
		})
	}
}

func TestBuildCandidateList_ContainsAllDefaults(t *testing.T) {
	candidates := buildCandidateList("openai")
	set := make(map[string]bool)
	for _, c := range candidates {
		set[c] = true
	}
	for _, def := range providers.DefaultFallbackOrder {
		if !set[def] {
			t.Errorf("missing default fallback provider: %s", def)
		}
	}
}

func TestBuildCandidateList_UnknownPrimary(t *testing.T) {
	candidates := buildCandidateList("custom-provider")
	if candidates[0] != "custom-provider" {
		t.Errorf("primary should still be first, got %s", candidates[0])
	}
	// Should include custom + all defaults.
	if len(candidates) != len(providers.DefaultFallbackOrder)+1 {
		t.Errorf("expected %d candidates, got %d",
			len(providers.DefaultFallbackOrder)+1, len(candidates))
	}
}

func TestIsRetryable_5xxErrors(t *testing.T) {
	for _, code := range []int{500, 502, 503, 504} {
		t.Run(fmt.Sprintf("status_%d", code), func(t *testing.T) {
			err := &providerError{status: code, msg: "server error"}
			if !isRetryable(err) {
				t.Errorf("status %d should be retryable", code)
			}
		})
	}
}

func TestIsRetryable_4xxErrors(t *testing.T) {
	for _, code := range []int{400, 401, 403, 404, 422} {
		t.Run(fmt.Sprintf("status_%d", code), func(t *testing.T) {
			err := &providerError{status: code, msg: "client error"}
			if isRetryable(err) {
				t.Errorf("status %d should NOT be retryable", code)
			}
		})
	}
}

func TestIsRetryable_429(t *testing.T) {
	err := &providerError{status: 429, msg: "rate limited"}
	if isRetryable(err) {
		t.Error("429 should NOT be retryable (it's a client-level rate limit)")
	}
}

func TestIsRetryable_Timeout(t *testing.T) {
	if !isRetryable(context.DeadlineExceeded) {
		t.Error("DeadlineExceeded should be retryable")
	}
}

func TestIsRetryable_GenericError(t *testing.T) {
	err := fmt.Errorf("connection refused")
	if !isRetryable(err) {
		t.Error("generic errors should be treated as retryable")
	}
}

func TestClassifyError_Timeout(t *testing.T) {
	if got := classifyError(context.DeadlineExceeded); got != "timeout" {
		t.Errorf("expected 'timeout', got %q", got)
	}
}

func TestClassifyError_HTTPStatus(t *testing.T) {
	err := &providerError{status: 503, msg: "unavailable"}
	if got := classifyError(err); got != "http_503" {
		t.Errorf("expected 'http_503', got %q", got)
	}
}

func TestClassifyError_Unknown(t *testing.T) {
	err := fmt.Errorf("some error")
	if got := classifyError(err); got != "unknown" {
		t.Errorf("expected 'unknown', got %q", got)
	}
}

func TestRequestWithFailover_PrimarySuccess(t *testing.T) {
	var callCount int32
	primary := &funcProvider{
		name: "openai",
		requestFn: func(_ context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			atomic.AddInt32(&callCount, 1)
			return &providers.ProxyResponse{
				ID: "ok", Model: req.Model, Content: "response",
			}, nil
		},
	}

	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": primary,
	}, nil)

	req := &providers.ProxyRequest{
		Model:     "gpt-4o",
		Messages:  []providers.Message{{Role: "user", Content: "hi"}},
		RequestID: "mock-primary",
	}

	resp, usedProv, err := gw.requestWithFailover(context.Background(), req, "openai", "chat_completions")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if usedProv != "openai" {
		t.Errorf("expected provider=openai, got %s", usedProv)
	}
	if resp.Content != "response" {
		t.Errorf("unexpected content: %s", resp.Content)
	}
	if atomic.LoadInt32(&callCount) != 1 {
		t.Errorf("primary should be called exactly once, got %d", callCount)
	}
}

func TestRequestWithFailover_FallbackOnFailure(t *testing.T) {
	failing := &funcProvider{
		name: "openai",
		requestFn: func(_ context.Context, _ *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			return nil, &providerError{status: 500, msg: "internal error"}
		},
	}
	fallback := &funcProvider{
		name: "anthropic",
		requestFn: func(_ context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			return &providers.ProxyResponse{
				ID: "fallback", Model: req.Model, Content: "from anthropic",
			}, nil
		},
	}

	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai":    failing,
		"anthropic": fallback,
	}, nil)

	req := &providers.ProxyRequest{
		Model:     "gpt-4o",
		Messages:  []providers.Message{{Role: "user", Content: "hi"}},
		RequestID: "mock-fallback",
	}

	resp, usedProv, err := gw.requestWithFailover(context.Background(), req, "openai", "chat_completions")
	if err != nil {
		t.Fatalf("expected successful failover, got: %v", err)
	}
	if usedProv != "anthropic" {
		t.Errorf("expected provider=anthropic, got %s", usedProv)
	}
	if resp.Content != "from anthropic" {
		t.Errorf("unexpected content: %s", resp.Content)
	}
}

func TestRequestWithFailover_AllProvidersFail(t *testing.T) {
	failing := &funcProvider{
		name: "openai",
		requestFn: func(_ context.Context, _ *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			return nil, &providerError{status: 500, msg: "down"}
		},
	}

	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": failing,
	}, nil)

	req := &providers.ProxyRequest{
		Model:     "gpt-4o",
		Messages:  []providers.Message{{Role: "user", Content: "hi"}},
		RequestID: "mock-allfail",
	}

	_, _, err := gw.requestWithFailover(context.Background(), req, "openai", "chat_completions")
	if err == nil {
		t.Fatal("expected error when all providers fail")
	}
}

func TestRequestWithFailover_NonRetryableStopsImmediately(t *testing.T) {
	var callCount int32
	failing := &funcProvider{
		name: "openai",
		requestFn: func(_ context.Context, _ *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			atomic.AddInt32(&callCount, 1)
			return nil, &providerError{status: 401, msg: "unauthorized"}
		},
	}
	shouldNotBeCalled := &funcProvider{
		name: "anthropic",
		requestFn: func(_ context.Context, _ *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			atomic.AddInt32(&callCount, 1)
			return &providers.ProxyResponse{ID: "x", Model: "x", Content: "x"}, nil
		},
	}

	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai":    failing,
		"anthropic": shouldNotBeCalled,
	}, nil)

	req := &providers.ProxyRequest{
		Model:     "gpt-4o",
		Messages:  []providers.Message{{Role: "user", Content: "hi"}},
		RequestID: "mock-nonretry",
	}

	_, _, err := gw.requestWithFailover(context.Background(), req, "openai", "chat_completions")
	if err == nil {
		t.Fatal("expected error for 401")
	}
	if atomic.LoadInt32(&callCount) != 1 {
		t.Errorf("expected exactly 1 call (no failover for 4xx), got %d", callCount)
	}
}

func TestRequestWithFailover_CircuitBreakerSkipsOpenProvider(t *testing.T) {
	gw := NewGateway(context.Background(), map[string]providers.Provider{
		"openai": &funcProvider{
			name: "openai",
			requestFn: func(_ context.Context, _ *providers.ProxyRequest) (*providers.ProxyResponse, error) {
				return nil, &providerError{status: 500, msg: "down"}
			},
		},
		"anthropic": okProvider("anthropic"),
	}, nil)

	// Trip the circuit breaker for openai.
	for i := 0; i < providers.CBErrorThreshold; i++ {
		gw.cb.RecordFailure("openai")
	}

	req := &providers.ProxyRequest{
		Model:     "gpt-4o",
		Messages:  []providers.Message{{Role: "user", Content: "hi"}},
		RequestID: "mock-cb-skip",
	}

	resp, usedProv, err := gw.requestWithFailover(context.Background(), req, "openai", "chat_completions")
	if err != nil {
		t.Fatalf("should fallback past open circuit: %v", err)
	}
	if usedProv != "anthropic" {
		t.Errorf("expected anthropic (openai breaker open), got %s", usedProv)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestRequestWithFailover_MaxRetriesRespected(t *testing.T) {
	var callCount int32
	failing := &funcProvider{
		name: "openai",
		requestFn: func(_ context.Context, _ *providers.ProxyRequest) (*providers.ProxyResponse, error) {
			atomic.AddInt32(&callCount, 1)
			return nil, &providerError{status: 500, msg: "down"}
		},
	}
	// Build providers map with multiple failing providers.
	provs := map[string]providers.Provider{
		"openai":    failing,
		"anthropic": &funcProvider{name: "anthropic", requestFn: failing.requestFn},
		"gemini":    &funcProvider{name: "gemini", requestFn: failing.requestFn},
		"mistral":   &funcProvider{name: "mistral", requestFn: failing.requestFn},
	}
	gw := NewGateway(context.Background(), provs, nil)

	req := &providers.ProxyRequest{
		Model:     "gpt-4o",
		Messages:  []providers.Message{{Role: "user", Content: "hi"}},
		RequestID: "mock-maxretries",
	}

	_, _, err := gw.requestWithFailover(context.Background(), req, "openai", "chat_completions")
	if err == nil {
		t.Fatal("expected error")
	}
	if int(atomic.LoadInt32(&callCount)) > providers.MaxRetries {
		t.Errorf("should not exceed MaxRetries=%d, got %d calls",
			providers.MaxRetries, callCount)
	}
}
