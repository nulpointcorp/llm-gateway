package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

func newTestProvider(srv *httptest.Server) *Provider {
	return New("mock-api-key", WithBaseURL(srv.URL))
}

func baseRequest() *providers.ProxyRequest {
	return &providers.ProxyRequest{
		Model:     "gpt-4o",
		Messages:  []providers.Message{{Role: "user", Content: "Hello"}},
		RequestID: "req-mock-1",
	}
}

func TestProvider_Name(t *testing.T) {
	p := New("key")
	if p.Name() != "openai" {
		t.Fatalf("expected 'openai', got %q", p.Name())
	}
}

func TestProvider_Request_Success(t *testing.T) {
	// Minimal chat.completion payload that openai-go/v3 can unmarshal.
	responseBody := map[string]any{
		"id":      "chatcmpl-123",
		"object":  "chat.completion",
		"created": 0,
		"model":   "gpt-4o",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "Hello, world!",
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     10,
			"completion_tokens": 5,
			"total_tokens":      15,
		},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if !strings.HasPrefix(r.URL.Path, "/v1/") {
			t.Errorf("expected path to start with /v1/, got %q", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer mock-api-key" {
			t.Errorf("missing or wrong Authorization header: %s", r.Header.Get("Authorization"))
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(responseBody)
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	resp, err := p.Request(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.ID != "chatcmpl-123" {
		t.Errorf("expected ID 'chatcmpl-123', got %q", resp.ID)
	}
	if resp.Content != "Hello, world!" {
		t.Errorf("expected content 'Hello, world!', got %q", resp.Content)
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("expected 10 input tokens, got %d", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 5 {
		t.Errorf("expected 5 output tokens, got %d", resp.Usage.OutputTokens)
	}
}

func TestProvider_Request_Streaming(t *testing.T) {
	// Minimal chat.completion.chunk payloads for SSE streaming.
	chunks := []string{
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":0,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":0,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":0,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		for _, chunk := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			if ok {
				flusher.Flush()
			}
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer srv.Close()

	req := baseRequest()
	req.Stream = true

	p := newTestProvider(srv)
	resp, err := p.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Stream == nil {
		t.Fatal("expected non-nil Stream channel")
	}

	var content string
	for chunk := range resp.Stream {
		content += chunk.Content
	}

	if content != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", content)
	}
}

func TestProvider_Request_RateLimit(t *testing.T) {
	// OpenAI-style error envelope.
	errBody := map[string]any{
		"error": map[string]any{
			"message": "Rate limit exceeded",
			"type":    "rate_limit_error",
			"code":    "rate_limit_exceeded",
		},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(errBody)
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	_, err := p.Request(context.Background(), baseRequest())
	if err == nil {
		t.Fatal("expected error for 429, got nil")
	}

	provErr, ok := err.(*ProviderError)
	if !ok {
		t.Fatalf("expected *ProviderError, got %T: %v", err, err)
	}
	if provErr.StatusCode != http.StatusTooManyRequests {
		t.Errorf("expected status 429, got %d", provErr.StatusCode)
	}

	// NOTE: current implementation sets Type to "openai_error" for all API errors.
	if provErr.Type != "openai_error" {
		t.Errorf("expected type 'openai_error', got %q", provErr.Type)
	}

	if !strings.Contains(strings.ToLower(provErr.Message), "rate limit") {
		t.Errorf("expected message to contain rate limit text, got %q", provErr.Message)
	}
}

func TestProvider_Request_ServerError(t *testing.T) {
	errBody := map[string]any{
		"error": map[string]any{
			"message": "Service unavailable",
			"type":    "server_error",
		},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(errBody)
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	_, err := p.Request(context.Background(), baseRequest())
	if err == nil {
		t.Fatal("expected error for 503, got nil")
	}

	provErr, ok := err.(*ProviderError)
	if !ok {
		t.Fatalf("expected *ProviderError, got %T: %v", err, err)
	}
	if provErr.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("expected status 503, got %d", provErr.StatusCode)
	}

	// NOTE: current implementation sets Type to "openai_error" for all API errors.
	if provErr.Type != "openai_error" {
		t.Errorf("expected type 'openai_error', got %q", provErr.Type)
	}
}
