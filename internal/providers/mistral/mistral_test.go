package mistral

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

func newTestProvider(srv *httptest.Server) *Provider {
	return New("mock-api-key", WithBaseURL(srv.URL))
}

func baseRequest() *providers.ProxyRequest {
	return &providers.ProxyRequest{
		Model:     "mistral-large-latest",
		Messages:  []providers.Message{{Role: "user", Content: "Hello"}},
		RequestID: "req-mock-1",
	}
}

func TestProvider_Name(t *testing.T) {
	p := New("key")
	if p.Name() != "mistral" {
		t.Fatalf("expected 'mistral', got %q", p.Name())
	}
}

func TestProvider_Request_Success(t *testing.T) {
	responseBody := chatResponse{
		ID:    "cmpl-mistral-123",
		Model: "mistral-large-latest",
		Choices: []choice{
			{Message: &chatMessage{Role: "assistant", Content: "Bonjour le monde!"}},
		},
		Usage: usage{PromptTokens: 8, CompletionTokens: 4},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected path /chat/completions, got %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer mock-api-key" {
			t.Errorf("missing or wrong Authorization header: %s", r.Header.Get("Authorization"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}

		var body chatRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("failed to decode request body: %v", err)
		}
		if body.Model != "mistral-large-latest" {
			t.Errorf("expected model 'mistral-large-latest', got %q", body.Model)
		}
		if len(body.Messages) != 1 || body.Messages[0].Content != "Hello" {
			t.Errorf("unexpected messages: %v", body.Messages)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(responseBody)
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	resp, err := p.Request(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.ID != "cmpl-mistral-123" {
		t.Errorf("expected ID 'cmpl-mistral-123', got %q", resp.ID)
	}
	if resp.Model != "mistral-large-latest" {
		t.Errorf("expected model 'mistral-large-latest', got %q", resp.Model)
	}
	if resp.Content != "Bonjour le monde!" {
		t.Errorf("expected content 'Bonjour le monde!', got %q", resp.Content)
	}
	if resp.Usage.InputTokens != 8 {
		t.Errorf("expected 8 input tokens, got %d", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 4 {
		t.Errorf("expected 4 output tokens, got %d", resp.Usage.OutputTokens)
	}
}

func TestProvider_Request_Streaming(t *testing.T) {
	chunks := []string{
		`{"id":"cmpl-1","model":"mistral-large-latest","choices":[{"delta":{"role":"assistant","content":"Bonjour"},"finish_reason":null}]}`,
		`{"id":"cmpl-1","model":"mistral-large-latest","choices":[{"delta":{"content":" monde"},"finish_reason":null}]}`,
		`{"id":"cmpl-1","model":"mistral-large-latest","choices":[{"delta":{},"finish_reason":"stop"}]}`,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") != "text/event-stream" {
			t.Errorf("expected Accept: text/event-stream, got %s", r.Header.Get("Accept"))
		}

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
	var lastFinishReason string
	for chunk := range resp.Stream {
		content += chunk.Content
		if chunk.FinishReason != "" {
			lastFinishReason = chunk.FinishReason
		}
	}

	if content != "Bonjour monde" {
		t.Errorf("expected 'Bonjour monde', got %q", content)
	}
	if lastFinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop', got %q", lastFinishReason)
	}
}

func TestProvider_Request_RateLimit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(chatResponse{
			Error: &apiErr{
				Message: "Rate limit exceeded",
				Type:    "rate_limit_error",
				Code:    "rate_limit_exceeded",
			},
		})
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
	if provErr.Type != "rate_limit_error" {
		t.Errorf("expected type 'rate_limit_error', got %q", provErr.Type)
	}
	if provErr.Code != "rate_limit_exceeded" {
		t.Errorf("expected code 'rate_limit_exceeded', got %q", provErr.Code)
	}
	if provErr.HTTPStatus() != http.StatusTooManyRequests {
		t.Errorf("HTTPStatus() should return 429, got %d", provErr.HTTPStatus())
	}
}

func TestProvider_Request_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(chatResponse{
			Error: &apiErr{
				Message: "Internal server error",
				Type:    "server_error",
			},
		})
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	_, err := p.Request(context.Background(), baseRequest())
	if err == nil {
		t.Fatal("expected error for 500, got nil")
	}

	provErr, ok := err.(*ProviderError)
	if !ok {
		t.Fatalf("expected *ProviderError, got %T: %v", err, err)
	}
	if provErr.StatusCode != http.StatusInternalServerError {
		t.Errorf("expected status 500, got %d", provErr.StatusCode)
	}
	if provErr.Type != "server_error" {
		t.Errorf("expected type 'server_error', got %q", provErr.Type)
	}
	if provErr.HTTPStatus() != http.StatusInternalServerError {
		t.Errorf("HTTPStatus() should return 500, got %d", provErr.HTTPStatus())
	}
}

func TestProvider_Request_OnlyIncludesFieldsWhenSet(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("failed to decode body: %v", err)
		}

		if _, ok := body["temperature"]; ok {
			t.Errorf("temperature should not be present when zero")
		}
		if _, ok := body["max_tokens"]; ok {
			t.Errorf("max_tokens should not be present when zero")
		}
		if _, ok := body["stream"]; ok {
			t.Errorf("stream should not be present when false")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chatResponse{
			ID:    "id-1",
			Model: "mistral-large-latest",
			Choices: []choice{
				{Message: &chatMessage{Role: "assistant", Content: "ok"}},
			},
		})
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	req := baseRequest()
	_, err := p.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestProvider_Request_IncludesOptionalFieldsWhenSet(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("failed to decode body: %v", err)
		}

		if temp, ok := body["temperature"]; !ok || temp.(float64) != 0.9 {
			t.Errorf("expected temperature=0.9, got %v (present=%v)", temp, ok)
		}
		if maxTok, ok := body["max_tokens"]; !ok || maxTok.(float64) != 512 {
			t.Errorf("expected max_tokens=512, got %v (present=%v)", maxTok, ok)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chatResponse{
			ID:    "id-2",
			Model: "mistral-large-latest",
			Choices: []choice{
				{Message: &chatMessage{Role: "assistant", Content: "ok"}},
			},
		})
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	req := baseRequest()
	req.Temperature = 0.9
	req.MaxTokens = 512
	_, err := p.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestProvider_HealthCheck_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != "/models" {
			t.Errorf("expected path /models, got %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer mock-api-key" {
			t.Errorf("missing or wrong Authorization header: %s", r.Header.Get("Authorization"))
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	if err := p.HealthCheck(context.Background()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestProvider_HealthCheck_Failure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	err := p.HealthCheck(context.Background())
	if err == nil {
		t.Fatal("expected error for 401, got nil")
	}
}
