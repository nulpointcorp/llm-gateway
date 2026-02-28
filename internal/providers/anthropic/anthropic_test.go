package anthropic

import (
	"context"
	"encoding/json"
	"errors"
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
		Model: "claude-3-5-sonnet",
		Messages: []providers.Message{
			{Role: "user", Content: "Hello"},
		},
		RequestID: "req-mock-1",
	}
}

func isMessagesPath(p string) bool {
	return p == "/messages" || p == "/v1/messages"
}

func isModelsPath(p string) bool {
	return p == "/models" || p == "/v1/models"
}

func decodeJSONMap(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.NewDecoder(r.Body).Decode(&m); err != nil {
		t.Fatalf("failed to decode request body as json: %v", err)
	}
	return m
}

func jsonFloatToInt(v any) (int, bool) {
	f, ok := v.(float64)
	if !ok {
		return 0, false
	}
	return int(f), true
}

func systemAsText(v any) (string, bool) {
	switch s := v.(type) {
	case string:
		return s, true
	case []any:
		if len(s) == 0 {
			return "", true
		}

		if m, ok := s[0].(map[string]any); ok {
			if txt, ok := m["text"].(string); ok {
				return txt, true
			}
		}
	}
	return "", false
}

func respondMessageJSON(w http.ResponseWriter, id, model, text string, inTok, outTok int) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":    id,
		"type":  "message",
		"role":  "assistant",
		"model": model,
		"content": []map[string]any{
			{"type": "text", "text": text},
		},
		"stop_reason":   "end_turn",
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  inTok,
			"output_tokens": outTok,
		},
	})
}

func respondErrorJSON(w http.ResponseWriter, status int, errType, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    errType,
			"message": msg,
		},
	})
}

func requireProviderError(t *testing.T, err error, wantStatus int) *ProviderError {
	t.Helper()
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	var pe *ProviderError
	if !errors.As(err, &pe) {
		t.Fatalf("expected error to be *ProviderError (via errors.As), got %T: %v", err, err)
	}
	if pe.StatusCode != wantStatus {
		t.Fatalf("expected status=%d, got %d", wantStatus, pe.StatusCode)
	}
	if pe.HTTPStatus() != wantStatus {
		t.Fatalf("expected HTTPStatus()=%d, got %d", wantStatus, pe.HTTPStatus())
	}
	if pe.Type != "anthropic_error" {
		t.Fatalf("expected Type='anthropic_error', got %q", pe.Type)
	}
	return pe
}

func TestProvider_Name(t *testing.T) {
	p := New("key")
	if p.Name() != "anthropic" {
		t.Fatalf("expected 'anthropic', got %q", p.Name())
	}
}

func TestProvider_Request_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if !isMessagesPath(r.URL.Path) {
			t.Fatalf("expected path ending with /messages, got %s", r.URL.Path)
		}

		if got := r.Header.Get("x-api-key"); got != "mock-api-key" {
			t.Fatalf("missing or wrong x-api-key header: %q", got)
		}
		// Не хардкодим anthropic-version: SDK может поменять значение.
		if got := r.Header.Get("anthropic-version"); got == "" {
			t.Fatalf("expected anthropic-version header to be present")
		}

		body := decodeJSONMap(t, r)

		// model
		if body["model"] != "claude-3-5-sonnet" {
			t.Fatalf("expected model=%q, got %#v", "claude-3-5-sonnet", body["model"])
		}

		// max_tokens default
		if got, ok := jsonFloatToInt(body["max_tokens"]); !ok || got != defaultMaxTokens {
			t.Fatalf("expected max_tokens=%d, got %#v", defaultMaxTokens, body["max_tokens"])
		}

		// system must be absent for this request
		if _, ok := body["system"]; ok {
			t.Fatalf("did not expect system field, got %#v", body["system"])
		}

		// messages array
		msgs, ok := body["messages"].([]any)
		if !ok || len(msgs) != 1 {
			t.Fatalf("expected exactly 1 message, got %#v", body["messages"])
		}
		m0, ok := msgs[0].(map[string]any)
		if !ok {
			t.Fatalf("message[0] not an object: %#v", msgs[0])
		}
		if m0["role"] != "user" {
			t.Fatalf("expected role=user, got %#v", m0["role"])
		}

		respondMessageJSON(w, "msg-123", "claude-3-5-sonnet", "Hello, world!", 10, 5)
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	resp, err := p.Request(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.ID != "msg-123" {
		t.Fatalf("expected ID 'msg-123', got %q", resp.ID)
	}
	if resp.Model != "claude-3-5-sonnet" {
		t.Fatalf("expected model 'claude-3-5-sonnet', got %q", resp.Model)
	}
	if resp.Content != "Hello, world!" {
		t.Fatalf("expected content 'Hello, world!', got %q", resp.Content)
	}
	if resp.Usage.InputTokens != 10 {
		t.Fatalf("expected 10 input tokens, got %d", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 5 {
		t.Fatalf("expected 5 output tokens, got %d", resp.Usage.OutputTokens)
	}
}

func TestProvider_Request_SystemMessageExtraction(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || !isMessagesPath(r.URL.Path) {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}

		body := decodeJSONMap(t, r)

		// System message must be extracted to top-level system field
		sysRaw, ok := body["system"]
		if !ok {
			t.Fatalf("expected system field to be present")
		}
		sysText, ok := systemAsText(sysRaw)
		if !ok {
			t.Fatalf("could not parse system field: %#v", sysRaw)
		}
		if sysText != "You are helpful." {
			t.Fatalf("expected system=%q, got %q", "You are helpful.", sysText)
		}

		// Only the non-system message should be in messages array
		msgs, ok := body["messages"].([]any)
		if !ok || len(msgs) != 1 {
			t.Fatalf("expected 1 message, got %#v", body["messages"])
		}
		m0 := msgs[0].(map[string]any)
		if m0["role"] != "user" {
			t.Fatalf("expected role=user, got %#v", m0["role"])
		}

		respondMessageJSON(w, "msg-456", "claude-3-5-sonnet", "Sure!", 8, 3)
	}))
	defer srv.Close()

	req := &providers.ProxyRequest{
		Model: "claude-3-5-sonnet",
		Messages: []providers.Message{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "Help me"},
		},
	}

	p := newTestProvider(srv)
	resp, err := p.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "Sure!" {
		t.Fatalf("expected content 'Sure!', got %q", resp.Content)
	}
}

func TestProvider_Request_Streaming(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || !isMessagesPath(r.URL.Path) {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		events := []string{
			// Минимально валидный набор SSE-событий для текстового стрима
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-5-sonnet\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\n",
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
		}

		for _, ev := range events {
			fmt.Fprint(w, ev)
			if flusher != nil {
				flusher.Flush()
			}
		}
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

	var content strings.Builder
	for chunk := range resp.Stream {
		content.WriteString(chunk.Content)
	}

	if content.String() != "Hello world" {
		t.Fatalf("expected %q, got %q", "Hello world", content.String())
	}
}

func TestProvider_Request_RateLimit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !isMessagesPath(r.URL.Path) {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		respondErrorJSON(w, http.StatusTooManyRequests, "rate_limit_error", "Rate limit exceeded")
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	_, err := p.Request(context.Background(), baseRequest())
	pe := requireProviderError(t, err, http.StatusTooManyRequests)

	// Сообщение зависит от SDK, но обычно содержит суть.
	if pe.Message == "" {
		t.Fatalf("expected non-empty ProviderError.Message")
	}
}

func TestProvider_Request_ServerError_529(t *testing.T) {
	// 529 is Anthropic's overloaded status code
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !isMessagesPath(r.URL.Path) {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		respondErrorJSON(w, 529, "overloaded_error", "Anthropic is temporarily overloaded")
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	_, err := p.Request(context.Background(), baseRequest())
	_ = requireProviderError(t, err, 529)
}

func TestProvider_Request_ServerError_503(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !isMessagesPath(r.URL.Path) {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		respondErrorJSON(w, http.StatusServiceUnavailable, "server_error", "Service unavailable")
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	_, err := p.Request(context.Background(), baseRequest())
	_ = requireProviderError(t, err, http.StatusServiceUnavailable)
}

func TestProvider_HealthCheck(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || !isModelsPath(r.URL.Path) {
			// HealthCheck делает GET /models?limit=1 (или близко)
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"id": "claude-3-5-sonnet", "type": "model"},
			},
		})
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	if err := p.HealthCheck(context.Background()); err != nil {
		t.Fatalf("unexpected healthcheck error: %v", err)
	}
}

func TestProvider_ProviderError_ErrorString(t *testing.T) {
	e := &ProviderError{
		StatusCode: 429,
		Message:    "Rate limit exceeded",
		Type:       "anthropic_error",
	}
	s := e.Error()
	if s == "" {
		t.Fatal("Error() returned empty string")
	}
	if !strings.Contains(s, "anthropic") {
		t.Fatalf("Error() should mention 'anthropic', got: %s", s)
	}
	if !strings.Contains(s, "429") {
		t.Fatalf("Error() should mention status code, got: %s", s)
	}
}
