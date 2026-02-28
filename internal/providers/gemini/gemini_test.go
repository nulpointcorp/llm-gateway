package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

// --- helpers ---

func newTestProvider(srv *httptest.Server) *Provider {
	// IMPORTANT: baseURL passed to the client should include an API version segment
	// so splitBaseURLAndVersion() can extract APIVersion correctly.
	return New(context.Background(), "mock-api-key", WithBaseURL(srv.URL+"/v1beta"))
}

func baseRequest() *providers.ProxyRequest {
	return &providers.ProxyRequest{
		Model:     "gemini-1.5-pro",
		Messages:  []providers.Message{{Role: "user", Content: "Hello"}},
		RequestID: "req-mock-1",
	}
}

func successResponse(text string) generateResponse {
	return generateResponse{
		Candidates: []candidate{
			{
				Content: content{
					Role:  "model",
					Parts: []part{{Text: text}},
				},
				FinishReason: "STOP",
			},
		},
		UsageMetadata: usageMetadata{
			PromptTokenCount:     10,
			CandidatesTokenCount: 5,
		},
	}
}

// --- tests ---

func TestProvider_Name(t *testing.T) {
	p := New(context.Background(), "key")
	if p == nil {
		t.Fatalf("expected non-nil provider from New()")
	}
	if p.Name() != "gemini" {
		t.Fatalf("expected 'gemini', got %q", p.Name())
	}
}

func TestProvider_Request_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}

		// Verify API key is passed (SDK may use query param OR header)
		gotKey := r.URL.Query().Get("key")
		if gotKey == "" {
			gotKey = r.Header.Get("X-Goog-Api-Key")
		}
		if gotKey != "mock-api-key" {
			t.Errorf("expected api key 'mock-api-key' (query 'key' or header 'X-Goog-Api-Key'), got %q", gotKey)
		}

		// Verify the URL path contains the model and action
		if !contains(r.URL.Path, "gemini-1.5-pro") {
			t.Errorf("expected model in path, got %q", r.URL.Path)
		}
		if !contains(r.URL.Path, "generateContent") {
			t.Errorf("expected generateContent in path, got %q", r.URL.Path)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(successResponse("Hello, world!"))
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	if p == nil {
		t.Fatalf("expected non-nil provider from newTestProvider()")
	}

	resp, err := p.Request(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
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
	// RequestID should be preserved
	if resp.ID != "req-mock-1" {
		t.Errorf("expected ID 'req-mock-1', got %q", resp.ID)
	}
}

func TestProvider_Request_RoleMapping_AssistantToModel(t *testing.T) {
	var capturedBody generateRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&capturedBody); err != nil {
			t.Errorf("failed to decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(successResponse("Sure!"))
	}))
	defer srv.Close()

	req := &providers.ProxyRequest{
		Model: "gemini-1.5-pro",
		Messages: []providers.Message{
			{Role: "user", Content: "What is 2+2?"},
			{Role: "assistant", Content: "4"},
			{Role: "user", Content: "And 3+3?"},
		},
		RequestID: "req-role-mock",
	}

	p := newTestProvider(srv)
	if p == nil {
		t.Fatalf("expected non-nil provider from newTestProvider()")
	}

	_, err := p.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(capturedBody.Contents) != 3 {
		t.Fatalf("expected 3 contents, got %d", len(capturedBody.Contents))
	}

	// Second message (index 1) was "assistant" and must be mapped to "model"
	if capturedBody.Contents[1].Role != "model" {
		t.Errorf("expected role 'model' for assistant message, got %q", capturedBody.Contents[1].Role)
	}
	if len(capturedBody.Contents[1].Parts) == 0 || capturedBody.Contents[1].Parts[0].Text != "4" {
		t.Errorf("expected text '4', got %+v", capturedBody.Contents[1].Parts)
	}

	// User messages stay as "user"
	if capturedBody.Contents[0].Role != "user" {
		t.Errorf("expected role 'user', got %q", capturedBody.Contents[0].Role)
	}
	if capturedBody.Contents[2].Role != "user" {
		t.Errorf("expected role 'user', got %q", capturedBody.Contents[2].Role)
	}
}

func TestProvider_Request_SystemMessage_UsesSystemInstruction(t *testing.T) {
	var capturedBody generateRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&capturedBody); err != nil {
			t.Errorf("failed to decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(successResponse("OK"))
	}))
	defer srv.Close()

	req := &providers.ProxyRequest{
		Model: "gemini-1.5-pro",
		Messages: []providers.Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Hello"},
		},
		RequestID: "req-system-mock",
	}

	p := newTestProvider(srv)
	if p == nil {
		t.Fatalf("expected non-nil provider from newTestProvider()")
	}

	_, err := p.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// System message goes to systemInstruction (NOT to contents)
	if capturedBody.SystemInstruction == nil || len(capturedBody.SystemInstruction.Parts) == 0 {
		t.Fatalf("expected systemInstruction to be set")
	}
	if capturedBody.SystemInstruction.Parts[0].Text != "You are a helpful assistant." {
		t.Errorf("expected systemInstruction text, got %q", capturedBody.SystemInstruction.Parts[0].Text)
	}

	// Only user message remains in contents
	if len(capturedBody.Contents) != 1 {
		t.Fatalf("expected 1 content (user only), got %d", len(capturedBody.Contents))
	}
	if capturedBody.Contents[0].Role != "user" {
		t.Errorf("expected role 'user', got %q", capturedBody.Contents[0].Role)
	}
	if len(capturedBody.Contents[0].Parts) == 0 || capturedBody.Contents[0].Parts[0].Text != "Hello" {
		t.Errorf("expected user message text 'Hello', got %+v", capturedBody.Contents[0].Parts)
	}
}

func TestProvider_Request_RateLimit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprintln(w, `{"error":{"code":429,"message":"Resource has been exhausted (e.g. check quota).","status":"RESOURCE_EXHAUSTED"}}`)
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	if p == nil {
		t.Fatalf("expected non-nil provider from newTestProvider()")
	}

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
	if provErr.Type != "RESOURCE_EXHAUSTED" {
		t.Errorf("expected type 'RESOURCE_EXHAUSTED', got %q", provErr.Type)
	}
	if provErr.Message != "Resource has been exhausted (e.g. check quota)." {
		t.Errorf("unexpected error message: %q", provErr.Message)
	}
}

func TestProvider_Request_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(w, `{"error":{"code":500,"message":"Internal server error","status":"INTERNAL"}}`)
	}))
	defer srv.Close()

	p := newTestProvider(srv)
	if p == nil {
		t.Fatalf("expected non-nil provider from newTestProvider()")
	}

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
	if provErr.HTTPStatus() != http.StatusInternalServerError {
		t.Errorf("HTTPStatus() should return 500, got %d", provErr.HTTPStatus())
	}
}

func TestProvider_Request_Streaming(t *testing.T) {
	chunks := []string{
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]},"finishReason":""}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]},"finishReason":""}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"}]}`,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !contains(r.URL.Path, "streamGenerateContent") {
			t.Errorf("expected streamGenerateContent in path, got %q", r.URL.Path)
		}
		if r.URL.Query().Get("alt") != "sse" {
			t.Errorf("expected alt=sse query param, got %q", r.URL.Query().Get("alt"))
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
	}))
	defer srv.Close()

	req := baseRequest()
	req.Stream = true

	p := newTestProvider(srv)
	if p == nil {
		t.Fatalf("expected non-nil provider from newTestProvider()")
	}

	resp, err := p.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Stream == nil {
		t.Fatal("expected non-nil Stream channel")
	}

	var content string
	var finalReason string
	for chunk := range resp.Stream {
		content += chunk.Content
		if chunk.FinishReason != "" {
			finalReason = chunk.FinishReason
		}
	}

	if content != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", content)
	}
	if finalReason != "STOP" {
		t.Errorf("expected finish reason 'STOP', got %q", finalReason)
	}
}

func TestProvider_Request_NoIDFallback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(successResponse("Hi"))
	}))
	defer srv.Close()

	req := baseRequest()
	req.RequestID = "" // No request ID provided

	p := newTestProvider(srv)
	if p == nil {
		t.Fatalf("expected non-nil provider from newTestProvider()")
	}

	resp, err := p.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.ID == "" {
		t.Error("expected a generated ID when RequestID is empty, got empty string")
	}
	if !contains(resp.ID, "gemini-") {
		t.Errorf("expected generated ID to start with 'gemini-', got %q", resp.ID)
	}
}

func TestProvider_Request_GenerationConfig(t *testing.T) {
	var capturedBody generateRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&capturedBody); err != nil {
			t.Errorf("failed to decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(successResponse("Response"))
	}))
	defer srv.Close()

	req := baseRequest()
	req.Temperature = 0.7
	req.MaxTokens = 1000

	p := newTestProvider(srv)
	if p == nil {
		t.Fatalf("expected non-nil provider from newTestProvider()")
	}

	_, err := p.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if capturedBody.GenerationConfig == nil {
		t.Fatal("expected generationConfig to be set")
	}
	if capturedBody.GenerationConfig.Temperature == nil || *capturedBody.GenerationConfig.Temperature != 0.7 {
		t.Errorf("expected temperature 0.7, got %v", capturedBody.GenerationConfig.Temperature)
	}
	if capturedBody.GenerationConfig.MaxOutputTokens == nil || *capturedBody.GenerationConfig.MaxOutputTokens != 1000 {
		t.Errorf("expected maxOutputTokens 1000, got %v", capturedBody.GenerationConfig.MaxOutputTokens)
	}
}

func TestProvider_Request_NoGenerationConfig_WhenZero(t *testing.T) {
	var capturedBody generateRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&capturedBody); err != nil {
			t.Errorf("failed to decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(successResponse("Response"))
	}))
	defer srv.Close()

	req := baseRequest()
	req.Temperature = 0
	req.MaxTokens = 0

	p := newTestProvider(srv)
	if p == nil {
		t.Fatalf("expected non-nil provider from newTestProvider()")
	}

	_, err := p.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Depending on SDK serialization this may be nil OR an empty object.
	// If it is present, ensure it has no meaningful values.
	if capturedBody.GenerationConfig != nil {
		if capturedBody.GenerationConfig.Temperature != nil {
			t.Errorf("expected temperature to be nil, got %v", capturedBody.GenerationConfig.Temperature)
		}
		if capturedBody.GenerationConfig.MaxOutputTokens != nil {
			t.Errorf("expected maxOutputTokens to be nil, got %v", capturedBody.GenerationConfig.MaxOutputTokens)
		}
	}
}

func TestProviderError_Error(t *testing.T) {
	e := &ProviderError{
		StatusCode: 429,
		Message:    "Rate limit exceeded",
		Type:       "RESOURCE_EXHAUSTED",
		Code:       "429",
	}
	s := e.Error()
	if !contains(s, "gemini:") {
		t.Errorf("error string should contain 'gemini:', got %q", s)
	}
	if !contains(s, "Rate limit exceeded") {
		t.Errorf("error string should contain the message, got %q", s)
	}
}

// --- local JSON shapes used by tests (request capture + response stubs) ---

type generateRequest struct {
	Contents          []content         `json:"contents"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
	SystemInstruction *content          `json:"systemInstruction,omitempty"`
}

type generationConfig struct {
	Temperature     *float32 `json:"temperature,omitempty"`
	MaxOutputTokens *int32   `json:"maxOutputTokens,omitempty"`
}

type generateResponse struct {
	Candidates    []candidate   `json:"candidates"`
	UsageMetadata usageMetadata `json:"usageMetadata,omitempty"`
	ResponseID    string        `json:"responseId,omitempty"`
}

type candidate struct {
	Content      content `json:"content"`
	FinishReason string  `json:"finishReason,omitempty"`
}

type usageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount int `json:"candidatesTokenCount,omitempty"`
}

type content struct {
	Role  string `json:"role,omitempty"`
	Parts []part `json:"parts"`
}

type part struct {
	Text string `json:"text,omitempty"`
}

// contains is a simple substring check helper.
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		func() bool {
			for i := 0; i <= len(s)-len(substr); i++ {
				if s[i:i+len(substr)] == substr {
					return true
				}
			}
			return false
		}())
}
