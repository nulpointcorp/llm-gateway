package main

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"net/http"
	"strings"
	"time"
)

// newOpenAIHandler returns an http.Handler that simulates the OpenAI API.
// It also handles OpenAI-compatible providers (same wire format).
func newOpenAIHandler(cfg Config) http.Handler {
	mux := http.NewServeMux()

	// Chat completions
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed", "method_not_allowed")
			return
		}
		applyLatency(cfg)
		if shouldError(cfg) {
			writeError(w, http.StatusInternalServerError, "mock internal server error", "server_error")
			return
		}

		var req struct {
			Model    string `json:"model"`
			Stream   bool   `json:"stream"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid request body", "invalid_request")
			return
		}

		model := req.Model
		if model == "" {
			model = "gpt-4o"
		}

		id := fmt.Sprintf("chatcmpl-mock%x", rand.Int64())
		content := fakeSentence(cfg.StreamWords)
		inTokens := 10
		outTokens := cfg.StreamWords

		if req.Stream {
			serveOpenAIStream(w, id, model, content)
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"id":      id,
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]string{
						"role":    "assistant",
						"content": content,
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]int{
				"prompt_tokens":     inTokens,
				"completion_tokens": outTokens,
				"total_tokens":      inTokens + outTokens,
			},
		})
	})

	// Embeddings
	mux.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed", "method_not_allowed")
			return
		}
		applyLatency(cfg)
		if shouldError(cfg) {
			writeError(w, http.StatusInternalServerError, "mock internal server error", "server_error")
			return
		}

		var req struct {
			Model string `json:"model"`
			Input any    `json:"input"` // string or []string
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid request body", "invalid_request")
			return
		}

		// Normalise input to []string
		var inputs []string
		switch v := req.Input.(type) {
		case string:
			inputs = []string{v}
		case []any:
			for _, x := range v {
				if s, ok := x.(string); ok {
					inputs = append(inputs, s)
				}
			}
		}
		if len(inputs) == 0 {
			inputs = []string{""}
		}

		model := req.Model
		if model == "" {
			model = "text-embedding-3-small"
		}

		data := make([]map[string]any, len(inputs))
		for i := range inputs {
			data[i] = map[string]any{
				"object":    "embedding",
				"index":     i,
				"embedding": fakeEmbedding(1536),
			}
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"object": "list",
			"data":   data,
			"model":  model,
			"usage": map[string]int{
				"prompt_tokens": len(inputs) * 5,
				"total_tokens":  len(inputs) * 5,
			},
		})
	})

	// Models list (used by health check)
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": "gpt-4o", "object": "model", "created": 1710000000, "owned_by": "openai"},
				{"id": "gpt-4-turbo", "object": "model", "created": 1710000000, "owned_by": "openai"},
				{"id": "gpt-3.5-turbo", "object": "model", "created": 1710000000, "owned_by": "openai"},
				{"id": "text-embedding-3-small", "object": "model", "created": 1710000000, "owned_by": "openai"},
				{"id": "text-embedding-3-large", "object": "model", "created": 1710000000, "owned_by": "openai"},
			},
		})
	})

	// Catch-all — some SDKs hit sub-paths
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeError(w, http.StatusNotFound, fmt.Sprintf("mock: unknown path %s", r.URL.Path), "not_found")
	})

	return mux
}

// serveOpenAIStream writes an SSE stream of chat completion chunks.
func serveOpenAIStream(w http.ResponseWriter, id, model, content string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)

	words := strings.Fields(content)
	for _, word := range words {
		chunk := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]any{
				{
					"index": 0,
					"delta": map[string]string{
						"content": word + " ",
					},
					"finish_reason": nil,
				},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		if flusher != nil {
			flusher.Flush()
		}
	}

	// Final chunk with finish_reason
	finalChunk := map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{
			{
				"index":         0,
				"delta":         map[string]string{},
				"finish_reason": "stop",
			},
		},
	}
	data, _ := json.Marshal(finalChunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}
