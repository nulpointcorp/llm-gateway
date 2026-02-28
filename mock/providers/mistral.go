package main

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"net/http"
	"strings"
	"time"
)

// newMistralHandler returns an http.Handler simulating the Mistral API.
// Mistral uses the same wire format as OpenAI (chat completions + embeddings).
func newMistralHandler(cfg Config) http.Handler {
	mux := http.NewServeMux()

	// POST /v1/chat/completions
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed", "method_not_allowed")
			return
		}
		applyLatency(cfg)
		if shouldError(cfg) {
			writeMistralError(w, http.StatusInternalServerError, "mock internal error", "server_error")
			return
		}

		var req struct {
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeMistralError(w, http.StatusBadRequest, "invalid request body", "invalid_request")
			return
		}

		model := req.Model
		if model == "" {
			model = "mistral-large-latest"
		}

		id := fmt.Sprintf("cmpl-%x", rand.Int64())
		content := fakeSentence(cfg.StreamWords)
		inTokens := 10
		outTokens := cfg.StreamWords

		if req.Stream {
			serveMistralStream(w, id, model, content)
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

	// POST /v1/embeddings
	mux.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed", "method_not_allowed")
			return
		}
		applyLatency(cfg)
		if shouldError(cfg) {
			writeMistralError(w, http.StatusInternalServerError, "mock internal error", "server_error")
			return
		}

		var req struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeMistralError(w, http.StatusBadRequest, "invalid request body", "invalid_request")
			return
		}

		if len(req.Input) == 0 {
			req.Input = []string{""}
		}
		model := req.Model
		if model == "" {
			model = "mistral-embed"
		}

		data := make([]map[string]any, len(req.Input))
		for i := range req.Input {
			data[i] = map[string]any{
				"object":    "embedding",
				"index":     i,
				"embedding": fakeEmbedding(1024),
			}
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"id":     fmt.Sprintf("embd-%x", rand.Int64()),
			"object": "list",
			"data":   data,
			"model":  model,
			"usage": map[string]int{
				"prompt_tokens": len(req.Input) * 4,
				"total_tokens":  len(req.Input) * 4,
			},
		})
	})

	// GET /v1/models — health check
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": "mistral-large-latest", "object": "model"},
				{"id": "mistral-small-latest", "object": "model"},
				{"id": "mistral-embed", "object": "model"},
			},
		})
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeMistralError(w, http.StatusNotFound, fmt.Sprintf("mock: unknown path %s", r.URL.Path), "not_found")
	})

	return mux
}

func writeMistralError(w http.ResponseWriter, status int, msg, typ string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]string{
			"message": msg,
			"type":    typ,
			"code":    strings.ToLower(strings.ReplaceAll(typ, " ", "_")),
		},
	})
}

func serveMistralStream(w http.ResponseWriter, id, model, content string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)

	words := strings.Fields(content)
	for i, word := range words {
		finishReason := ""
		if i == len(words)-1 {
			finishReason = "stop"
		}

		var deltaContent *string
		wordWithSpace := word + " "
		deltaContent = &wordWithSpace

		chunk := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]any{
				{
					"index": 0,
					"delta": map[string]any{
						"role":    "assistant",
						"content": *deltaContent,
					},
					"finish_reason": finishReason,
				},
			},
		}

		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		if flusher != nil {
			flusher.Flush()
		}
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}
