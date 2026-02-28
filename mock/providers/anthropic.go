package main

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"net/http"
	"strings"
	"time"
)

// newAnthropicHandler returns an http.Handler that simulates the Anthropic API.
func newAnthropicHandler(cfg Config) http.Handler {
	mux := http.NewServeMux()

	// POST /v1/messages  — used by chat and streaming
	mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed", "method_not_allowed")
			return
		}
		applyLatency(cfg)
		if shouldError(cfg) {
			writeAnthropicError(w, http.StatusInternalServerError, "mock internal error", "overloaded_error")
			return
		}

		var req struct {
			Model     string `json:"model"`
			MaxTokens int    `json:"max_tokens"`
			Stream    bool   `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeAnthropicError(w, http.StatusBadRequest, "invalid request body", "invalid_request_error")
			return
		}

		model := req.Model
		if model == "" {
			model = "claude-3-5-sonnet-20241022"
		}

		id := fmt.Sprintf("msg_%x", rand.Int64())
		content := fakeSentence(cfg.StreamWords)
		inTokens := 15
		outTokens := cfg.StreamWords

		if req.Stream {
			serveAnthropicStream(w, id, model, content, inTokens, outTokens)
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"id":           id,
			"type":         "message",
			"role":         "assistant",
			"model":        model,
			"stop_reason":  "end_turn",
			"stop_sequence": nil,
			"content": []map[string]string{
				{"type": "text", "text": content},
			},
			"usage": map[string]int{
				"input_tokens":  inTokens,
				"output_tokens": outTokens,
			},
		})
	})

	// GET /v1/models — used by health check
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"data": []map[string]any{
				{"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet", "created_at": time.Now().Unix()},
				{"id": "claude-3-haiku-20240307", "display_name": "Claude 3 Haiku", "created_at": time.Now().Unix()},
			},
			"has_more":     false,
			"first_id":     "claude-3-5-sonnet-20241022",
			"last_id":      "claude-3-haiku-20240307",
		})
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeAnthropicError(w, http.StatusNotFound, fmt.Sprintf("mock: unknown path %s", r.URL.Path), "not_found_error")
	})

	return mux
}

func writeAnthropicError(w http.ResponseWriter, status int, msg, typ string) {
	writeJSON(w, status, map[string]any{
		"type": "error",
		"error": map[string]string{
			"type":    typ,
			"message": msg,
		},
	})
}

// serveAnthropicStream writes SSE events in the Anthropic streaming format.
func serveAnthropicStream(w http.ResponseWriter, id, model, content string, inTokens, outTokens int) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)

	send := func(eventType string, data any) {
		dataBytes, _ := json.Marshal(data)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, dataBytes)
		if flusher != nil {
			flusher.Flush()
		}
	}

	// message_start
	send("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":    id,
			"type":  "message",
			"role":  "assistant",
			"model": model,
			"content": []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]int{
				"input_tokens":  inTokens,
				"output_tokens": 0,
			},
		},
	})

	// content_block_start
	send("content_block_start", map[string]any{
		"type":  "content_block_start",
		"index": 0,
		"content_block": map[string]string{
			"type": "text",
			"text": "",
		},
	})

	// ping
	send("ping", map[string]string{"type": "ping"})

	// content_block_delta events for each word
	words := strings.Fields(content)
	for _, word := range words {
		send("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]string{
				"type": "text_delta",
				"text": word + " ",
			},
		})
	}

	// content_block_stop
	send("content_block_stop", map[string]any{
		"type":  "content_block_stop",
		"index": 0,
	})

	// message_delta
	send("message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]string{
			"stop_reason":   "end_turn",
			"stop_sequence": "",
		},
		"usage": map[string]int{
			"output_tokens": outTokens,
		},
	})

	// message_stop
	send("message_stop", map[string]string{"type": "message_stop"})
}
