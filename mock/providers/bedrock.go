package main

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"net/http"
	"strings"
)

// newBedrockHandler returns an http.Handler simulating the AWS Bedrock runtime API.
//
// Bedrock uses two endpoints per model:
//
//	POST /model/{modelId}/converse          — non-streaming
//	POST /model/{modelId}/converse-stream   — streaming
//	GET  /foundation-models                 — health check (listFoundationModels)
func newBedrockHandler(cfg Config) http.Handler {
	mux := http.NewServeMux()

	// Match both /model/{id}/converse and /model/{id}/converse-stream
	mux.HandleFunc("/model/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed", "method_not_allowed")
			return
		}

		path := r.URL.Path
		modelID := extractBedrockModel(path)
		isStream := strings.HasSuffix(path, "/converse-stream")

		applyLatency(cfg)
		if shouldError(cfg) {
			writeBedrockError(w, http.StatusInternalServerError, "mock internal error", "ServiceUnavailableException")
			return
		}

		if isStream {
			serveBedrockStream(w, modelID, cfg)
		} else {
			serveBedrockConverse(w, modelID, cfg)
		}
	})

	// GET /foundation-models — health check
	mux.HandleFunc("/foundation-models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"modelSummaries": []map[string]any{
				{
					"modelId":   "anthropic.claude-3-5-sonnet-20241022-v2:0",
					"modelName": "Claude 3.5 Sonnet",
					"providerName": "Anthropic",
				},
				{
					"modelId":   "amazon.titan-text-express-v1",
					"modelName": "Titan Text Express",
					"providerName": "Amazon",
				},
			},
		})
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeBedrockError(w, http.StatusNotFound, fmt.Sprintf("mock: unknown path %s", r.URL.Path), "ResourceNotFoundException")
	})

	return mux
}

func serveBedrockConverse(w http.ResponseWriter, modelID string, cfg Config) {
	content := fakeSentence(cfg.StreamWords)

	writeJSON(w, http.StatusOK, map[string]any{
		"output": map[string]any{
			"message": map[string]any{
				"role": "assistant",
				"content": []map[string]string{
					{"text": content},
				},
			},
		},
		"stopReason": "end_turn",
		"usage": map[string]int{
			"inputTokens":  12,
			"outputTokens": cfg.StreamWords,
			"totalTokens":  12 + cfg.StreamWords,
		},
		"metrics": map[string]int{
			"latencyMs": 100,
		},
		"additionalModelResponseFields": nil,
		// Returned for identification in tests
		"model": modelID,
	})
}

func serveBedrockStream(w http.ResponseWriter, _ string, cfg Config) {
	// Bedrock streaming uses HTTP/1.1 chunked responses where each line is
	// a newline-delimited JSON event (simplified from the actual binary framing).
	w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)
	content := fakeSentence(cfg.StreamWords)

	sendEvent := func(ev any) {
		data, _ := json.Marshal(ev)
		fmt.Fprintf(w, "data:%s\n", data)
		if flusher != nil {
			flusher.Flush()
		}
	}

	// messageStart
	sendEvent(map[string]any{
		"messageStart": map[string]string{"role": "assistant"},
	})

	// contentBlockStart
	sendEvent(map[string]any{
		"contentBlockStart": map[string]any{
			"start": map[string]any{"text": ""},
			"contentBlockIndex": 0,
		},
	})

	// contentBlockDelta for each word
	words := strings.Fields(content)
	for _, word := range words {
		sendEvent(map[string]any{
			"contentBlockDelta": map[string]any{
				"delta": map[string]string{"text": word + " "},
				"contentBlockIndex": 0,
			},
		})
	}

	// contentBlockStop
	sendEvent(map[string]any{
		"contentBlockStop": map[string]int{"contentBlockIndex": 0},
	})

	// messageStop
	sendEvent(map[string]any{
		"messageStop": map[string]any{
			"stopReason": "end_turn",
			"additionalModelResponseFields": nil,
		},
	})

	// metadata
	sendEvent(map[string]any{
		"metadata": map[string]any{
			"usage": map[string]any{
				"inputTokens":  12,
				"outputTokens": cfg.StreamWords,
				"totalTokens":  12 + cfg.StreamWords,
			},
			"metrics": map[string]any{
				"latencyMs": 100,
			},
			"trace": nil,
		},
	})

	// Bedrock streaming mock: signal end with an id
	sendEvent(map[string]any{"id": fmt.Sprintf("mock-%x", rand.Int64())})
}

func writeBedrockError(w http.ResponseWriter, status int, msg, errType string) {
	writeJSON(w, status, map[string]any{
		"message": msg,
		"__type":  errType,
	})
}

// extractBedrockModel extracts the model ID from a path like
// /model/anthropic.claude-3-5-sonnet-20241022-v2:0/converse
func extractBedrockModel(path string) string {
	const prefix = "/model/"
	if !strings.HasPrefix(path, prefix) {
		return "unknown"
	}
	rest := path[len(prefix):]
	if idx := strings.Index(rest, "/"); idx >= 0 {
		return rest[:idx]
	}
	return rest
}
