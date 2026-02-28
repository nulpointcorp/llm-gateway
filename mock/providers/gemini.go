package main

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"net/http"
	"strings"
)

// newGeminiHandler returns an http.Handler simulating the Google Gemini API.
//
// The Gemini SDK (google.golang.org/genai) communicates with:
//
//	POST {base}/models/{model}:generateContent
//	POST {base}/models/{model}:streamGenerateContent
//	POST {base}/models/{model}:embedContent
//	GET  {base}/models           (list models — used by health check)
//
// where {base} defaults to https://generativelanguage.googleapis.com/v1beta.
func newGeminiHandler(cfg Config) http.Handler {
	mux := http.NewServeMux()

	// generateContent — non-streaming
	mux.HandleFunc("/v1beta/models/", func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path // e.g. /v1beta/models/gemini-1.5-pro:generateContent
		model := extractModel(path)

		switch {
		case strings.HasSuffix(path, ":generateContent"):
			if r.Method != http.MethodPost {
				writeError(w, http.StatusMethodNotAllowed, "method not allowed", "method_not_allowed")
				return
			}
			applyLatency(cfg)
			if shouldError(cfg) {
				writeGeminiError(w, http.StatusInternalServerError, "mock internal error")
				return
			}
			handleGeminiGenerate(w, r, cfg, model, false)

		case strings.HasSuffix(path, ":streamGenerateContent"):
			if r.Method != http.MethodPost {
				writeError(w, http.StatusMethodNotAllowed, "method not allowed", "method_not_allowed")
				return
			}
			applyLatency(cfg)
			if shouldError(cfg) {
				writeGeminiError(w, http.StatusInternalServerError, "mock internal error")
				return
			}
			handleGeminiGenerate(w, r, cfg, model, true)

		case strings.HasSuffix(path, ":embedContent"):
			if r.Method != http.MethodPost {
				writeError(w, http.StatusMethodNotAllowed, "method not allowed", "method_not_allowed")
				return
			}
			applyLatency(cfg)
			handleGeminiEmbed(w, r, model)

		case strings.HasSuffix(path, ":batchEmbedContents"):
			if r.Method != http.MethodPost {
				writeError(w, http.StatusMethodNotAllowed, "method not allowed", "method_not_allowed")
				return
			}
			applyLatency(cfg)
			handleGeminiBatchEmbed(w, r, model)

		default:
			writeGeminiError(w, http.StatusNotFound, fmt.Sprintf("mock: unknown path %s", path))
		}
	})

	// GET /v1beta/models — health check
	mux.HandleFunc("/v1beta/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"models": []map[string]any{
				{
					"name":        "models/gemini-1.5-pro",
					"displayName": "Gemini 1.5 Pro",
					"description": "Mock Gemini 1.5 Pro",
				},
				{
					"name":        "models/gemini-2.0-flash",
					"displayName": "Gemini 2.0 Flash",
					"description": "Mock Gemini 2.0 Flash",
				},
			},
		})
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeGeminiError(w, http.StatusNotFound, fmt.Sprintf("mock: unknown path %s", r.URL.Path))
	})

	return mux
}

func handleGeminiGenerate(w http.ResponseWriter, _ *http.Request, cfg Config, model string, stream bool) {
	id := fmt.Sprintf("gemini-%x", rand.Int64())
	content := fakeSentence(cfg.StreamWords)
	inTokens := 10
	outTokens := cfg.StreamWords

	candidate := map[string]any{
		"content": map[string]any{
			"role": "model",
			"parts": []map[string]string{
				{"text": content},
			},
		},
		"finishReason": "STOP",
		"index":        0,
	}

	resp := map[string]any{
		"candidates": []any{candidate},
		"usageMetadata": map[string]int{
			"promptTokenCount":     inTokens,
			"candidatesTokenCount": outTokens,
			"totalTokenCount":      inTokens + outTokens,
		},
		"responseId": id,
		"modelVersion": model,
	}

	if stream {
		// Gemini streaming returns a JSON array of GenerateContentResponse objects.
		// In practice the SDK uses SSE; the genai package uses newline-delimited JSON.
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode([]any{resp})
		return
	}

	writeJSON(w, http.StatusOK, resp)
}

func handleGeminiEmbed(w http.ResponseWriter, _ *http.Request, _ string) {
	writeJSON(w, http.StatusOK, map[string]any{
		"embedding": map[string]any{
			"values": fakeEmbedding(768),
		},
	})
}

func handleGeminiBatchEmbed(w http.ResponseWriter, r *http.Request, _ string) {
	var req struct {
		Requests []any `json:"requests"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)

	n := len(req.Requests)
	if n == 0 {
		n = 1
	}

	embeddings := make([]map[string]any, n)
	for i := range embeddings {
		embeddings[i] = map[string]any{
			"embedding": map[string]any{
				"values": fakeEmbedding(768),
			},
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"embeddings": embeddings,
	})
}

func writeGeminiError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"code":    status,
			"message": msg,
			"status":  "INTERNAL",
		},
	})
}

// extractModel pulls the model name out of a path like
// /v1beta/models/gemini-1.5-pro:generateContent
func extractModel(path string) string {
	// strip leading /v1beta/models/
	const prefix = "/v1beta/models/"
	if idx := strings.Index(path, prefix); idx >= 0 {
		rest := path[idx+len(prefix):]
		if col := strings.Index(rest, ":"); col >= 0 {
			return rest[:col]
		}
		return rest
	}
	return "gemini-1.5-pro"
}
