package main

import (
	"encoding/json"
	"math/rand/v2"
	"net/http"
	"strings"
	"time"
)

// fakeWords is a pool of words used to build mock responses.
var fakeWords = []string{
	"The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
	"Hello", "world", "This", "is", "a", "mock", "response", "from", "the",
	"mock", "provider", "simulating", "a", "real", "LLM", "API", "call",
	"for", "development", "and", "testing", "purposes",
}

// fakeSentence returns a fake response text of roughly n words.
func fakeSentence(n int) string {
	words := make([]string, n)
	for i := range words {
		words[i] = fakeWords[rand.IntN(len(fakeWords))]
	}
	return strings.Join(words, " ") + "."
}

// fakeEmbedding returns a slice of floats simulating an embedding vector.
func fakeEmbedding(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()*2 - 1
	}
	return v
}

// applyLatency sleeps for the configured latency.
func applyLatency(cfg Config) {
	if cfg.LatencyMS > 0 {
		time.Sleep(time.Duration(cfg.LatencyMS) * time.Millisecond)
	}
}

// shouldError returns true if this request should simulate an error.
func shouldError(cfg Config) bool {
	if cfg.ErrorRate <= 0 {
		return false
	}
	return rand.Float64() < cfg.ErrorRate
}

// writeJSON writes v as JSON with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// errorResponse is the generic OpenAI-style error envelope.
type errorResponse struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

func writeError(w http.ResponseWriter, status int, msg, typ string) {
	writeJSON(w, status, errorResponse{Error: errorDetail{
		Message: msg,
		Type:    typ,
		Code:    strings.ToLower(strings.ReplaceAll(typ, " ", "_")),
	}})
}
