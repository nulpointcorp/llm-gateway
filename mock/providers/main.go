// Command providers runs lightweight HTTP mock servers that simulate each
// LLM provider API.  It is used for E2E/load testing without real credentials.
//
// Each provider listens on its own port:
//
//	OpenAI / OpenAI-compat  :19001
//	Anthropic               :19002
//	Gemini                  :19003
//	Mistral                 :19004
//	Bedrock                 :19005
//
// Environment overrides (PORT_<PROVIDER>):
//
//	PORT_OPENAI, PORT_ANTHROPIC, PORT_GEMINI, PORT_MISTRAL, PORT_BEDROCK
//
// Behaviour flags (via env):
//
//	MOCK_LATENCY_MS   — artificial latency added to every response (default 0)
//	MOCK_ERROR_RATE   — fraction [0,1] of requests that return HTTP 500 (default 0)
//	MOCK_STREAM_WORDS — words in streaming response (default 10)
package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"syscall"
	"time"
)

// Config holds runtime configuration shared across all mock servers.
type Config struct {
	LatencyMS   int
	ErrorRate   float64
	StreamWords int
}

func loadConfig() Config {
	c := Config{StreamWords: 10}

	if v := os.Getenv("MOCK_LATENCY_MS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.LatencyMS = n
		}
	}
	if v := os.Getenv("MOCK_ERROR_RATE"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 && f <= 1 {
			c.ErrorRate = f
		}
	}
	if v := os.Getenv("MOCK_STREAM_WORDS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.StreamWords = n
		}
	}
	return c
}

func portFromEnv(key string, defaultPort int) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return strconv.Itoa(defaultPort)
}

func startServer(name, addr string, h http.Handler, log *slog.Logger) *http.Server {
	srv := &http.Server{
		Addr:         addr,
		Handler:      h,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}
	go func() {
		log.Info("mock provider listening", slog.String("provider", name), slog.String("addr", addr))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("server error", slog.String("provider", name), slog.String("error", err.Error()))
		}
	}()
	return srv
}

func main() {
	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg := loadConfig()

	log.Info("starting mock providers",
		slog.Int("latency_ms", cfg.LatencyMS),
		slog.Float64("error_rate", cfg.ErrorRate),
		slog.Int("stream_words", cfg.StreamWords),
	)

	servers := []*http.Server{
		startServer("openai", ":"+portFromEnv("PORT_OPENAI", 19001), newOpenAIHandler(cfg), log),
		startServer("anthropic", ":"+portFromEnv("PORT_ANTHROPIC", 19002), newAnthropicHandler(cfg), log),
		startServer("gemini", ":"+portFromEnv("PORT_GEMINI", 19003), newGeminiHandler(cfg), log),
		startServer("mistral", ":"+portFromEnv("PORT_MISTRAL", 19004), newMistralHandler(cfg), log),
		startServer("bedrock", ":"+portFromEnv("PORT_BEDROCK", 19005), newBedrockHandler(cfg), log),
	}

	// Print readiness
	fmt.Println("READY")

	// Wait for signal.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info("shutting down mock providers")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	for _, srv := range servers {
		wg.Add(1)
		go func(s *http.Server) {
			defer wg.Done()
			_ = s.Shutdown(ctx)
		}(srv)
	}
	wg.Wait()
	log.Info("mock providers stopped")
}
