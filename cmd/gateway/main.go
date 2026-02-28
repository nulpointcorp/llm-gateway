// Command gateway is the nulpoint LLM proxy server.
//
// It reads configuration from environment variables (or config.example.yaml) and
// starts an OpenAI-compatible HTTP proxy on the configured port.
//
// Quick-start (in-memory cache, no Redis required):
//
//	OPENAI_API_KEY=sk-... ./gateway
//
// See .env.example for all available configuration variables.
package main

import (
	"context"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/nulpointcorp/llm-gateway/internal/app"
	"github.com/nulpointcorp/llm-gateway/internal/config"
)

// version is overridden at build time via -ldflags="-X main.version=x.y.z".
var version = "0.1.0"

func main() {
	// Graceful shutdown on SIGINT / SIGTERM.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Load configuration — exits with a descriptive error if required vars are missing.
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	// Build the structured logger. All subsystems share this instance.
	logger := buildLogger(cfg.LogLevel)
	slog.SetDefault(logger)

	// Initialise and run the application.
	a, err := app.New(ctx, cfg, logger, version)
	if err != nil {
		logger.Error("startup failed", slog.String("error", err.Error()))
		os.Exit(1)
	}
	defer a.Close()

	if err := a.Run(ctx); err != nil {
		logger.Error("gateway stopped", slog.String("error", err.Error()))
		os.Exit(1)
	}
}

// buildLogger constructs a JSON slog.Logger for the given level string.
// Unknown level strings default to INFO.
func buildLogger(level string) *slog.Logger {
	var l slog.Level
	switch level {
	case "debug":
		l = slog.LevelDebug
	case "warn":
		l = slog.LevelWarn
	case "error":
		l = slog.LevelError
	default:
		l = slog.LevelInfo
	}

	return slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level:     l,
		AddSource: l == slog.LevelDebug, // include file:line only in debug mode
	}))
}
