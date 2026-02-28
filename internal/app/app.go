// Package app wires up all subsystems and owns the application lifecycle.
//
// Startup order:
//  1. initInfra  — external connections (Redis when needed)
//  2. initProviders — LLM provider clients
//  3. initServices — cache, metrics registry
//  4. initGateway  — proxy + management routes
package app

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/redis/go-redis/v9"
	"golang.org/x/sync/errgroup"

	npCache "github.com/nulpointcorp/llm-gateway/internal/cache"
	"github.com/nulpointcorp/llm-gateway/internal/config"
	"github.com/nulpointcorp/llm-gateway/internal/logger"
	"github.com/nulpointcorp/llm-gateway/internal/metrics"
	"github.com/nulpointcorp/llm-gateway/internal/providers"
	anthropicprov "github.com/nulpointcorp/llm-gateway/internal/providers/anthropic"
	azureprov "github.com/nulpointcorp/llm-gateway/internal/providers/azure"
	bedrockprov "github.com/nulpointcorp/llm-gateway/internal/providers/bedrock"
	geminiprov "github.com/nulpointcorp/llm-gateway/internal/providers/gemini"
	mistralprov "github.com/nulpointcorp/llm-gateway/internal/providers/mistral"
	openaiprov "github.com/nulpointcorp/llm-gateway/internal/providers/openai"
	openaicompatprov "github.com/nulpointcorp/llm-gateway/internal/providers/openaicompat"
	vertexaiprov "github.com/nulpointcorp/llm-gateway/internal/providers/vertexai"
	"github.com/nulpointcorp/llm-gateway/internal/proxy"
)

// App owns all long-lived resources and exposes Run / Close.
type App struct {
	version string
	cfg     *config.Config
	baseCtx context.Context
	log     *slog.Logger

	// Optional external connections — nil when not configured.
	rdb *redis.Client

	reqLogger *logger.Logger
	memCache  *npCache.MemoryCache

	prom *metrics.Registry

	provs map[string]providers.Provider
	mgmt  *proxy.ManagementRoutes
	gw    *proxy.Gateway
}

// New initialises all subsystems and returns a ready-to-run App.
// All resources allocated here are released by Close.
func New(ctx context.Context, cfg *config.Config, log *slog.Logger, version string) (*App, error) {
	if ctx == nil {
		return nil, fmt.Errorf("app: context must not be nil")
	}

	a := &App{cfg: cfg, version: version, baseCtx: ctx, log: log}

	steps := []struct {
		name string
		fn   func(context.Context) error
	}{
		{"infra", a.initInfra},
		{"providers", a.initProviders},
		{"services", a.initServices},
		{"gateway", a.initGateway},
	}

	for _, s := range steps {
		if err := s.fn(ctx); err != nil {
			a.Close()
			return nil, fmt.Errorf("app: init %s: %w", s.name, err)
		}
	}

	return a, nil
}

// Run starts the HTTP server and blocks until ctx is cancelled or an error
// occurs. It closes the app gracefully when returning.
func (a *App) Run(ctx context.Context) error {
	addr := fmt.Sprintf(":%d", a.cfg.Port)

	a.log.Info("starting gateway",
		slog.String("version", a.version),
		slog.String("addr", addr),
		slog.String("cache_mode", a.cfg.Cache.Mode),
		slog.Int("providers", len(a.provs)),
	)

	g, gctx := errgroup.WithContext(ctx)

	g.Go(func() error {
		return a.gw.StartWithRoutes(addr, a.mgmt)
	})

	g.Go(func() error {
		<-gctx.Done()
		a.Close()
		return nil
	})

	return g.Wait()
}

// Close releases all resources in reverse-init order. Safe to call multiple
// times and from multiple goroutines.
func (a *App) Close() {
	if a.reqLogger != nil {
		if err := a.reqLogger.Close(); err != nil {
			a.log.Error("logger close error", slog.String("error", err.Error()))
		}
		a.reqLogger = nil
	}
	if a.memCache != nil {
		a.memCache.Close()
		a.memCache = nil
	}
	if a.rdb != nil {
		if err := a.rdb.Close(); err != nil {
			a.log.Error("redis close error", slog.String("error", err.Error()))
		}
		a.rdb = nil
	}
}

// ── Private helpers ──────────────────────────────────────────────────────────

// connectRedis parses the URL and verifies connectivity with a PING.
// Returns an error — callers decide whether to fatal or degrade.
func connectRedis(ctx context.Context, url string) (*redis.Client, error) {
	opts, err := redis.ParseURL(url)
	if err != nil {
		return nil, fmt.Errorf("parse url: %w", err)
	}

	rdb := redis.NewClient(opts)
	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if err := rdb.Ping(pingCtx).Err(); err != nil {
		_ = rdb.Close()
		return nil, fmt.Errorf("ping: %w", err)
	}

	return rdb, nil
}

// redisPinger returns a zero-argument probe function suitable for the
// HealthChecker. Reuses the existing client — no new connections.
func redisPinger(ctx context.Context, rdb *redis.Client) func() bool {
	return func() bool {
		pingCtx, cancel := context.WithTimeout(ctx, time.Second)
		defer cancel()
		return rdb.Ping(pingCtx).Err() == nil
	}
}

// buildProviders creates a provider map from non-empty API keys / credentials.
func buildProviders(ctx context.Context, cfg *config.Config) map[string]providers.Provider {
	provs := make(map[string]providers.Provider)

	// ── Original four ─────────────────────────────────────────────────────────
	if cfg.OpenAI.APIKey != "" {
		var openaiOpts []openaiprov.Option
		if cfg.OpenAI.BaseURL != "" {
			openaiOpts = append(openaiOpts, openaiprov.WithBaseURL(cfg.OpenAI.BaseURL))
		}
		provs["openai"] = openaiprov.New(cfg.OpenAI.APIKey, openaiOpts...)
	}
	if cfg.Anthropic.APIKey != "" {
		var anthropicOpts []anthropicprov.Option
		if cfg.Anthropic.BaseURL != "" {
			anthropicOpts = append(anthropicOpts, anthropicprov.WithBaseURL(cfg.Anthropic.BaseURL))
		}
		provs["anthropic"] = anthropicprov.New(cfg.Anthropic.APIKey, anthropicOpts...)
	}
	if cfg.Gemini.APIKey != "" {
		var geminiOpts []geminiprov.Option
		if cfg.Gemini.BaseURL != "" {
			geminiOpts = append(geminiOpts, geminiprov.WithBaseURL(cfg.Gemini.BaseURL))
		}
		provs["gemini"] = geminiprov.New(ctx, cfg.Gemini.APIKey, geminiOpts...)
	}
	if cfg.Mistral.APIKey != "" {
		var mistralOpts []mistralprov.Option
		if cfg.Mistral.BaseURL != "" {
			mistralOpts = append(mistralOpts, mistralprov.WithBaseURL(cfg.Mistral.BaseURL))
		}
		provs["mistral"] = mistralprov.New(cfg.Mistral.APIKey, mistralOpts...)
	}

	// ── OpenAI-compatible providers ───────────────────────────────────────────
	type ocEntry struct {
		key     string
		name    string
		baseURL string
	}
	ocProviders := []ocEntry{
		{cfg.XAI.APIKey, "xai", "https://api.x.ai/v1"},
		{cfg.DeepSeek.APIKey, "deepseek", "https://api.deepseek.com/v1"},
		{cfg.Groq.APIKey, "groq", "https://api.groq.com/openai/v1"},
		{cfg.Together.APIKey, "together", "https://api.together.xyz/v1"},
		{cfg.Perplexity.APIKey, "perplexity", "https://api.perplexity.ai"},
		{cfg.Cerebras.APIKey, "cerebras", "https://api.cerebras.ai/v1"},
		{cfg.Moonshot.APIKey, "moonshot", "https://api.moonshot.cn/v1"},
		{cfg.MiniMax.APIKey, "minimax", "https://api.minimax.chat/v1"},
		{cfg.Qwen.APIKey, "qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"},
		{cfg.Nebius.APIKey, "nebius", "https://api.studio.nebius.ai/v1"},
		{cfg.NovitaAI.APIKey, "novita", "https://api.novita.ai/v3/openai"},
		{cfg.ByteDance.APIKey, "bytedance", "https://ark.cn-beijing.volces.com/api/v3"},
		{cfg.ZAI.APIKey, "zai", "https://api.z.ai/api/openai/v1"},
		{cfg.CanopyWave.APIKey, "canopywave", "https://api.canopywave.com/v1"},
		{cfg.Inference.APIKey, "inference", "https://api.inference.net/v1"},
		{cfg.NanoGPT.APIKey, "nanogpt", "https://nano-gpt.com/api/v1"},
	}
	for _, e := range ocProviders {
		if e.key != "" {
			provs[e.name] = openaicompatprov.New(e.name, e.key, e.baseURL)
		}
	}

	// ── Google Vertex AI ──────────────────────────────────────────────────────
	if cfg.VertexAI.Project != "" {
		loc := cfg.VertexAI.Location
		var opts []vertexaiprov.Option
		if loc != "" {
			opts = append(opts, vertexaiprov.WithLocation(loc))
		}
		if p, err := vertexaiprov.New(ctx, cfg.VertexAI.Project, opts...); err == nil {
			provs["vertexai"] = p
		}
	}

	// ── AWS Bedrock ───────────────────────────────────────────────────────────
	if cfg.Bedrock.AccessKey != "" && cfg.Bedrock.SecretKey != "" && cfg.Bedrock.Region != "" {
		var opts []bedrockprov.Option
		if cfg.Bedrock.SessionToken != "" {
			opts = append(opts, bedrockprov.WithSessionToken(cfg.Bedrock.SessionToken))
		}
		if cfg.Bedrock.EndpointURL != "" {
			opts = append(opts, bedrockprov.WithEndpointURL(cfg.Bedrock.EndpointURL))
		}
		provs["bedrock"] = bedrockprov.New(
			cfg.Bedrock.AccessKey, cfg.Bedrock.SecretKey, cfg.Bedrock.Region, opts...,
		)
	}

	// ── Azure OpenAI ──────────────────────────────────────────────────────────
	if cfg.Azure.APIKey != "" && cfg.Azure.Endpoint != "" {
		apiVersion := cfg.Azure.APIVersion
		if apiVersion == "" {
			apiVersion = "2024-12-01-preview"
		}
		provs["azure"] = azureprov.New(cfg.Azure.Endpoint, cfg.Azure.APIKey, apiVersion)
	}

	return provs
}
