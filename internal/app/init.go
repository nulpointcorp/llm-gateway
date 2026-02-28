package app

import (
	"context"
	"fmt"
	"log/slog"

	npCache "github.com/nulpointcorp/llm-gateway/internal/cache"
	"github.com/nulpointcorp/llm-gateway/internal/metrics"
	"github.com/nulpointcorp/llm-gateway/internal/proxy"
	"github.com/nulpointcorp/llm-gateway/internal/ratelimit"
)

// initInfra establishes optional external connections.
// Redis is only required when CACHE_MODE=redis.
func (a *App) initInfra(ctx context.Context) error {
	if a.cfg.Cache.Mode == "redis" {
		a.log.Info("connecting to redis", slog.String("url", redactURL(a.cfg.Redis.URL)))

		rdb, err := connectRedis(ctx, a.cfg.Redis.URL)
		if err != nil {
			return fmt.Errorf("redis: %w", err)
		}
		a.rdb = rdb
		a.log.Info("redis connected")
	}

	return nil
}

// initProviders builds the LLM provider map. At least one provider must be
// configured — this is enforced by config.Validate() before we reach here.
func (a *App) initProviders(_ context.Context) error {
	a.provs = buildProviders(a.baseCtx, a.cfg)
	if len(a.provs) == 0 {
		return fmt.Errorf("no provider API keys configured")
	}

	names := make([]string, 0, len(a.provs))
	for n := range a.provs {
		names = append(names, n)
	}
	a.log.Info("providers loaded", slog.Any("providers", names))

	return nil
}

// initServices creates the cache backend and Prometheus metrics registry.
func (a *App) initServices(ctx context.Context) error {
	switch a.cfg.Cache.Mode {
	case "redis":
		// ExactCache wraps the already-connected Redis client.
		a.log.Info("cache backend: redis")

	case "memory":
		// MemoryCache — zero external dependencies, not shared across replicas.
		a.memCache = npCache.NewMemoryCache(ctx)
		a.log.Info("cache backend: memory (in-process)")

	case "none":
		a.log.Info("cache backend: disabled")

	default:
		return fmt.Errorf("unknown cache mode: %s", a.cfg.Cache.Mode)
	}

	a.prom = metrics.New()
	a.prom.SetBuildInfo(a.version)

	return nil
}

// initGateway wires together the Gateway with all configured subsystems.
func (a *App) initGateway(_ context.Context) error {
	// ── Determine cache implementation ────────────────────────────────────────
	var cacheImpl npCache.Cache
	var cacheReady func() bool

	switch a.cfg.Cache.Mode {
	case "redis":
		cacheImpl = npCache.NewExactCacheFromClient(a.rdb)
		cacheReady = redisPinger(a.baseCtx, a.rdb)
	case "memory":
		cacheImpl = a.memCache
		cacheReady = func() bool { return true }
	case "none":
		// nil cache — gateway handles nil gracefully (no caching)
	}

	// ── Build the gateway ────────────────────────────────────────────────────
	opts := proxy.GatewayOptions{
		Logger:             a.log,
		MaxRetries:         a.cfg.Failover.MaxRetries,
		ProviderTimeout:    a.cfg.Failover.ProviderTimeout,
		CacheTTL:           a.cfg.Cache.TTL,
		Metrics:            a.prom,
		AllowClientAPIKeys: a.cfg.AllowClientAPIKeys,
		CBConfig: proxy.CBConfig{
			ErrorThreshold:  a.cfg.CircuitBreaker.ErrorThreshold,
			TimeWindow:      a.cfg.CircuitBreaker.TimeWindow,
			HalfOpenTimeout: a.cfg.CircuitBreaker.HalfOpenTimeout,
		},
	}

	gw := proxy.NewGatewayWithOptions(a.baseCtx, a.provs, cacheImpl, cacheReady, opts)

	// ── Optional subsystems ──────────────────────────────────────────────────

	// Rate limiting — only when Redis is available.
	if a.rdb != nil && a.cfg.RateLimit.RPMLimit > 0 {
		gw.SetRateLimiters(ratelimit.NewRPMLimiter(a.rdb, a.cfg.RateLimit.RPMLimit))
		a.log.Info("rate limiting enabled", slog.Int("rpm_limit", a.cfg.RateLimit.RPMLimit))
	}

	// Async request logger — not wired in the open-source build.
	// In the managed version this connects to ClickHouse for analytics.
	// Request metadata is still written via slog (see gateway.go logRequest).

	// CORS.
	gw.SetCORSOrigins(a.cfg.CORSOrigins)

	// Cache exclusions.
	if len(a.cfg.Cache.ExcludeExact) > 0 || len(a.cfg.Cache.ExcludePatterns) > 0 {
		el, err := npCache.NewExclusionList(a.cfg.Cache.ExcludeExact, a.cfg.Cache.ExcludePatterns)
		if err != nil {
			return fmt.Errorf("cache exclusions: %w", err)
		}
		gw.SetCacheExclusions(el)
		a.log.Info("cache exclusions loaded", slog.Int("rules", el.Len()))
	}

	// ── Management routes ────────────────────────────────────────────────────
	a.mgmt = &proxy.ManagementRoutes{
		Metrics: a.prom.Handler(),
	}

	a.gw = gw

	return nil
}

// redactURL replaces the userinfo portion of a URL with "***" for safe logging.
// e.g. "redis://:secret@localhost:6379" → "redis://***@localhost:6379"
func redactURL(raw string) string {
	for i, c := range raw {
		if c == '@' {
			// Find the scheme end ("://") and keep only scheme + "***" + @host.
			for j := i - 1; j >= 0; j-- {
				if j+2 < len(raw) && raw[j:j+3] == "://" {
					return raw[:j+3] + "***" + raw[i:]
				}
			}
			return "***" + raw[i:]
		}
	}
	return raw
}
