// Package config loads and validates all runtime configuration for the gateway.
//
// Configuration is read from environment variables (preferred for containers)
// or from a config.example.yaml file in the working directory. Environment variables
// take precedence over the YAML file.
//
// Naming convention: env vars use UPPER_SNAKE_CASE; the YAML file uses the
// same names in lower_snake_case. For example OPENAI_API_KEY becomes
// openai_api_key in YAML.
//
// Only one LLM provider key is strictly required for the gateway to start.
// Redis is optional — set CACHE_MODE=memory to use the built-in in-process
// cache with no external dependencies.
package config

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/viper"
	"github.com/subosito/gotenv"
)

// Config is the top-level configuration container.
type Config struct {
	// Port is the TCP port the HTTP server listens on. Default: 8080.
	Port int

	// LogLevel controls the minimum log level. One of: debug, info, warn, error.
	// Default: info.
	LogLevel string

	// Provider API keys — at least one must be non-empty.
	OpenAI    ProviderConfig
	Anthropic ProviderConfig
	Gemini    ProviderConfig
	Mistral   ProviderConfig

	// OpenAI-compatible providers.
	XAI        ProviderConfig
	DeepSeek   ProviderConfig
	Groq       ProviderConfig
	Together   ProviderConfig
	Perplexity ProviderConfig
	Cerebras   ProviderConfig
	Moonshot   ProviderConfig
	MiniMax    ProviderConfig
	Qwen       ProviderConfig
	Nebius     ProviderConfig
	NovitaAI   ProviderConfig
	ByteDance  ProviderConfig
	ZAI        ProviderConfig
	CanopyWave ProviderConfig
	Inference  ProviderConfig
	NanoGPT    ProviderConfig

	// Google Vertex AI (uses ADC instead of an API key).
	VertexAI VertexAIConfig

	// AWS Bedrock.
	Bedrock BedrockConfig

	// Azure OpenAI.
	Azure AzureConfig

	// Redis holds the connection URL for the Redis-backed cache and rate limiter.
	// Required only when CacheMode is "redis".
	Redis RedisConfig

	// Cache controls caching behaviour.
	Cache CacheConfig

	// CircuitBreaker controls per-provider circuit breaker thresholds.
	CircuitBreaker CircuitBreakerConfig

	// RateLimit controls request-rate limiting.
	RateLimit RateLimitConfig

	// Failover controls multi-provider fallback behaviour.
	Failover FailoverConfig

	// CORSOrigins is the list of allowed CORS origins.
	// Use ["*"] to allow any origin (default). Set to specific origins in prod.
	CORSOrigins []string

	// AppBaseURL is used to construct absolute URLs (e.g. in webhook callbacks).
	AppBaseURL string

	// AllowClientAPIKeys enables forwarding client-supplied Authorization headers
	// directly to the upstream provider. When false (default) the gateway only
	// uses the API keys configured in this file/.env.
	AllowClientAPIKeys bool
}

// ProviderConfig holds configuration for a single LLM provider.
type ProviderConfig struct {
	// APIKey is the provider API key. Leave empty to disable the provider.
	APIKey string

	// BaseURL overrides the provider's default API endpoint.
	// Useful for local mocks and development. Leave empty to use the default.
	BaseURL string
}

// VertexAIConfig holds Google Vertex AI configuration.
// Auth is resolved via Application Default Credentials (ADC).
type VertexAIConfig struct {
	// Project is the Google Cloud project ID. Required.
	Project string
	// Location is the Vertex AI region. Default: "us-central1".
	Location string
}

// BedrockConfig holds AWS Bedrock configuration.
type BedrockConfig struct {
	// AccessKey is the AWS access key ID.
	AccessKey string
	// SecretKey is the AWS secret access key.
	SecretKey string
	// SessionToken is the optional STS session token for temporary credentials.
	SessionToken string
	// Region is the AWS region, e.g. "us-east-1".
	Region string
	// EndpointURL overrides the Bedrock runtime endpoint. Useful for local mocks.
	EndpointURL string
}

// AzureConfig holds Azure OpenAI configuration.
type AzureConfig struct {
	// Endpoint is the Azure OpenAI resource URL,
	// e.g. "https://myresource.openai.azure.com".
	Endpoint string
	// APIKey is the Azure OpenAI resource key.
	APIKey string
	// APIVersion is the API version string, e.g. "2024-12-01-preview".
	APIVersion string
}

// RedisConfig holds Redis connection configuration.
type RedisConfig struct {
	// URL is a redis:// or rediss:// URL. Example: redis://localhost:6379
	URL string
}

// CacheConfig controls the response cache.
type CacheConfig struct {
	// Mode selects the cache backend:
	//   "redis"  — Redis-backed cache (requires REDIS_URL). Recommended for production.
	//   "memory" — In-process TTL cache. No external deps; not shared across replicas.
	//   "none"   — Cache disabled entirely.
	// Default: "memory".
	Mode string

	// TTL is the default time-to-live for cached responses. Default: 1h.
	TTL time.Duration

	// ExcludeExact is a list of exact model names that must never be cached.
	// Example: ["gpt-4o-realtime", "claude-3-haiku"]
	ExcludeExact []string

	// ExcludePatterns is a list of Go regular expressions matched against model
	// names. Requests whose model matches any pattern are not cached.
	// Example: ["^ft:", ".*-preview$"]
	ExcludePatterns []string
}

// CircuitBreakerConfig controls per-provider circuit breaker settings.
type CircuitBreakerConfig struct {
	// ErrorThreshold is the number of consecutive errors that trip the breaker.
	// Default: 5.
	ErrorThreshold int

	// TimeWindow is the rolling window over which errors are counted.
	// Default: 60s.
	TimeWindow time.Duration

	// HalfOpenTimeout is how long the breaker stays open before allowing a
	// single probe request. Default: 30s.
	HalfOpenTimeout time.Duration
}

// RateLimitConfig controls request-rate limiting.
type RateLimitConfig struct {
	// RPMLimit is the maximum requests per minute allowed globally.
	// 0 disables rate limiting. Default: 0.
	RPMLimit int
}

// FailoverConfig controls multi-provider failover.
type FailoverConfig struct {
	// MaxRetries is the maximum number of provider attempts per request
	// (including the first). Default: 3.
	MaxRetries int

	// ProviderTimeout is the per-provider HTTP timeout. Default: 30s.
	ProviderTimeout time.Duration
}

// Load reads configuration from environment variables and (optionally) from
// config.example.yaml in the current working directory.
//
// At least one provider API key must be configured.
// REDIS_URL is only required when CACHE_MODE=redis.
func Load() (*Config, error) {
	if err := loadDotEnv(".env"); err != nil {
		return nil, err
	}

	v := viper.New()

	v.SetConfigName("config")
	v.SetConfigType("yaml")
	v.AddConfigPath(".")

	_ = v.ReadInConfig()

	v.AutomaticEnv()
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	// ── Defaults ──────────────────────────────────────────────────────────────
	v.SetDefault("PORT", 8080)
	v.SetDefault("LOG_LEVEL", "info")
	v.SetDefault("CACHE_MODE", "memory")
	v.SetDefault("CACHE_TTL", "1h")
	v.SetDefault("CORS_ORIGINS", []string{"*"})

	// Circuit breaker defaults.
	v.SetDefault("CB_ERROR_THRESHOLD", 5)
	v.SetDefault("CB_TIME_WINDOW", "60s")
	v.SetDefault("CB_HALF_OPEN_TIMEOUT", "30s")

	// Failover defaults.
	v.SetDefault("MAX_RETRIES", 3)
	v.SetDefault("PROVIDER_TIMEOUT", "30s")

	// Rate limit: 0 = disabled.
	v.SetDefault("RPM_LIMIT", 0)

	// Client API key mode disabled by default.
	v.SetDefault("ALLOW_CLIENT_API_KEYS", false)

	// ── Build config ──────────────────────────────────────────────────────────
	cfg := &Config{
		Port:     v.GetInt("PORT"),
		LogLevel: strings.ToLower(v.GetString("LOG_LEVEL")),

		OpenAI:    ProviderConfig{APIKey: v.GetString("OPENAI_API_KEY"), BaseURL: v.GetString("OPENAI_BASE_URL")},
		Anthropic: ProviderConfig{APIKey: v.GetString("ANTHROPIC_API_KEY"), BaseURL: v.GetString("ANTHROPIC_BASE_URL")},
		Gemini:    ProviderConfig{APIKey: v.GetString("GOOGLE_API_KEY"), BaseURL: v.GetString("GEMINI_BASE_URL")},
		Mistral:   ProviderConfig{APIKey: v.GetString("MISTRAL_API_KEY"), BaseURL: v.GetString("MISTRAL_BASE_URL")},

		// OpenAI-compatible providers
		XAI:        ProviderConfig{APIKey: v.GetString("XAI_API_KEY")},
		DeepSeek:   ProviderConfig{APIKey: v.GetString("DEEPSEEK_API_KEY")},
		Groq:       ProviderConfig{APIKey: v.GetString("GROQ_API_KEY")},
		Together:   ProviderConfig{APIKey: v.GetString("TOGETHER_API_KEY")},
		Perplexity: ProviderConfig{APIKey: v.GetString("PERPLEXITY_API_KEY")},
		Cerebras:   ProviderConfig{APIKey: v.GetString("CEREBRAS_API_KEY")},
		Moonshot:   ProviderConfig{APIKey: v.GetString("MOONSHOT_API_KEY")},
		MiniMax:    ProviderConfig{APIKey: v.GetString("MINIMAX_API_KEY")},
		Qwen:       ProviderConfig{APIKey: v.GetString("QWEN_API_KEY")},
		Nebius:     ProviderConfig{APIKey: v.GetString("NEBIUS_API_KEY")},
		NovitaAI:   ProviderConfig{APIKey: v.GetString("NOVITA_API_KEY")},
		ByteDance:  ProviderConfig{APIKey: v.GetString("BYTEDANCE_API_KEY")},
		ZAI:        ProviderConfig{APIKey: v.GetString("ZAI_API_KEY")},
		CanopyWave: ProviderConfig{APIKey: v.GetString("CANOPYWAVE_API_KEY")},
		Inference:  ProviderConfig{APIKey: v.GetString("INFERENCE_API_KEY")},
		NanoGPT:    ProviderConfig{APIKey: v.GetString("NANOGPT_API_KEY")},

		// Google Vertex AI
		VertexAI: VertexAIConfig{
			Project:  v.GetString("VERTEX_PROJECT"),
			Location: v.GetString("VERTEX_LOCATION"),
		},

		// AWS Bedrock
		Bedrock: BedrockConfig{
			AccessKey:    v.GetString("AWS_ACCESS_KEY_ID"),
			SecretKey:    v.GetString("AWS_SECRET_ACCESS_KEY"),
			SessionToken: v.GetString("AWS_SESSION_TOKEN"),
			Region:       v.GetString("AWS_REGION"),
			EndpointURL:  v.GetString("BEDROCK_ENDPOINT_URL"),
		},

		// Azure OpenAI
		Azure: AzureConfig{
			Endpoint:   v.GetString("AZURE_OPENAI_ENDPOINT"),
			APIKey:     v.GetString("AZURE_OPENAI_API_KEY"),
			APIVersion: v.GetString("AZURE_OPENAI_API_VERSION"),
		},

		Redis: RedisConfig{URL: v.GetString("REDIS_URL")},

		Cache: CacheConfig{
			Mode:            strings.ToLower(v.GetString("CACHE_MODE")),
			TTL:             v.GetDuration("CACHE_TTL"),
			ExcludeExact:    v.GetStringSlice("CACHE_EXCLUDE_EXACT"),
			ExcludePatterns: v.GetStringSlice("CACHE_EXCLUDE_PATTERNS"),
		},

		CircuitBreaker: CircuitBreakerConfig{
			ErrorThreshold:  v.GetInt("CB_ERROR_THRESHOLD"),
			TimeWindow:      v.GetDuration("CB_TIME_WINDOW"),
			HalfOpenTimeout: v.GetDuration("CB_HALF_OPEN_TIMEOUT"),
		},

		RateLimit: RateLimitConfig{
			RPMLimit: v.GetInt("RPM_LIMIT"),
		},

		Failover: FailoverConfig{
			MaxRetries:      v.GetInt("MAX_RETRIES"),
			ProviderTimeout: v.GetDuration("PROVIDER_TIMEOUT"),
		},

		CORSOrigins: v.GetStringSlice("CORS_ORIGINS"),
		AppBaseURL:  v.GetString("APP_BASE_URL"),

		AllowClientAPIKeys: v.GetBool("ALLOW_CLIENT_API_KEYS"),
	}

	// ── Validation ────────────────────────────────────────────────────────────
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	return cfg, nil
}

// validate checks all semantic constraints that cannot be expressed as defaults.
func (c *Config) validate() error {
	// At least one provider must be configured unless client-supplied keys are enabled.
	if !c.AllowClientAPIKeys && !c.AtLeastOneProviderKey() {
		return fmt.Errorf(
			"config: at least one provider API key is required " +
				"(OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, MISTRAL_API_KEY, " +
				"XAI_API_KEY, DEEPSEEK_API_KEY, GROQ_API_KEY, TOGETHER_API_KEY, " +
				"PERPLEXITY_API_KEY, CEREBRAS_API_KEY, MOONSHOT_API_KEY, MINIMAX_API_KEY, " +
				"QWEN_API_KEY, NEBIUS_API_KEY, NOVITA_API_KEY, BYTEDANCE_API_KEY, " +
				"ZAI_API_KEY, CANOPYWAVE_API_KEY, INFERENCE_API_KEY, NANOGPT_API_KEY, " +
				"VERTEX_PROJECT, AWS_ACCESS_KEY_ID, or AZURE_OPENAI_API_KEY). " +
				"Set ALLOW_CLIENT_API_KEYS=true to require clients to supply their own keys.",
		)
	}

	// Redis URL is required when cache mode is "redis".
	if c.Cache.Mode == "redis" && c.Redis.URL == "" {
		return fmt.Errorf(
			"config: REDIS_URL is required when CACHE_MODE=redis; " +
				"set CACHE_MODE=memory to use the built-in in-process cache",
		)
	}

	// Validate cache mode value.
	switch c.Cache.Mode {
	case "redis", "memory", "none":
	default:
		return fmt.Errorf(
			"config: invalid CACHE_MODE %q; must be one of: redis, memory, none",
			c.Cache.Mode,
		)
	}

	// Validate log level.
	switch c.LogLevel {
	case "debug", "info", "warn", "error":
	default:
		return fmt.Errorf(
			"config: invalid LOG_LEVEL %q; must be one of: debug, info, warn, error",
			c.LogLevel,
		)
	}

	// Circuit breaker sanity checks.
	if c.CircuitBreaker.ErrorThreshold < 1 {
		return fmt.Errorf("config: CB_ERROR_THRESHOLD must be ≥ 1, got %d", c.CircuitBreaker.ErrorThreshold)
	}
	if c.CircuitBreaker.TimeWindow <= 0 {
		return fmt.Errorf("config: CB_TIME_WINDOW must be a positive duration")
	}
	if c.Failover.MaxRetries < 1 {
		return fmt.Errorf("config: MAX_RETRIES must be ≥ 1, got %d", c.Failover.MaxRetries)
	}

	return nil
}

// AtLeastOneProviderKey returns true if at least one provider is configured.
func (c *Config) AtLeastOneProviderKey() bool {
	return c.OpenAI.APIKey != "" ||
		c.Anthropic.APIKey != "" ||
		c.Gemini.APIKey != "" ||
		c.Mistral.APIKey != "" ||
		c.XAI.APIKey != "" ||
		c.DeepSeek.APIKey != "" ||
		c.Groq.APIKey != "" ||
		c.Together.APIKey != "" ||
		c.Perplexity.APIKey != "" ||
		c.Cerebras.APIKey != "" ||
		c.Moonshot.APIKey != "" ||
		c.MiniMax.APIKey != "" ||
		c.Qwen.APIKey != "" ||
		c.Nebius.APIKey != "" ||
		c.NovitaAI.APIKey != "" ||
		c.ByteDance.APIKey != "" ||
		c.ZAI.APIKey != "" ||
		c.CanopyWave.APIKey != "" ||
		c.Inference.APIKey != "" ||
		c.NanoGPT.APIKey != "" ||
		c.VertexAI.Project != "" ||
		c.Bedrock.AccessKey != "" ||
		c.Azure.APIKey != ""
}

// loadDotEnv populates process env vars from a .env file when present.
func loadDotEnv(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("config: failed to stat %s: %w", path, err)
	}
	if info.IsDir() {
		return fmt.Errorf("config: %s is a directory, expected a file", path)
	}
	if err := gotenv.Load(path); err != nil {
		return fmt.Errorf("config: failed to load %s: %w", path, err)
	}
	return nil
}
