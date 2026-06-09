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
	"net"
	"net/url"
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
	//
	// Security note: when enabled, all configured upstream endpoints must remain
	// pinned to explicitly allowed public provider hosts. Arbitrary/custom/local
	// BaseURL overrides are rejected to prevent SSRF with bring-your-own-key mode.
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
		XAI:        ProviderConfig{APIKey: v.GetString("XAI_API_KEY"), BaseURL: v.GetString("XAI_BASE_URL")},
		DeepSeek:   ProviderConfig{APIKey: v.GetString("DEEPSEEK_API_KEY"), BaseURL: v.GetString("DEEPSEEK_BASE_URL")},
		Groq:       ProviderConfig{APIKey: v.GetString("GROQ_API_KEY"), BaseURL: v.GetString("GROQ_BASE_URL")},
		Together:   ProviderConfig{APIKey: v.GetString("TOGETHER_API_KEY"), BaseURL: v.GetString("TOGETHER_BASE_URL")},
		Perplexity: ProviderConfig{APIKey: v.GetString("PERPLEXITY_API_KEY"), BaseURL: v.GetString("PERPLEXITY_BASE_URL")},
		Cerebras:   ProviderConfig{APIKey: v.GetString("CEREBRAS_API_KEY"), BaseURL: v.GetString("CEREBRAS_BASE_URL")},
		Moonshot:   ProviderConfig{APIKey: v.GetString("MOONSHOT_API_KEY"), BaseURL: v.GetString("MOONSHOT_BASE_URL")},
		MiniMax:    ProviderConfig{APIKey: v.GetString("MINIMAX_API_KEY"), BaseURL: v.GetString("MINIMAX_BASE_URL")},
		Qwen:       ProviderConfig{APIKey: v.GetString("QWEN_API_KEY"), BaseURL: v.GetString("QWEN_BASE_URL")},
		Nebius:     ProviderConfig{APIKey: v.GetString("NEBIUS_API_KEY"), BaseURL: v.GetString("NEBIUS_BASE_URL")},
		NovitaAI:   ProviderConfig{APIKey: v.GetString("NOVITA_API_KEY"), BaseURL: v.GetString("NOVITA_BASE_URL")},
		ByteDance:  ProviderConfig{APIKey: v.GetString("BYTEDANCE_API_KEY"), BaseURL: v.GetString("BYTEDANCE_BASE_URL")},
		ZAI:        ProviderConfig{APIKey: v.GetString("ZAI_API_KEY"), BaseURL: v.GetString("ZAI_BASE_URL")},
		CanopyWave: ProviderConfig{APIKey: v.GetString("CANOPYWAVE_API_KEY"), BaseURL: v.GetString("CANOPYWAVE_BASE_URL")},
		Inference:  ProviderConfig{APIKey: v.GetString("INFERENCE_API_KEY"), BaseURL: v.GetString("INFERENCE_BASE_URL")},
		NanoGPT:    ProviderConfig{APIKey: v.GetString("NANOGPT_API_KEY"), BaseURL: v.GetString("NANOGPT_BASE_URL")},

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
			"config: at least one provider API key is required "+
				"(OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, MISTRAL_API_KEY, "+
				"XAI_API_KEY, DEEPSEEK_API_KEY, GROQ_API_KEY, TOGETHER_API_KEY, "+
				"PERPLEXITY_API_KEY, CEREBRAS_API_KEY, MOONSHOT_API_KEY, MINIMAX_API_KEY, "+
				"QWEN_API_KEY, NEBIUS_API_KEY, NOVITA_API_KEY, BYTEDANCE_API_KEY, "+
				"ZAI_API_KEY, CANOPYWAVE_API_KEY, INFERENCE_API_KEY, NANOGPT_API_KEY, "+
				"or enable ALLOW_CLIENT_API_KEYS=true)",
		)
	}

	switch c.LogLevel {
	case "debug", "info", "warn", "error":
	default:
		return fmt.Errorf("config: invalid LOG_LEVEL %q (must be debug, info, warn, error)", c.LogLevel)
	}

	switch c.Cache.Mode {
	case "memory", "redis", "none":
	default:
		return fmt.Errorf("config: invalid CACHE_MODE %q (must be memory, redis, none)", c.Cache.Mode)
	}

	if c.Cache.Mode == "redis" && strings.TrimSpace(c.Redis.URL) == "" {
		return errors.New("config: REDIS_URL is required when CACHE_MODE=redis")
	}

	if c.Cache.TTL < 0 {
		return errors.New("config: CACHE_TTL must be >= 0")
	}

	if c.CircuitBreaker.ErrorThreshold < 1 {
		return errors.New("config: CB_ERROR_THRESHOLD must be >= 1")
	}
	if c.CircuitBreaker.TimeWindow <= 0 {
		return errors.New("config: CB_TIME_WINDOW must be > 0")
	}
	if c.CircuitBreaker.HalfOpenTimeout <= 0 {
		return errors.New("config: CB_HALF_OPEN_TIMEOUT must be > 0")
	}

	if c.RateLimit.RPMLimit < 0 {
		return errors.New("config: RPM_LIMIT must be >= 0")
	}

	if c.Failover.MaxRetries < 1 {
		return errors.New("config: MAX_RETRIES must be >= 1")
	}
	if c.Failover.ProviderTimeout <= 0 {
		return errors.New("config: PROVIDER_TIMEOUT must be > 0")
	}

	if strings.TrimSpace(c.AppBaseURL) != "" {
		if err := validatePublicHTTPURL("APP_BASE_URL", c.AppBaseURL, false, nil); err != nil {
			return err
		}
	}

	if strings.TrimSpace(c.Azure.Endpoint) != "" {
		if err := validatePublicHTTPURL("AZURE_OPENAI_ENDPOINT", c.Azure.Endpoint, false, []string{
			"openai.azure.com",
			"azure.com",
		}); err != nil {
			return err
		}
	}

	if strings.TrimSpace(c.Bedrock.EndpointURL) != "" {
		if err := validatePublicHTTPURL("BEDROCK_ENDPOINT_URL", c.Bedrock.EndpointURL, !c.AllowClientAPIKeys, []string{
			"amazonaws.com",
			"amazonaws.com.cn",
		}); err != nil {
			return err
		}
	}

	if err := c.validateProviderBaseURLs(); err != nil {
		return err
	}

	if c.AllowClientAPIKeys {
		if err := c.validateBYOKUpstreams(); err != nil {
			return err
		}
	}

	return nil
}

// AtLeastOneProviderKey reports whether any provider auth is configured.
func (c *Config) AtLeastOneProviderKey() bool {
	return hasValue(
		c.OpenAI.APIKey,
		c.Anthropic.APIKey,
		c.Gemini.APIKey,
		c.Mistral.APIKey,
		c.XAI.APIKey,
		c.DeepSeek.APIKey,
		c.Groq.APIKey,
		c.Together.APIKey,
		c.Perplexity.APIKey,
		c.Cerebras.APIKey,
		c.Moonshot.APIKey,
		c.MiniMax.APIKey,
		c.Qwen.APIKey,
		c.Nebius.APIKey,
		c.NovitaAI.APIKey,
		c.ByteDance.APIKey,
		c.ZAI.APIKey,
		c.CanopyWave.APIKey,
		c.Inference.APIKey,
		c.NanoGPT.APIKey,
		c.Azure.APIKey,
		c.VertexAI.Project,
		c.Bedrock.AccessKey,
	)
}

func hasValue(values ...string) bool {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return true
		}
	}
	return false
}

func (c *Config) validateProviderBaseURLs() error {
	providers := []struct {
		name          string
		baseURL       string
		allowOverride bool
		allowedHosts  []string
	}{
		{"OPENAI_BASE_URL", c.OpenAI.BaseURL, !c.AllowClientAPIKeys, []string{"openai.com"}},
		{"ANTHROPIC_BASE_URL", c.Anthropic.BaseURL, !c.AllowClientAPIKeys, []string{"anthropic.com"}},
		{"GEMINI_BASE_URL", c.Gemini.BaseURL, !c.AllowClientAPIKeys, []string{"googleapis.com", "google.com"}},
		{"MISTRAL_BASE_URL", c.Mistral.BaseURL, !c.AllowClientAPIKeys, []string{"mistral.ai"}},
		{"XAI_BASE_URL", c.XAI.BaseURL, !c.AllowClientAPIKeys, []string{"x.ai"}},
		{"DEEPSEEK_BASE_URL", c.DeepSeek.BaseURL, !c.AllowClientAPIKeys, []string{"deepseek.com"}},
		{"GROQ_BASE_URL", c.Groq.BaseURL, !c.AllowClientAPIKeys, []string{"groq.com"}},
		{"TOGETHER_BASE_URL", c.Together.BaseURL, !c.AllowClientAPIKeys, []string{"together.xyz"}},
		{"PERPLEXITY_BASE_URL", c.Perplexity.BaseURL, !c.AllowClientAPIKeys, []string{"perplexity.ai"}},
		{"CEREBRAS_BASE_URL", c.Cerebras.BaseURL, !c.AllowClientAPIKeys, []string{"cerebras.ai"}},
		{"MOONSHOT_BASE_URL", c.Moonshot.BaseURL, !c.AllowClientAPIKeys, []string{"moonshot.cn", "kimi.com"}},
		{"MINIMAX_BASE_URL", c.MiniMax.BaseURL, !c.AllowClientAPIKeys, []string{"minimax.io"}},
		{"QWEN_BASE_URL", c.Qwen.BaseURL, !c.AllowClientAPIKeys, []string{"aliyuncs.com", "dashscope.com"}},
		{"NEBIUS_BASE_URL", c.Nebius.BaseURL, !c.AllowClientAPIKeys, []string{"nebius.com", "nebius.ai"}},
		{"NOVITA_BASE_URL", c.NovitaAI.BaseURL, !c.AllowClientAPIKeys, []string{"novita.ai"}},
		{"BYTEDANCE_BASE_URL", c.ByteDance.BaseURL, !c.AllowClientAPIKeys, []string{"volces.com", "volcengine.com"}},
		{"ZAI_BASE_URL", c.ZAI.BaseURL, !c.AllowClientAPIKeys, []string{"z.ai", "bigmodel.cn"}},
		{"CANOPYWAVE_BASE_URL", c.CanopyWave.BaseURL, !c.AllowClientAPIKeys, []string{"canopywave.com"}},
		{"INFERENCE_BASE_URL", c.Inference.BaseURL, !c.AllowClientAPIKeys, []string{"inference.net"}},
		{"NANOGPT_BASE_URL", c.NanoGPT.BaseURL, !c.AllowClientAPIKeys, []string{"nanogpt.com"}},
	}

	for _, p := range providers {
		if strings.TrimSpace(p.baseURL) == "" {
			continue
		}
		if err := validatePublicHTTPURL(p.name, p.baseURL, p.allowOverride, p.allowedHosts); err != nil {
			return err
		}
	}

	return nil
}

func (c *Config) validateBYOKUpstreams() error {
	if strings.TrimSpace(c.OpenAI.BaseURL) != "" {
		if err := validatePublicHTTPURL("OPENAI_BASE_URL", c.OpenAI.BaseURL, false, []string{"openai.com"}); err != nil {
			return err
		}
	}
	if strings.TrimSpace(c.Anthropic.BaseURL) != "" {
		if err := validatePublicHTTPURL("ANTHROPIC_BASE_URL", c.Anthropic.BaseURL, false, []string{"anthropic.com"}); err != nil {
			return err
		}
	}
	if strings.TrimSpace(c.Gemini.BaseURL) != "" {
		if err := validatePublicHTTPURL("GEMINI_BASE_URL", c.Gemini.BaseURL, false, []string{"googleapis.com", "google.com"}); err != nil {
			return err
		}
	}
	if strings.TrimSpace(c.Mistral.BaseURL) != "" {
		if err := validatePublicHTTPURL("MISTRAL_BASE_URL", c.Mistral.BaseURL, false, []string{"mistral.ai"}); err != nil {
			return err
		}
	}
	return nil
}

func validatePublicHTTPURL(fieldName, raw string, allowAnyPublicHost bool, allowedHostSuffixes []string) error {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return fmt.Errorf("config: invalid %s URL: %w", fieldName, err)
	}

	if u.Scheme != "https" && u.Scheme != "http" {
		return fmt.Errorf("config: %s must use http or https", fieldName)
	}
	if u.Host == "" {
		return fmt.Errorf("config: %s must include a host", fieldName)
	}
	if u.User != nil {
		return fmt.Errorf("config: %s must not include userinfo", fieldName)
	}

	host := u.Hostname()
	if host == "" {
		return fmt.Errorf("config: %s must include a valid hostname", fieldName)
	}

	if isLocalOrPrivateHost(host) {
		return fmt.Errorf("config: %s must not point to localhost, private, loopback, link-local, multicast, or unspecified addresses", fieldName)
	}

	if !allowAnyPublicHost && len(allowedHostSuffixes) > 0 && !hostMatchesAllowedSuffixes(host, allowedHostSuffixes) {
		return fmt.Errorf("config: %s host %q is not in the allowed upstream list", fieldName, host)
	}

	return nil
}

func hostMatchesAllowedSuffixes(host string, suffixes []string) bool {
	h := strings.ToLower(strings.TrimSuffix(host, "."))
	for _, s := range suffixes {
		s = strings.ToLower(strings.TrimSpace(strings.TrimPrefix(s, ".")))
		if s == "" {
			continue
		}
		if h == s || strings.HasSuffix(h, "."+s) {
			return true
		}
	}
	return false
}

func isLocalOrPrivateHost(host string) bool {
	h := strings.ToLower(strings.TrimSuffix(strings.TrimSpace(host), "."))
	if h == "" {
		return true
	}

	switch h {
	case "localhost", "localhost.localdomain":
		return true
	}

	if ip := net.ParseIP(h); ip != nil {
		return isDisallowedIP(ip)
	}

	return false
}

func isDisallowedIP(ip net.IP) bool {
	if ip == nil {
		return true
	}

	if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() ||
		ip.IsMulticast() || ip.IsUnspecified() {
		return true
	}

	if v4 := ip.To4(); v4 != nil {
		// 0.0.0.0/8
		if v4[0] == 0 {
			return true
		}
		// 100.64.0.0/10 (CGNAT)
		if v4[0] == 100 && v4[1] >= 64 && v4[1] <= 127 {
			return true
		}
		// 169.254.0.0/16
		if v4[0] == 169 && v4[1] == 254 {
			return true
		}
		// 198.18.0.0/15 (benchmarking)
		if v4[0] == 198 && (v4[1] == 18 || v4[1] == 19) {
			return true
		}
		// 224.0.0.0/4 multicast and 240.0.0.0/4 reserved
		if v4[0] >= 224 {
			return true
		}
		return false
	}

	// IPv6 unique-local fc00::/7
	if len(ip) == net.IPv6len {
		if ip[0]&0xfe == 0xfc {
			return true
		}
		// fe80::/10 link-local
		if ip[0] == 0xfe && (ip[1]&0xc0) == 0x80 {
			return true
		}
		// ff00::/8 multicast
		if ip[0] == 0xff {
			return true
		}
	}

	return false
}

func loadDotEnv(path string) error {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("config: stat %s: %w", path, err)
	}
	if err := gotenv.Load(path); err != nil {
		return fmt.Errorf("config: load %s: %w", path, err)
	}
	return nil
}