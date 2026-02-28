// Package providers defines the common interfaces and types used by all LLM
// provider implementations (OpenAI, Anthropic, Gemini, Mistral, and others).
//
// Each provider lives in its own sub-package and implements the Provider
// interface. Providers that support vector embeddings additionally implement
// EmbeddingProvider.
package providers

import (
	"context"
	"time"
)

type (
	// StreamChunk is a single token chunk delivered during a streaming response.
	StreamChunk struct {
		Content      string
		FinishReason string
	}

	// Message is a single turn in a conversation (role + text content).
	Message struct {
		Role    string
		Content string
	}

	// Usage — token usage stats.
	Usage struct {
		InputTokens  int
		OutputTokens int
	}

	// ProxyRequest — normalized client request.
	ProxyRequest struct {
		Model       string
		Messages    []Message
		Stream      bool
		Temperature float64
		MaxTokens   int
		WorkspaceID string
		APIKey      string
		APIKeyID    string
		RequestID   string
	}

	// ProxyResponse — normalized provider response.
	ProxyResponse struct {
		ID      string
		Model   string
		Content string
		Usage   Usage
		Stream  <-chan StreamChunk // nil if it's not a stream.
	}

	// EmbeddingRequest — normalized embedding request.
	EmbeddingRequest struct {
		// Input is the list of texts to embed. Always at least one element.
		Input []string
		// Model is the provider-native model name (e.g. "text-embedding-3-small").
		Model       string
		WorkspaceID string
		APIKey      string
		APIKeyID    string
		RequestID   string
	}

	// EmbeddingData — a single embedding vector.
	EmbeddingData struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	}

	// EmbeddingResponse — normalized embedding response.
	EmbeddingResponse struct {
		Model string
		Data  []EmbeddingData
		Usage Usage
	}
)

// Provider — LLM provider interface.
type Provider interface {
	Name() string
	Request(ctx context.Context, req *ProxyRequest) (*ProxyResponse, error)
	HealthCheck(ctx context.Context) error
}

// EmbeddingProvider is an optional interface implemented by providers that
// support the embeddings API. Check with a type assertion before calling.
type EmbeddingProvider interface {
	Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)
}

// EmbeddingModelAliases maps embedding model names to provider names.
// Used by the proxy to route POST /v1/embeddings requests.
var EmbeddingModelAliases = map[string]string{
	// OpenAI
	"text-embedding-3-small": "openai",
	"text-embedding-3-large": "openai",
	"text-embedding-ada-002": "openai",
	// Mistral
	"mistral-embed": "mistral",
	// Google Gemini
	"text-embedding-004": "gemini",
	"embedding-001":      "gemini",
}

// ModelAliases maps model names to provider names.
// Used by the proxy to route POST /v1/chat/completions requests.
var ModelAliases = map[string]string{

	// ─── OpenAI ───────────────────────────────────────────────────────────────
	"gpt-4":                  "openai",
	"gpt-4-0613":             "openai",
	"gpt-4o":                 "openai",
	"gpt-4o-2024-11-20":      "openai",
	"gpt-4o-2024-08-06":      "openai",
	"gpt-4o-2024-05-13":      "openai",
	"gpt-4o-mini":            "openai",
	"gpt-4o-mini-2024-07-18": "openai",
	"gpt-4-turbo":            "openai",
	"gpt-4-turbo-2024-04-09": "openai",
	"gpt-4-turbo-preview":    "openai",
	"gpt-3.5-turbo":          "openai",
	"gpt-3.5-turbo-0125":     "openai",
	"gpt-3.5-turbo-1106":     "openai",
	"o1":                     "openai",
	"o1-mini":                "openai",
	"o1-preview":             "openai",
	"o1-2024-12-17":          "openai",
	"o3":                     "openai",
	"o3-mini":                "openai",
	"o3-mini-2025-01-31":     "openai",
	"o4-mini":                "openai",
	"gpt-4.1":                "openai",
	"gpt-4.1-mini":           "openai",
	"gpt-4.1-nano":           "openai",

	// ─── Anthropic ────────────────────────────────────────────────────────────
	"claude-3-5-sonnet":          "anthropic",
	"claude-3-5-sonnet-20241022": "anthropic",
	"claude-3-5-haiku":           "anthropic",
	"claude-3-5-haiku-20241022":  "anthropic",
	"claude-3-opus":              "anthropic",
	"claude-3-opus-20240229":     "anthropic",
	"claude-3-haiku":             "anthropic",
	"claude-3-haiku-20240307":    "anthropic",
	"claude-3-sonnet-20240229":   "anthropic",
	"claude-3-7-sonnet-20250219": "anthropic",
	"claude-3-7-sonnet":          "anthropic",
	"claude-opus-4":              "anthropic",
	"claude-sonnet-4":            "anthropic",
	"claude-haiku-4":             "anthropic",
	"claude-opus-4-5":            "anthropic",
	"claude-sonnet-4-5":          "anthropic",
	"claude-haiku-4-5":           "anthropic",
	"claude-opus-4-6":            "anthropic",
	"claude-sonnet-4-6":          "anthropic",
	"claude-haiku-4-6":           "anthropic",

	// ─── Google AI Studio ─────────────────────────────────────────────────────
	"gemini-pro":                    "gemini",
	"gemini-1.0-pro":                "gemini",
	"gemini-1.5-pro":                "gemini",
	"gemini-1.5-pro-002":            "gemini",
	"gemini-1.5-flash":              "gemini",
	"gemini-1.5-flash-002":          "gemini",
	"gemini-1.5-flash-8b":           "gemini",
	"gemini-2.0-flash":              "gemini",
	"gemini-2.0-flash-lite":         "gemini",
	"gemini-2.0-flash-exp":          "gemini",
	"gemini-2.0-pro-exp":            "gemini",
	"gemini-2.5-pro":                "gemini",
	"gemini-2.5-flash":              "gemini",
	"gemini-exp-1206":               "gemini",
	"gemini-2.0-flash-thinking-exp": "gemini",
	"gemma-3-27b-it":                "gemini",
	"gemma-3-12b-it":                "gemini",
	"gemma-3-4b-it":                 "gemini",
	"gemma-2-27b-it":                "gemini",
	"gemma-2-9b-it":                 "gemini",
	"gemma-2-2b-it":                 "gemini",
	"learnlm-1.5-pro-experimental":  "gemini",

	// ─── Mistral AI ───────────────────────────────────────────────────────────
	"mistral-large-latest":  "mistral",
	"mistral-small-latest":  "mistral",
	"mistral-large":         "mistral",
	"mistral-large-2411":    "mistral",
	"mistral-medium":        "mistral",
	"mistral-small-2501":    "mistral",
	"mistral-small-2412":    "mistral",
	"mistral-nemo":          "mistral",
	"open-mistral-nemo":     "mistral",
	"mixtral-8x7b":          "mistral",
	"open-mixtral-8x22b":    "mistral",
	"pixtral-large-2411":    "mistral",
	"pixtral-12b-2409":      "mistral",
	"codestral-2501":        "mistral",
	"codestral-latest":      "mistral",
	"ministral-3b-latest":   "mistral",
	"ministral-8b-latest":   "mistral",

	// ─── xAI (Grok) ───────────────────────────────────────────────────────────
	"grok-3":             "xai",
	"grok-3-fast":        "xai",
	"grok-3-mini":        "xai",
	"grok-3-mini-fast":   "xai",
	"grok-3-latest":      "xai",
	"grok-2":             "xai",
	"grok-2-mini":        "xai",
	"grok-2-1212":        "xai",
	"grok-2-vision":      "xai",
	"grok-2-vision-1212": "xai",
	"grok-2-image-1212":  "xai",
	"grok-beta":          "xai",
	"grok-vision-beta":   "xai",

	// ─── DeepSeek ─────────────────────────────────────────────────────────────
	"deepseek-chat":     "deepseek",
	"deepseek-reasoner": "deepseek",

	// ─── Groq ─────────────────────────────────────────────────────────────────
	// Groq uses its own model naming distinct from HuggingFace IDs.
	"llama-3.3-70b-versatile": "groq",
	"llama-3.1-70b-versatile": "groq",
	"llama-3.1-8b-instant":    "groq",
	"llama3-70b-8192":         "groq",
	"llama3-8b-8192":          "groq",
	"gemma2-9b-it":            "groq",

	// ─── Together AI ──────────────────────────────────────────────────────────
	// Uses HuggingFace-style names with provider/model format.
	"meta-llama/Llama-3.3-70B-Instruct-Turbo":       "together",
	"meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "together",
	"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo":  "together",
	"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo":   "together",
	"mistralai/Mixtral-8x7B-Instruct-v0.1":          "together",
	"mistralai/Mixtral-8x22B-Instruct-v0.1":         "together",
	"Qwen/Qwen2.5-72B-Instruct-Turbo":               "together",
	"deepseek-ai/DeepSeek-R1":                       "together",
	"google/gemma-2-27b-it":                         "together",

	// ─── Cerebras ─────────────────────────────────────────────────────────────
	// Cerebras uses short model names (note: llama3.1 not llama-3.1).
	"llama3.1-8b":                   "cerebras",
	"llama3.1-70b":                  "cerebras",
	"llama3.3-70b":                  "cerebras",
	"qwen-3-32b":                    "cerebras",
	"deepseek-r1-distill-llama-70b": "cerebras",
	"qwen-3-235b":                   "cerebras",
	"llama4-scout-17b-16e":          "cerebras",

	// ─── Moonshot AI ──────────────────────────────────────────────────────────
	"moonshot-v1-8k":   "moonshot",
	"moonshot-v1-32k":  "moonshot",
	"moonshot-v1-128k": "moonshot",
	"moonshot-v1-auto": "moonshot",
	"kimi-latest":      "moonshot",

	// ─── MiniMax ──────────────────────────────────────────────────────────────
	"MiniMax-Text-01": "minimax",
	"MiniMax-VL-01":   "minimax",
	"abab6.5s-chat":   "minimax",
	"abab6.5-chat":    "minimax",
	"abab5.5-chat":    "minimax",

	// ─── Perplexity ───────────────────────────────────────────────────────────
	"sonar":           "perplexity",
	"sonar-pro":       "perplexity",
	"sonar-reasoning": "perplexity",

	// ─── Alibaba Cloud (Qwen) ─────────────────────────────────────────────────
	"qwen-turbo":           "qwen",
	"qwen-plus":            "qwen",
	"qwen-max":             "qwen",
	"qwen-max-2025-01-25":  "qwen",
	"qwen-long":            "qwen",
	"qwen-vl-plus":         "qwen",
	"qwen-vl-max":          "qwen",
	"qwq-plus":             "qwen",
	"qwq-32b":              "qwen",
	"qwen2.5-72b-instruct": "qwen",
	"qwen2.5-32b-instruct": "qwen",
	"qwen2.5-7b-instruct":  "qwen",

	// ─── Nebius AI Studio ─────────────────────────────────────────────────────
	// Uses HuggingFace IDs — note different variant names from Together AI.
	"meta-llama/Meta-Llama-3.1-70B-Instruct": "nebius",
	"meta-llama/Meta-Llama-3.1-8B-Instruct":  "nebius",
	"meta-llama/Meta-Llama-3.3-70B-Instruct": "nebius",
	"Qwen/Qwen2.5-72B-Instruct":              "nebius",
	"mistralai/Mistral-7B-Instruct-v0.3":     "nebius",
	"mistralai/Mistral-Nemo-Instruct-2407":   "nebius",
	"deepseek-ai/DeepSeek-V3":                "nebius",
	"deepseek-ai/DeepSeek-R1-Nebius":         "nebius",

	// ─── NovitaAI ─────────────────────────────────────────────────────────────
	// Uses lowercase HuggingFace IDs.
	"meta-llama/llama-3.1-8b-instruct":   "novita",
	"meta-llama/llama-3.1-70b-instruct":  "novita",
	"meta-llama/llama-3.1-405b-instruct": "novita",
	"meta-llama/llama-3.3-70b-instruct":  "novita",
	"deepseek/deepseek-v3":               "novita",
	"deepseek/deepseek-r1":               "novita",
	"mistralai/mistral-7b-instruct-v0.3": "novita",
	"qwen/qwen2.5-72b-instruct":          "novita",

	// ─── ByteDance ModelArk ───────────────────────────────────────────────────
	"doubao-1.5-pro-32k":  "bytedance",
	"doubao-1.5-lite-32k": "bytedance",
	"doubao-pro-32k":      "bytedance",
	"doubao-lite-32k":     "bytedance",
	"doubao-pro-4k":       "bytedance",
	"doubao-pro-128k":     "bytedance",

	// ─── Z AI ─────────────────────────────────────────────────────────────────
	"glm-4-plus":  "zai",
	"glm-4-air":   "zai",
	"glm-4-flash": "zai",
	"glm-4-0520":  "zai",
	"glm-4":       "zai",
	"glm-3-turbo": "zai",

	// ─── CanopyWave ───────────────────────────────────────────────────────────
	// OpenAI-compatible infrastructure provider; model names match OpenAI format.
	// Routes to CanopyWave when explicitly configured as primary provider.

	// ─── Inference.net ────────────────────────────────────────────────────────
	"inference-llama-3.1-8b":  "inference",
	"inference-llama-3.1-70b": "inference",

	// ─── NanoGPT ──────────────────────────────────────────────────────────────
	// NanoGPT aggregates many models; use the nanogpt- prefix for routing.
	"nanogpt-gpt-4o":   "nanogpt",
	"nanogpt-claude-3": "nanogpt",

	// ─── AWS Bedrock ──────────────────────────────────────────────────────────
	// Bedrock uses provider-namespaced model IDs.
	"anthropic.claude-3-5-sonnet-20241022-v2:0": "bedrock",
	"anthropic.claude-3-opus-20240229-v1:0":     "bedrock",
	"anthropic.claude-3-haiku-20240307-v1:0":    "bedrock",
	"anthropic.claude-3-sonnet-20240229-v1:0":   "bedrock",
	"meta.llama3-70b-instruct-v1:0":             "bedrock",
	"meta.llama3-8b-instruct-v1:0":              "bedrock",
	"meta.llama3-1-70b-instruct-v1:0":           "bedrock",
	"amazon.titan-text-express-v1":              "bedrock",
	"amazon.titan-text-lite-v1":                 "bedrock",
	"amazon.nova-pro-v1:0":                      "bedrock",
	"amazon.nova-lite-v1:0":                     "bedrock",
	"amazon.nova-micro-v1:0":                    "bedrock",
	"mistral.mistral-large-2402-v1:0":           "bedrock",
	"ai21.jamba-1-5-large-v1:0":                 "bedrock",

	// ─── Azure OpenAI ─────────────────────────────────────────────────────────
	// Use the "azure-" prefix to route explicitly to Azure. The prefix is
	// stripped to derive the Azure deployment name.
	"azure-gpt-4":        "azure",
	"azure-gpt-4o":       "azure",
	"azure-gpt-4-turbo":  "azure",
	"azure-gpt-4o-mini":  "azure",
	"azure-o1":           "azure",
	"azure-o3-mini":      "azure",
	"azure-gpt-4.1":      "azure",
	"azure-gpt-4.1-mini": "azure",

	// ─── Google Vertex AI ─────────────────────────────────────────────────────
	// Use the "vertexai-" prefix to route explicitly to Vertex AI.
	// Without the prefix, Gemini models default to Google AI Studio.
	"vertexai-gemini-2.0-flash":      "vertexai",
	"vertexai-gemini-2.0-flash-lite": "vertexai",
	"vertexai-gemini-1.5-pro":        "vertexai",
	"vertexai-gemini-1.5-flash":      "vertexai",
	"vertexai-gemini-2.5-pro":        "vertexai",
	"vertexai-gemini-2.5-flash":      "vertexai",
}

// DefaultFallbackOrder is the default provider failover sequence.
// When the primary provider fails, the gateway tries each provider in this
// order until one succeeds or MaxRetries is exhausted.
var DefaultFallbackOrder = []string{
	"openai",
	"anthropic",
	"gemini",
	"mistral",
	"xai",
	"groq",
	"azure",
	"vertexai",
	"bedrock",
}

// Default circuit breaker and failover constants.
const (
	CBErrorThreshold  = 5
	CBTimeWindow      = 60 * time.Second
	CBHalfOpenTimeout = 30 * time.Second
	MaxRetries        = 3
	ProviderTimeout   = 30 * time.Second
)

type StatusCoder interface {
	HTTPStatus() int
}
