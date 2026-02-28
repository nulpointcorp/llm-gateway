package proxy

import (
	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

// resolveProvider returns the provider name for the given chat/completion model.
// Falls back to "openai" if the model is unknown.
func resolveProvider(model string) string {
	if name, ok := providers.ModelAliases[model]; ok {
		return name
	}
	return "openai"
}

// resolveEmbeddingProvider returns the provider name for the given embedding model.
// It checks EmbeddingModelAliases first, then ModelAliases for provider detection,
// and falls back to "openai".
func resolveEmbeddingProvider(model string) string {
	if name, ok := providers.EmbeddingModelAliases[model]; ok {
		return name
	}
	// A user might pass a chat model name; resolve to its provider so it can
	// attempt the embedding call (the provider API will return a clear error).
	if name, ok := providers.ModelAliases[model]; ok {
		return name
	}
	return "openai"
}
