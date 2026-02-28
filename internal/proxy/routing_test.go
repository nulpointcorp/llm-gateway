package proxy

import (
	"testing"
)

func TestResolveProvider_KnownModels(t *testing.T) {
	tests := []struct {
		model    string
		expected string
	}{
		// OpenAI
		{"gpt-4", "openai"},
		{"gpt-4o", "openai"},
		{"gpt-4-turbo", "openai"},
		{"gpt-3.5-turbo", "openai"},
		// Anthropic
		{"claude-3-5-sonnet", "anthropic"},
		{"claude-3-opus", "anthropic"},
		{"claude-3-haiku", "anthropic"},
		// Google
		{"gemini-pro", "gemini"},
		{"gemini-1.5-pro", "gemini"},
		{"gemini-1.5-flash", "gemini"},
		// Mistral
		{"mistral-large", "mistral"},
		{"mistral-medium", "mistral"},
		{"mixtral-8x7b", "mistral"},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			got := resolveProvider(tt.model)
			if got != tt.expected {
				t.Errorf("resolveProvider(%q) = %q, want %q", tt.model, got, tt.expected)
			}
		})
	}
}

func TestResolveProvider_UnknownModel_DefaultsToOpenAI(t *testing.T) {
	got := resolveProvider("some-unknown-model")
	if got != "openai" {
		t.Errorf("resolveProvider(unknown) = %q, want 'openai'", got)
	}
}

func TestResolveProvider_EmptyString(t *testing.T) {
	got := resolveProvider("")
	if got != "openai" {
		t.Errorf("resolveProvider('') = %q, want 'openai'", got)
	}
}
