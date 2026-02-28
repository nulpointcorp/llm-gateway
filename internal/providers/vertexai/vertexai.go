// Package vertexai implements the providers.Provider interface for Google Vertex AI.
// It uses the same google.golang.org/genai SDK as the Gemini provider but
// connects to Vertex AI using Application Default Credentials instead of an API key.
//
// Required configuration:
//   - VERTEX_PROJECT  — Google Cloud project ID
//   - VERTEX_LOCATION — region, e.g. "us-central1" (default)
//
// Authentication is handled via ADC:
//   - GOOGLE_APPLICATION_CREDENTIALS pointing to a service account key file, or
//   - Workload Identity / GCE metadata server when running on GCP.
package vertexai

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"strings"

	"google.golang.org/genai"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

const (
	defaultLocation = "us-central1"
	providerName    = "vertexai"
)

// Provider implements providers.Provider for Google Vertex AI.
type Provider struct {
	project  string
	location string
	client   *genai.Client
}

// Option configures a Provider.
type Option func(*Provider)

// WithLocation overrides the default Vertex AI region.
func WithLocation(loc string) Option {
	return func(p *Provider) { p.location = loc }
}

// New creates a new Vertex AI Provider.
// Auth is resolved via Application Default Credentials — no API key needed.
func New(ctx context.Context, project string, opts ...Option) (*Provider, error) {
	p := &Provider{
		project:  project,
		location: defaultLocation,
	}
	for _, o := range opts {
		o(p)
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Project:  p.project,
		Location: p.location,
		Backend:  genai.BackendVertexAI,
	})
	if err != nil {
		return nil, fmt.Errorf("vertexai: create client: %w", err)
	}

	p.client = client
	return p, nil
}

func (p *Provider) Name() string { return providerName }

func (p *Provider) HealthCheck(ctx context.Context) error {
	_, err := p.client.Models.List(ctx, &genai.ListModelsConfig{PageSize: 1})
	if err != nil {
		return fmt.Errorf("vertexai: health check: %w", toProviderError(err))
	}
	return nil
}

func (p *Provider) Request(ctx context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	contents, cfg := buildContentsAndConfig(req)

	if req.Stream {
		return p.handleStreaming(ctx, req.Model, contents, cfg)
	}
	return p.handleResponse(ctx, req, contents, cfg)
}

func buildContentsAndConfig(req *providers.ProxyRequest) ([]*genai.Content, *genai.GenerateContentConfig) {
	var systemPrompt string
	contents := make([]*genai.Content, 0, len(req.Messages))

	for _, m := range req.Messages {
		switch strings.ToLower(m.Role) {
		case "system", "developer":
			if systemPrompt != "" {
				systemPrompt += "\n"
			}
			systemPrompt += m.Content
		case "assistant", "model":
			contents = append(contents, genai.NewContentFromText(m.Content, genai.RoleModel))
		default:
			contents = append(contents, genai.NewContentFromText(m.Content, genai.RoleUser))
		}
	}

	var cfg *genai.GenerateContentConfig
	if systemPrompt != "" || req.Temperature > 0 || req.MaxTokens > 0 {
		cfg = &genai.GenerateContentConfig{}
	}
	if cfg != nil && systemPrompt != "" {
		cfg.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: systemPrompt}},
		}
	}
	if cfg != nil && req.Temperature > 0 {
		cfg.Temperature = genai.Ptr[float32](float32(req.Temperature))
	}
	if cfg != nil && req.MaxTokens > 0 {
		cfg.MaxOutputTokens = int32(req.MaxTokens)
	}

	return contents, cfg
}

func (p *Provider) handleResponse(
	ctx context.Context,
	req *providers.ProxyRequest,
	contents []*genai.Content,
	cfg *genai.GenerateContentConfig,
) (*providers.ProxyResponse, error) {
	resp, err := p.client.Models.GenerateContent(ctx, req.Model, contents, cfg)
	if err != nil {
		return nil, toProviderError(err)
	}

	id := req.RequestID
	if id == "" {
		if resp != nil && resp.ResponseID != "" {
			id = resp.ResponseID
		} else {
			id = generateID()
		}
	}

	out := ""
	if resp != nil {
		out = resp.Text()
	}

	var inTok, outTok int
	if resp != nil && resp.UsageMetadata != nil {
		inTok = int(resp.UsageMetadata.PromptTokenCount)
		outTok = int(resp.UsageMetadata.CandidatesTokenCount)
	}

	return &providers.ProxyResponse{
		ID:      id,
		Model:   req.Model,
		Content: out,
		Usage: providers.Usage{
			InputTokens:  inTok,
			OutputTokens: outTok,
		},
	}, nil
}

func (p *Provider) handleStreaming(
	ctx context.Context,
	model string,
	contents []*genai.Content,
	cfg *genai.GenerateContentConfig,
) (*providers.ProxyResponse, error) {
	ch := make(chan providers.StreamChunk, 64)

	go func() {
		defer close(ch)

		for resp, err := range p.client.Models.GenerateContentStream(ctx, model, contents, cfg) {
			if err != nil {
				ch <- providers.StreamChunk{
					Content:      fmt.Sprintf("[stream error] %v", err),
					FinishReason: "error",
				}
				return
			}
			if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0] == nil {
				continue
			}

			c := resp.Candidates[0]
			text := firstCandidateText(c)
			finish := string(c.FinishReason)

			if text != "" || finish != "" {
				ch <- providers.StreamChunk{Content: text, FinishReason: finish}
			}
		}
	}()

	return &providers.ProxyResponse{Stream: ch}, nil
}

func firstCandidateText(c *genai.Candidate) string {
	if c == nil || c.Content == nil || len(c.Content.Parts) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, p := range c.Content.Parts {
		if p != nil && p.Text != "" {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

func generateID() string {
	return fmt.Sprintf("vertexai-%x", rand.Int63())
}

// ProviderError wraps a Vertex AI API error.
type ProviderError struct {
	StatusCode int
	Message    string
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("vertexai: %s (status=%d)", e.Message, e.StatusCode)
}

func (e *ProviderError) HTTPStatus() int { return e.StatusCode }

func toProviderError(err error) error {
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		return &ProviderError{
			StatusCode: apiErr.Code,
			Message:    apiErr.Message,
		}
	}
	return err
}
