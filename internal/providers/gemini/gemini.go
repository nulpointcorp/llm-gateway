package gemini

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"net/url"
	"strings"

	"google.golang.org/genai"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

const (
	defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"
	providerName   = "gemini"
)

// Provider implements providers.Provider for Google Gemini (official GenAI SDK).
type Provider struct {
	apiKey     string
	baseURL    string
	client     *genai.Client
	httpClient *http.Client
	base       string
	apiVersion string
}

// Option configures a Provider.
type Option func(*Provider)

// WithBaseURL overrides the API base URL (useful for testing).
func WithBaseURL(u string) Option {
	return func(p *Provider) { p.baseURL = u }
}

// New creates a new Gemini Provider.
func New(ctx context.Context, apiKey string, opts ...Option) *Provider {
	if ctx == nil {
		panic("gemini: context must not be nil")
	}
	p := &Provider{
		apiKey:  apiKey,
		baseURL: defaultBaseURL,
	}
	for _, o := range opts {
		o(p)
	}

	httpClient := &http.Client{Timeout: providers.ProviderTimeout}
	p.httpClient = httpClient

	base, ver := splitBaseURLAndVersion(p.baseURL)
	p.base = base
	p.apiVersion = ver

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:      p.apiKey,
		Backend:     genai.BackendGeminiAPI,
		HTTPClient:  p.httpClient,
		HTTPOptions: genai.HTTPOptions{BaseURL: p.base, APIVersion: p.apiVersion},
	})
	if err != nil {
		return nil
	}

	p.client = client

	return p
}

func (p *Provider) Name() string { return providerName }

func (p *Provider) HealthCheck(ctx context.Context) error {
	_, err := p.client.Models.List(ctx, &genai.ListModelsConfig{PageSize: 1})
	if err != nil {
		return fmt.Errorf("gemini: health check: %w", toProviderError(err))
	}
	return nil
}

func (p *Provider) Request(ctx context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	contents, cfg := p.buildContentsAndConfig(req)

	client, err := p.clientForKey(ctx, req.APIKey)
	if err != nil {
		return nil, err
	}

	if req.Stream {
		return p.handleStreaming(ctx, client, req.Model, contents, cfg)
	}
	return p.handleResponse(ctx, client, req, contents, cfg)
}

func (p *Provider) buildContentsAndConfig(req *providers.ProxyRequest) ([]*genai.Content, *genai.GenerateContentConfig) {
	var systemPrompt string
	contents := make([]*genai.Content, 0, len(req.Messages))

	for _, m := range req.Messages {
		switch strings.ToLower(m.Role) {
		case "system", "developer":
			if systemPrompt != "" {
				systemPrompt += "\n"
			}
			systemPrompt += m.Content

		case "assistant":
			contents = append(contents, genai.NewContentFromText(m.Content, genai.RoleModel))

		case "model":
			contents = append(contents, genai.NewContentFromText(m.Content, genai.RoleModel))

		default: // user / unknown
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
	client *genai.Client,
	req *providers.ProxyRequest,
	contents []*genai.Content,
	cfg *genai.GenerateContentConfig,
) (*providers.ProxyResponse, error) {
	resp, err := client.Models.GenerateContent(ctx, req.Model, contents, cfg)
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
	client *genai.Client,
	model string,
	contents []*genai.Content,
	cfg *genai.GenerateContentConfig,
) (*providers.ProxyResponse, error) {
	ch := make(chan providers.StreamChunk, 64)

	go func() {
		defer close(ch)

		for resp, err := range client.Models.GenerateContentStream(ctx, model, contents, cfg) {
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
			finish := ""
			if c.FinishReason != "" {
				finish = string(c.FinishReason)
			}

			if text != "" || finish != "" {
				ch <- providers.StreamChunk{
					Content:      text,
					FinishReason: finish,
				}
			}
		}
	}()

	return &providers.ProxyResponse{Stream: ch}, nil
}

// Embed implements providers.EmbeddingProvider.
// All input strings are sent in a single EmbedContent call as a batch of Contents.
func (p *Provider) Embed(ctx context.Context, req *providers.EmbeddingRequest) (*providers.EmbeddingResponse, error) {
	contents := make([]*genai.Content, len(req.Input))
	for i, text := range req.Input {
		contents[i] = genai.NewContentFromText(text, genai.RoleUser)
	}

	client, err := p.clientForKey(ctx, req.APIKey)
	if err != nil {
		return nil, err
	}

	resp, err := client.Models.EmbedContent(ctx, req.Model, contents, nil)
	if err != nil {
		return nil, fmt.Errorf("gemini: embed: %w", toProviderError(err))
	}
	if resp == nil || len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("gemini: embed: empty response")
	}

	data := make([]providers.EmbeddingData, len(resp.Embeddings))
	for i, emb := range resp.Embeddings {
		if emb == nil {
			continue
		}
		data[i] = providers.EmbeddingData{
			Index:     i,
			Embedding: emb.Values,
		}
	}

	return &providers.EmbeddingResponse{
		Model: req.Model,
		Data:  data,
	}, nil
}

func (p *Provider) clientForKey(ctx context.Context, overrideKey string) (*genai.Client, error) {
	key := overrideKey
	if key == "" {
		key = p.apiKey
	}
	if key == "" {
		return nil, fmt.Errorf("gemini: no API key configured")
	}
	if key == p.apiKey {
		return p.client, nil
	}
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:      key,
		Backend:     genai.BackendGeminiAPI,
		HTTPClient:  p.httpClient,
		HTTPOptions: genai.HTTPOptions{BaseURL: p.base, APIVersion: p.apiVersion},
	})
	if err != nil {
		return nil, fmt.Errorf("gemini: override client: %w", err)
	}
	return client, nil
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

func splitBaseURLAndVersion(raw string) (baseURL string, apiVersion string) {
	u, err := url.Parse(raw)
	if err != nil {
		return raw, ""
	}

	path := strings.Trim(u.Path, "/")
	if path == "" {
		base := u.String()
		if !strings.HasSuffix(base, "/") {
			base += "/"
		}
		return base, ""
	}

	parts := strings.Split(path, "/")
	last := parts[len(parts)-1]

	if looksLikeAPIVersion(last) {
		apiVersion = last
		parts = parts[:len(parts)-1]
	}

	u.Path = "/" + strings.Join(parts, "/")
	if u.Path == "/" {
		u.Path = ""
	}

	baseURL = u.String()
	if !strings.HasSuffix(baseURL, "/") {
		baseURL += "/"
	}
	return baseURL, apiVersion
}

func looksLikeAPIVersion(s string) bool {
	if !strings.HasPrefix(s, "v") || len(s) < 2 {
		return false
	}
	// Вторая руна должна быть цифрой
	return s[1] >= '0' && s[1] <= '9'
}

// generateID produces a random hex ID for responses that don't include one.
func generateID() string {
	return fmt.Sprintf("gemini-%x", rand.Int63())
}

// ProviderError is a structured error returned by the Gemini API (SDK wrapper).
type ProviderError struct {
	StatusCode int
	Message    string
	Type       string
	Code       string
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("gemini: %s (status=%d, type=%s)", e.Message, e.StatusCode, e.Type)
}

// HTTPStatus implements providers.StatusCoder.
func (e *ProviderError) HTTPStatus() int { return e.StatusCode }

func toProviderError(err error) error {
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		return &ProviderError{
			StatusCode: apiErr.Code,
			Message:    apiErr.Message,
			Type:       apiErr.Status,
			Code:       fmt.Sprintf("%d", apiErr.Code),
		}
	}
	return err
}
