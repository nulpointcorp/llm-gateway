package openai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
	openaiSDK "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

const (
	defaultBaseURL = "https://api.openai.com/v1"
	providerName   = "openai"
)

type Provider struct {
	apiKey  string
	baseURL string
	client  openaiSDK.Client
}

type Option func(*Provider)

func WithBaseURL(u string) Option {
	return func(p *Provider) { p.baseURL = u }
}

func New(apiKey string, opts ...Option) *Provider {
	p := &Provider{
		apiKey:  apiKey,
		baseURL: defaultBaseURL,
	}

	for _, o := range opts {
		o(p)
	}

	httpClient := &http.Client{Timeout: providers.ProviderTimeout}
	if p.baseURL != "" && p.baseURL != defaultBaseURL {
		httpClient.Transport = newBaseURLTransport(http.DefaultTransport, p.baseURL)
	}

	p.client = openaiSDK.NewClient(
		option.WithAPIKey(p.apiKey),
		option.WithHTTPClient(httpClient),
	)

	return p
}

func (p *Provider) Name() string { return providerName }

func (p *Provider) HealthCheck(ctx context.Context) error {
	_, err := p.client.Models.List(ctx)
	if err != nil {
		return fmt.Errorf("openai: health check: %w", toProviderError(err))
	}
	return nil
}

func (p *Provider) Request(ctx context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	params, err := p.buildChatCompletionParams(req)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}

	opts, err := p.requestOptions(req.APIKey)
	if err != nil {
		return nil, err
	}

	if req.Stream {
		return p.handleStreaming(ctx, params, opts...)
	}
	return p.handleResponse(ctx, params, opts...)
}

func (p *Provider) buildChatCompletionParams(req *providers.ProxyRequest) (openaiSDK.ChatCompletionNewParams, error) {
	msgs := make([]openaiSDK.ChatCompletionMessageParamUnion, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, toSDKMessage(m.Role, m.Content))
	}

	params := openaiSDK.ChatCompletionNewParams{
		Messages: msgs,
		Model:    req.Model,
	}

	if req.Temperature != 0 {
		params.Temperature = openaiSDK.Float(req.Temperature)
	}

	if req.MaxTokens > 0 {
		params.MaxCompletionTokens = openaiSDK.Int(int64(req.MaxTokens))
	}

	return params, nil
}

func (p *Provider) handleResponse(
	ctx context.Context,
	params openaiSDK.ChatCompletionNewParams,
	opts ...option.RequestOption,
) (*providers.ProxyResponse, error) {
	resp, err := p.client.Chat.Completions.New(ctx, params, opts...)
	if err != nil {
		return nil, toProviderError(err)
	}

	content := ""
	if len(resp.Choices) > 0 {
		content = resp.Choices[0].Message.Content
	}

	return &providers.ProxyResponse{
		ID:      resp.ID,
		Model:   resp.Model,
		Content: content,
		Usage: providers.Usage{
			InputTokens:  int(resp.Usage.PromptTokens),
			OutputTokens: int(resp.Usage.CompletionTokens),
		},
	}, nil
}

func (p *Provider) handleStreaming(
	ctx context.Context,
	params openaiSDK.ChatCompletionNewParams,
	opts ...option.RequestOption,
) (*providers.ProxyResponse, error) {
	ch := make(chan providers.StreamChunk, 64)

	stream := p.client.Chat.Completions.NewStreaming(ctx, params, opts...)

	go func() {
		defer close(ch)

		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) == 0 {
				continue
			}

			c := chunk.Choices[0]

			if c.Delta.Content != "" {
				ch <- providers.StreamChunk{
					Content:      c.Delta.Content,
					FinishReason: c.FinishReason,
				}
				continue
			}

			if c.FinishReason != "" {
				ch <- providers.StreamChunk{
					Content:      "",
					FinishReason: c.FinishReason,
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- providers.StreamChunk{
				Content:      fmt.Sprintf("[stream error] %v", err),
				FinishReason: "error",
			}
		}
	}()

	return &providers.ProxyResponse{Stream: ch}, nil
}

// Embed implements providers.EmbeddingProvider.
func (p *Provider) Embed(ctx context.Context, req *providers.EmbeddingRequest) (*providers.EmbeddingResponse, error) {
	params := openaiSDK.EmbeddingNewParams{
		Model: openaiSDK.EmbeddingModel(req.Model),
		Input: openaiSDK.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: req.Input,
		},
	}

	opts, err := p.requestOptions(req.APIKey)
	if err != nil {
		return nil, err
	}

	resp, err := p.client.Embeddings.New(ctx, params, opts...)
	if err != nil {
		return nil, toProviderError(err)
	}

	data := make([]providers.EmbeddingData, len(resp.Data))
	for i, d := range resp.Data {
		f32 := make([]float32, len(d.Embedding))
		for j, v := range d.Embedding {
			f32[j] = float32(v)
		}
		data[i] = providers.EmbeddingData{
			Index:     int(d.Index),
			Embedding: f32,
		}
	}

	return &providers.EmbeddingResponse{
		Model: resp.Model,
		Data:  data,
		Usage: providers.Usage{
			InputTokens: int(resp.Usage.PromptTokens),
		},
	}, nil
}

func (p *Provider) requestOptions(overrideKey string) ([]option.RequestOption, error) {
	key := overrideKey
	if key == "" {
		key = p.apiKey
	}
	if key == "" {
		return nil, fmt.Errorf("openai: no API key configured")
	}
	return []option.RequestOption{option.WithAPIKey(key)}, nil
}

type ProviderError struct {
	StatusCode int
	Message    string
	Type       string
	Code       string
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("openai: %s (status=%d, type=%s)", e.Message, e.StatusCode, e.Type)
}

func (e *ProviderError) HTTPStatus() int { return e.StatusCode }

func toProviderError(err error) error {
	var apierr *openaiSDK.Error
	if errors.As(err, &apierr) {
		return &ProviderError{
			StatusCode: apierr.StatusCode,
			Message:    apierr.Error(),
			Type:       "openai_error",
		}
	}
	return err
}

type baseURLTransport struct {
	base *url.URL
	rt   http.RoundTripper
}

func newBaseURLTransport(next http.RoundTripper, base string) http.RoundTripper {
	u, err := url.Parse(base)
	if err != nil {

		return next
	}
	return &baseURLTransport{base: u, rt: next}
}

func (t *baseURLTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r2 := req.Clone(req.Context())
	u2 := *req.URL

	u2.Scheme = t.base.Scheme
	u2.Host = t.base.Host

	basePath := strings.TrimRight(t.base.Path, "/")
	if basePath != "" && basePath != "/" {
		if !strings.HasPrefix(u2.Path, basePath+"/") && u2.Path != basePath {
			u2.Path = basePath + "/" + strings.TrimLeft(u2.Path, "/")
		}
	}

	r2.URL = &u2

	return t.rt.RoundTrip(r2)
}

func toSDKMessage(role, content string) openaiSDK.ChatCompletionMessageParamUnion {
	switch strings.ToLower(role) {
	case "developer":
		return openaiSDK.DeveloperMessage(content)
	case "system":
		return openaiSDK.SystemMessage(content)
	case "assistant":

		return openaiSDK.AssistantMessage(content)
	case "user":
		fallthrough
	default:
		return openaiSDK.UserMessage(content)
	}
}
