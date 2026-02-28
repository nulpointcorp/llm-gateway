// Package openaicompat provides a generic OpenAI-compatible LLM provider.
// Use it for any service that implements the OpenAI chat completions API
// (xAI, Groq, DeepSeek, Together AI, Perplexity, Cerebras, etc.).
package openaicompat

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
	openaiSDK "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// Provider is a configurable OpenAI-compatible LLM provider.
type Provider struct {
	name    string
	apiKey  string
	baseURL string
	client  openaiSDK.Client
}

// New creates a new OpenAI-compatible Provider.
//
//   - name    — unique provider identifier used for routing and logs.
//   - apiKey  — API key sent as "Authorization: Bearer <key>".
//   - baseURL — API base URL, e.g. "https://api.x.ai/v1".
func New(name, apiKey, baseURL string) *Provider {
	p := &Provider{
		name:    name,
		apiKey:  apiKey,
		baseURL: baseURL,
	}

	opts := []option.RequestOption{
		option.WithAPIKey(p.apiKey),
		option.WithHTTPClient(&http.Client{Timeout: providers.ProviderTimeout}),
	}
	if p.baseURL != "" {
		opts = append(opts, option.WithBaseURL(p.baseURL))
	}

	p.client = openaiSDK.NewClient(opts...)
	return p
}

func (p *Provider) Name() string { return p.name }

func (p *Provider) HealthCheck(ctx context.Context) error {
	_, err := p.client.Models.List(ctx)
	if err != nil {
		return fmt.Errorf("%s: health check: %w", p.name, p.toProviderError(err))
	}
	return nil
}

func (p *Provider) Request(ctx context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	params := p.buildParams(req)
	opts, err := p.requestOptions(req.APIKey)
	if err != nil {
		return nil, err
	}
	if req.Stream {
		return p.handleStreaming(ctx, params, opts...)
	}
	return p.handleResponse(ctx, params, opts...)
}

func (p *Provider) buildParams(req *providers.ProxyRequest) openaiSDK.ChatCompletionNewParams {
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

	return params
}

func (p *Provider) handleResponse(
	ctx context.Context,
	params openaiSDK.ChatCompletionNewParams,
	opts ...option.RequestOption,
) (*providers.ProxyResponse, error) {
	resp, err := p.client.Chat.Completions.New(ctx, params, opts...)
	if err != nil {
		return nil, p.toProviderError(err)
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
				ch <- providers.StreamChunk{FinishReason: c.FinishReason}
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

// ProviderError is a structured error returned by an OpenAI-compatible API.
type ProviderError struct {
	Name       string
	StatusCode int
	Message    string
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("%s: %s (status=%d)", e.Name, e.Message, e.StatusCode)
}

func (e *ProviderError) HTTPStatus() int { return e.StatusCode }

func (p *Provider) toProviderError(err error) error {
	var apierr *openaiSDK.Error
	if errors.As(err, &apierr) {
		return &ProviderError{
			Name:       p.name,
			StatusCode: apierr.StatusCode,
			Message:    apierr.Error(),
		}
	}
	return err
}

func (p *Provider) requestOptions(overrideKey string) ([]option.RequestOption, error) {
	key := overrideKey
	if key == "" {
		key = p.apiKey
	}
	if key == "" {
		return nil, fmt.Errorf("%s: no API key configured", p.name)
	}
	return []option.RequestOption{option.WithAPIKey(key)}, nil
}

func toSDKMessage(role, content string) openaiSDK.ChatCompletionMessageParamUnion {
	switch strings.ToLower(role) {
	case "developer":
		return openaiSDK.DeveloperMessage(content)
	case "system":
		return openaiSDK.SystemMessage(content)
	case "assistant":
		return openaiSDK.AssistantMessage(content)
	default:
		return openaiSDK.UserMessage(content)
	}
}
