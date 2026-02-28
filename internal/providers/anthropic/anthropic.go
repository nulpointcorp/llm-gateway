package anthropic

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

const (
	defaultBaseURL   = "https://api.anthropic.com/v1"
	providerName     = "anthropic"
	defaultMaxTokens = 4096
)

// Provider implements providers.Provider for Anthropic (official SDK).
type Provider struct {
	apiKey  string
	baseURL string
	client  anthropic.Client
}

// Option configures a Provider.
type Option func(*Provider)

// WithBaseURL overrides the API base URL (useful for testing).
func WithBaseURL(url string) Option {
	return func(p *Provider) { p.baseURL = url }
}

// New creates a new Anthropic Provider.
func New(apiKey string, opts ...Option) *Provider {
	p := &Provider{
		apiKey:  apiKey,
		baseURL: defaultBaseURL,
	}
	for _, o := range opts {
		o(p)
	}

	httpClient := &http.Client{Timeout: providers.ProviderTimeout}

	p.client = anthropic.NewClient(
		option.WithAPIKey(p.apiKey),
		option.WithBaseURL(p.baseURL),
		option.WithHTTPClient(httpClient),
	)

	return p
}

func (p *Provider) Name() string { return providerName }

func (p *Provider) HealthCheck(ctx context.Context) error {
	// Simple auth/connectivity check: GET /v1/models
	_, err := p.client.Models.List(ctx, anthropic.ModelListParams{
		Limit: anthropic.Int(1),
	})
	if err != nil {
		return fmt.Errorf("anthropic: health check: %w", toProviderError(err))
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

func (p *Provider) buildParams(req *providers.ProxyRequest) anthropic.MessageNewParams {
	var systemPrompt string
	msgs := make([]anthropic.MessageParam, 0, len(req.Messages))

	for _, m := range req.Messages {
		switch strings.ToLower(m.Role) {
		case "system", "developer":
			if systemPrompt != "" {
				systemPrompt += "\n"
			}
			systemPrompt += m.Content
		default:
			msgs = append(msgs, toSDKMessage(m.Role, m.Content))
		}
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = defaultMaxTokens
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(req.Model),
		MaxTokens: int64(maxTokens),
		Messages:  msgs,
	}

	if systemPrompt != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: systemPrompt},
		}
	}

	// Temperature is optional in Anthropic; set only if provided.
	// (param.Field[float64] -> use helper, as in SDK examples)
	if req.Temperature > 0 {
		params.Temperature = anthropic.Float(req.Temperature)
	}

	return params
}

func toSDKMessage(role, content string) anthropic.MessageParam {
	r := strings.ToLower(role)
	anthRole := anthropic.MessageParamRoleUser
	if r == "assistant" {
		anthRole = anthropic.MessageParamRoleAssistant
	}

	return anthropic.MessageParam{
		Role: anthRole,
		Content: []anthropic.ContentBlockParamUnion{
			{
				OfText: &anthropic.TextBlockParam{
					Text: content,
				},
			},
		},
	}
}

func (p *Provider) handleResponse(
	ctx context.Context,
	params anthropic.MessageNewParams,
	opts ...option.RequestOption,
) (*providers.ProxyResponse, error) {
	msg, err := p.client.Messages.New(ctx, params, opts...)
	if err != nil {
		return nil, toProviderError(err)
	}

	// Собираем весь текст из всех text-блоков.
	var sb strings.Builder
	for _, b := range msg.Content {
		switch v := b.AsAny().(type) {
		case anthropic.TextBlock:
			sb.WriteString(v.Text)
		case *anthropic.TextBlock:
			sb.WriteString(v.Text)
		}
	}

	return &providers.ProxyResponse{
		ID:      msg.ID,
		Model:   string(msg.Model),
		Content: sb.String(),
		Usage: providers.Usage{
			InputTokens:  int(msg.Usage.InputTokens),
			OutputTokens: int(msg.Usage.OutputTokens),
		},
	}, nil
}

func (p *Provider) handleStreaming(
	ctx context.Context,
	params anthropic.MessageNewParams,
	opts ...option.RequestOption,
) (*providers.ProxyResponse, error) {
	ch := make(chan providers.StreamChunk, 64)

	stream := p.client.Messages.NewStreaming(ctx, params, opts...)

	go func() {
		defer close(ch)

		for stream.Next() {
			ev := stream.Current()

			switch eventVariant := ev.AsAny().(type) {
			case anthropic.ContentBlockDeltaEvent:
				switch deltaVariant := eventVariant.Delta.AsAny().(type) {
				case anthropic.TextDelta:
					if deltaVariant.Text != "" {
						ch <- providers.StreamChunk{Content: deltaVariant.Text}
					}
				case *anthropic.TextDelta:
					if deltaVariant.Text != "" {
						ch <- providers.StreamChunk{Content: deltaVariant.Text}
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			// У вас нет error-канала в StreamChunk, поэтому шлём как финальный chunk.
			ch <- providers.StreamChunk{
				Content:      fmt.Sprintf("[stream error] %v", err),
				FinishReason: "error",
			}
		}
	}()

	return &providers.ProxyResponse{Stream: ch}, nil
}

func (p *Provider) requestOptions(overrideKey string) ([]option.RequestOption, error) {
	key := overrideKey
	if key == "" {
		key = p.apiKey
	}
	if key == "" {
		return nil, fmt.Errorf("anthropic: no API key configured")
	}
	return []option.RequestOption{option.WithAPIKey(key)}, nil
}

// ProviderError is a structured error returned by the Anthropic API.
type ProviderError struct {
	StatusCode int
	Message    string
	Type       string
	Code       string
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("anthropic: %s (status=%d, type=%s)", e.Message, e.StatusCode, e.Type)
}

// HTTPStatus implements providers.StatusCoder.
func (e *ProviderError) HTTPStatus() int { return e.StatusCode }

func toProviderError(err error) error {
	var apierr *anthropic.Error
	if errors.As(err, &apierr) {
		return &ProviderError{
			StatusCode: apierr.StatusCode,
			Message:    apierr.Error(),
			Type:       "anthropic_error",
		}
	}
	return err
}
