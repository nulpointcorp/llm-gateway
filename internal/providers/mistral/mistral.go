package mistral

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

const (
	defaultBaseURL = "https://api.mistral.ai/v1"
	providerName   = "mistral"
)

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Stream      bool          `json:"stream,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	ID      string   `json:"id"`
	Model   string   `json:"model"`
	Choices []choice `json:"choices"`
	Usage   usage    `json:"usage"`
	Error   *apiErr  `json:"error,omitempty"`
}

type choice struct {
	Message      *chatMessage `json:"message,omitempty"`
	Delta        *chatMessage `json:"delta,omitempty"`
	FinishReason string       `json:"finish_reason"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type apiErr struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

type embeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embeddingData struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type embeddingResponse struct {
	Object string          `json:"object"`
	Model  string          `json:"model"`
	Data   []embeddingData `json:"data"`
	Usage  embeddingUsage  `json:"usage"`
	Error  *apiErr         `json:"error,omitempty"`
}

type Provider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

type Option func(*Provider)

func WithBaseURL(url string) Option {
	return func(p *Provider) { p.baseURL = url }
}

func New(apiKey string, opts ...Option) *Provider {
	p := &Provider{
		apiKey:  apiKey,
		baseURL: defaultBaseURL,
		client:  &http.Client{Timeout: providers.ProviderTimeout},
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *Provider) Name() string { return providerName }

func (p *Provider) HealthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/models", nil)
	if err != nil {
		return fmt.Errorf("mistral: health check: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("mistral: health check: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("mistral: health check: status %d", resp.StatusCode)
	}
	return nil
}

func (p *Provider) Request(ctx context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	body, err := p.buildRequest(req)
	if err != nil {
		return nil, fmt.Errorf("mistral: %w", err)
	}

	apiKey, err := p.effectiveAPIKey(req.APIKey)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("mistral: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	if req.Stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("mistral: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, p.parseError(resp)
	}

	if req.Stream {
		return p.handleStreaming(resp)
	}
	defer resp.Body.Close()

	return p.handleResponse(resp)
}

func (p *Provider) buildRequest(req *providers.ProxyRequest) ([]byte, error) {
	msgs := make([]chatMessage, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = chatMessage{Role: m.Role, Content: m.Content}
	}
	cr := chatRequest{
		Model:    req.Model,
		Messages: msgs,
	}
	if req.Stream {
		cr.Stream = true
	}
	if req.Temperature > 0 {
		cr.Temperature = req.Temperature
	}
	if req.MaxTokens > 0 {
		cr.MaxTokens = req.MaxTokens
	}

	data, err := json.Marshal(cr)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	return data, nil
}

func (p *Provider) handleResponse(resp *http.Response) (*providers.ProxyResponse, error) {
	var cr chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return nil, fmt.Errorf("mistral: decode response: %w", err)
	}

	content := ""
	if len(cr.Choices) > 0 && cr.Choices[0].Message != nil {
		content = cr.Choices[0].Message.Content
	}

	return &providers.ProxyResponse{
		ID:      cr.ID,
		Model:   cr.Model,
		Content: content,
		Usage: providers.Usage{
			InputTokens:  cr.Usage.PromptTokens,
			OutputTokens: cr.Usage.CompletionTokens,
		},
	}, nil
}

func (p *Provider) handleStreaming(resp *http.Response) (*providers.ProxyResponse, error) {
	ch := make(chan providers.StreamChunk, 64)

	go func() {
		defer resp.Body.Close()
		defer close(ch)

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				return
			}

			var cr chatResponse
			if err := json.Unmarshal([]byte(data), &cr); err != nil {
				continue
			}
			if len(cr.Choices) == 0 || cr.Choices[0].Delta == nil {
				continue
			}

			ch <- providers.StreamChunk{
				Content:      cr.Choices[0].Delta.Content,
				FinishReason: cr.Choices[0].FinishReason,
			}
		}
	}()

	return &providers.ProxyResponse{Stream: ch}, nil
}

// Embed implements providers.EmbeddingProvider.
func (p *Provider) Embed(ctx context.Context, req *providers.EmbeddingRequest) (*providers.EmbeddingResponse, error) {
	body, err := json.Marshal(embeddingRequest{
		Model: req.Model,
		Input: req.Input,
	})
	if err != nil {
		return nil, fmt.Errorf("mistral: embed: marshal request: %w", err)
	}

	apiKey, err := p.effectiveAPIKey(req.APIKey)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("mistral: embed: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("mistral: embed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, p.parseError(resp)
	}

	var er embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&er); err != nil {
		return nil, fmt.Errorf("mistral: embed: decode response: %w", err)
	}

	data := make([]providers.EmbeddingData, len(er.Data))
	for i, d := range er.Data {
		data[i] = providers.EmbeddingData{
			Index:     d.Index,
			Embedding: d.Embedding,
		}
	}

	return &providers.EmbeddingResponse{
		Model: er.Model,
		Data:  data,
		Usage: providers.Usage{
			InputTokens: er.Usage.PromptTokens,
		},
	}, nil
}

func (p *Provider) parseError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	var cr chatResponse
	if json.Unmarshal(body, &cr) == nil && cr.Error != nil {
		return &ProviderError{
			StatusCode: resp.StatusCode,
			Message:    cr.Error.Message,
			Type:       cr.Error.Type,
			Code:       cr.Error.Code,
		}
	}

	return &ProviderError{
		StatusCode: resp.StatusCode,
		Message:    fmt.Sprintf("unexpected status %d", resp.StatusCode),
		Type:       "provider_error",
	}
}

type ProviderError struct {
	StatusCode int
	Message    string
	Type       string
	Code       string
}

// Error implements the error interface.
func (e *ProviderError) Error() string {
	return fmt.Sprintf("mistral: %s (status=%d, type=%s)", e.Message, e.StatusCode, e.Type)
}

// HTTPStatus implements providers.StatusCoder.
func (e *ProviderError) HTTPStatus() int { return e.StatusCode }

func (p *Provider) effectiveAPIKey(override string) (string, error) {
	if override != "" {
		return override, nil
	}
	if p.apiKey == "" {
		return "", fmt.Errorf("mistral: no API key configured")
	}
	return p.apiKey, nil
}
