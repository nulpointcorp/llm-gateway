// Package azure implements the providers.Provider interface for Azure OpenAI.
// Azure OpenAI uses deployment-based URLs and the "api-key" header instead of
// the standard "Authorization: Bearer" scheme.
//
// Required configuration:
//   - AZURE_OPENAI_ENDPOINT   — e.g. "https://myresource.openai.azure.com"
//   - AZURE_OPENAI_API_KEY    — your Azure OpenAI resource key
//   - AZURE_OPENAI_API_VERSION — API version, e.g. "2024-12-01-preview"
//
// Model routing: model names with the "azure-" prefix have the prefix stripped
// to derive the deployment name. E.g. "azure-gpt-4o" → deployment "gpt-4o".
// Models without the prefix are used as-is as the deployment name.
package azure

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

const providerName = "azure"

type chatRequest struct {
	Model       string        `json:"model,omitempty"`
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

// Provider implements providers.Provider for Azure OpenAI.
type Provider struct {
	endpoint   string // e.g. "https://myresource.openai.azure.com"
	apiKey     string
	apiVersion string
	client     *http.Client
}

// Option configures a Provider.
type Option func(*Provider)

// New creates a new Azure OpenAI Provider.
func New(endpoint, apiKey, apiVersion string, opts ...Option) *Provider {
	p := &Provider{
		endpoint:   strings.TrimRight(endpoint, "/"),
		apiKey:     apiKey,
		apiVersion: apiVersion,
		client:     &http.Client{Timeout: providers.ProviderTimeout},
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *Provider) Name() string { return providerName }

func (p *Provider) HealthCheck(ctx context.Context) error {
	url := fmt.Sprintf("%s/openai/models?api-version=%s", p.endpoint, p.apiVersion)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("azure: health check: %w", err)
	}
	req.Header.Set("api-key", p.apiKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("azure: health check: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("azure: health check: status %d", resp.StatusCode)
	}
	return nil
}

func (p *Provider) Request(ctx context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	deployment := deploymentName(req.Model)
	url := p.completionsURL(deployment)

	body, err := p.buildRequest(req)
	if err != nil {
		return nil, fmt.Errorf("azure: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("azure: %w", err)
	}
	httpReq.Header.Set("api-key", p.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	if req.Stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("azure: %w", err)
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

// deploymentName strips the "azure-" prefix if present, yielding the
// Azure deployment name used in the URL.
func deploymentName(model string) string {
	return strings.TrimPrefix(model, "azure-")
}

func (p *Provider) completionsURL(deployment string) string {
	return fmt.Sprintf(
		"%s/openai/deployments/%s/chat/completions?api-version=%s",
		p.endpoint, deployment, p.apiVersion,
	)
}

func (p *Provider) buildRequest(req *providers.ProxyRequest) ([]byte, error) {
	msgs := make([]chatMessage, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = chatMessage{Role: m.Role, Content: m.Content}
	}
	cr := chatRequest{Messages: msgs}
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
		return nil, fmt.Errorf("azure: decode response: %w", err)
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

// ProviderError is a structured error returned by the Azure OpenAI API.
type ProviderError struct {
	StatusCode int
	Message    string
	Type       string
	Code       string
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("azure: %s (status=%d, type=%s)", e.Message, e.StatusCode, e.Type)
}

func (e *ProviderError) HTTPStatus() int { return e.StatusCode }

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
		Type:       "azure_error",
	}
}

func (p *Provider) effectiveAPIKey(override string) (string, error) {
	if override != "" {
		return override, nil
	}
	if p.apiKey == "" {
		return "", fmt.Errorf("azure: no API key configured")
	}
	return p.apiKey, nil
}
