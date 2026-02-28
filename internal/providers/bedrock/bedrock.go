// Package bedrock implements the providers.Provider interface for AWS Bedrock.
// It uses the Bedrock Converse API with AWS SigV4 request signing.
//
// Required configuration:
//   - AWS_ACCESS_KEY_ID
//   - AWS_SECRET_ACCESS_KEY
//   - AWS_REGION (e.g. "us-east-1")
//
// Optional:
//   - AWS_SESSION_TOKEN — for temporary credentials (IAM roles, STS).
package bedrock

import (
	"bufio"
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

const (
	providerName = "bedrock"
	service      = "bedrock"
	algorithm    = "AWS4-HMAC-SHA256"
)

// Provider implements providers.Provider for AWS Bedrock via the Converse API.
type Provider struct {
	accessKey    string
	secretKey    string
	sessionToken string
	region       string
	endpointURL  string // optional override for the base endpoint (testing)
	client       *http.Client
}

// Option configures a Provider.
type Option func(*Provider)

// WithSessionToken sets the AWS session token for temporary credentials.
func WithSessionToken(token string) Option {
	return func(p *Provider) { p.sessionToken = token }
}

// WithEndpointURL overrides the Bedrock endpoint base URL (e.g. for local mocks).
// When set, all API calls use this URL instead of the regional AWS endpoint.
func WithEndpointURL(u string) Option {
	return func(p *Provider) { p.endpointURL = u }
}

// New creates a new AWS Bedrock Provider.
func New(accessKey, secretKey, region string, opts ...Option) *Provider {
	p := &Provider{
		accessKey: accessKey,
		secretKey: secretKey,
		region:    region,
		client:    &http.Client{Timeout: providers.ProviderTimeout},
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *Provider) Name() string { return providerName }

func (p *Provider) HealthCheck(ctx context.Context) error {
	// GET /foundation-models — list available models
	base := p.baseEndpoint("bedrock")
	endpoint := base + "/foundation-models"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return fmt.Errorf("bedrock: health check: %w", err)
	}

	if err := p.signRequest(req, nil); err != nil {
		return fmt.Errorf("bedrock: health check sign: %w", err)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("bedrock: health check: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bedrock: health check: status %d", resp.StatusCode)
	}
	return nil
}

func (p *Provider) Request(ctx context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	if req.Stream {
		return p.handleStreaming(ctx, req)
	}
	return p.handleResponse(ctx, req)
}

// ─── Converse API types ───────────────────────────────────────────────────────

type converseRequest struct {
	Messages        []converseMessage `json:"messages"`
	System          []systemContent   `json:"system,omitempty"`
	InferenceConfig *inferenceConfig  `json:"inferenceConfig,omitempty"`
}

type converseMessage struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
}

type contentBlock struct {
	Text string `json:"text"`
}

type systemContent struct {
	Text string `json:"text"`
}

type inferenceConfig struct {
	MaxTokens   int     `json:"maxTokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
}

type converseResponse struct {
	Output converseOutput `json:"output"`
	Usage  converseUsage  `json:"usage"`
}

type converseOutput struct {
	Message converseMessage `json:"message"`
}

type converseUsage struct {
	InputTokens  int `json:"inputTokens"`
	OutputTokens int `json:"outputTokens"`
}

// ─── Request building ─────────────────────────────────────────────────────────

func (p *Provider) buildConverseRequest(req *providers.ProxyRequest) (converseRequest, error) {
	var systemTexts []systemContent
	msgs := make([]converseMessage, 0, len(req.Messages))

	for _, m := range req.Messages {
		switch strings.ToLower(m.Role) {
		case "system", "developer":
			systemTexts = append(systemTexts, systemContent{Text: m.Content})
		default:
			role := "user"
			if strings.ToLower(m.Role) == "assistant" {
				role = "assistant"
			}
			msgs = append(msgs, converseMessage{
				Role:    role,
				Content: []contentBlock{{Text: m.Content}},
			})
		}
	}

	cr := converseRequest{
		Messages: msgs,
		System:   systemTexts,
	}

	if req.MaxTokens > 0 || req.Temperature > 0 {
		cr.InferenceConfig = &inferenceConfig{
			MaxTokens:   req.MaxTokens,
			Temperature: req.Temperature,
		}
	}

	return cr, nil
}

// ─── Non-streaming ────────────────────────────────────────────────────────────

func (p *Provider) handleResponse(ctx context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	body, err := p.buildConverseRequest(req)
	if err != nil {
		return nil, fmt.Errorf("bedrock: build request: %w", err)
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("bedrock: marshal: %w", err)
	}

	endpoint := p.converseEndpoint(req.Model)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("bedrock: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	if err := p.signRequest(httpReq, payload); err != nil {
		return nil, fmt.Errorf("bedrock: sign: %w", err)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, p.parseError(resp)
	}

	var cr converseResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return nil, fmt.Errorf("bedrock: decode response: %w", err)
	}

	content := ""
	if len(cr.Output.Message.Content) > 0 {
		content = cr.Output.Message.Content[0].Text
	}

	return &providers.ProxyResponse{
		ID:      req.RequestID,
		Model:   req.Model,
		Content: content,
		Usage: providers.Usage{
			InputTokens:  cr.Usage.InputTokens,
			OutputTokens: cr.Usage.OutputTokens,
		},
	}, nil
}

// ─── Streaming ────────────────────────────────────────────────────────────────

type streamEvent struct {
	ContentBlockDelta *struct {
		Delta struct {
			Text string `json:"text"`
		} `json:"delta"`
	} `json:"contentBlockDelta"`
	MessageStop *struct {
		StopReason string `json:"stopReason"`
	} `json:"messageStop"`
}

func (p *Provider) handleStreaming(ctx context.Context, req *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	body, err := p.buildConverseRequest(req)
	if err != nil {
		return nil, fmt.Errorf("bedrock: build request: %w", err)
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("bedrock: marshal: %w", err)
	}

	endpoint := p.converseStreamEndpoint(req.Model)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("bedrock: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	if err := p.signRequest(httpReq, payload); err != nil {
		return nil, fmt.Errorf("bedrock: sign: %w", err)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, p.parseError(resp)
	}

	ch := make(chan providers.StreamChunk, 64)

	go func() {
		defer resp.Body.Close()
		defer close(ch)

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			data := strings.TrimPrefix(line, "data:")
			data = strings.TrimSpace(data)

			var ev streamEvent
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				continue
			}

			if ev.ContentBlockDelta != nil && ev.ContentBlockDelta.Delta.Text != "" {
				ch <- providers.StreamChunk{Content: ev.ContentBlockDelta.Delta.Text}
			}
			if ev.MessageStop != nil {
				ch <- providers.StreamChunk{FinishReason: ev.MessageStop.StopReason}
			}
		}
	}()

	return &providers.ProxyResponse{Stream: ch}, nil
}

// ─── Endpoints ───────────────────────────────────────────────────────────────

// baseEndpoint returns the root URL for a given Bedrock sub-service.
// When endpointURL is set (e.g. for testing), it is used for all services.
func (p *Provider) baseEndpoint(subservice string) string {
	if p.endpointURL != "" {
		return strings.TrimRight(p.endpointURL, "/")
	}
	return fmt.Sprintf("https://%s.%s.amazonaws.com", subservice, p.region)
}

func (p *Provider) converseEndpoint(modelID string) string {
	if p.endpointURL != "" {
		return fmt.Sprintf("%s/model/%s/converse", strings.TrimRight(p.endpointURL, "/"), modelID)
	}
	return fmt.Sprintf(
		"https://bedrock-runtime.%s.amazonaws.com/model/%s/converse",
		p.region, modelID,
	)
}

func (p *Provider) converseStreamEndpoint(modelID string) string {
	if p.endpointURL != "" {
		return fmt.Sprintf("%s/model/%s/converse-stream", strings.TrimRight(p.endpointURL, "/"), modelID)
	}
	return fmt.Sprintf(
		"https://bedrock-runtime.%s.amazonaws.com/model/%s/converse-stream",
		p.region, modelID,
	)
}

// ─── AWS SigV4 signing ────────────────────────────────────────────────────────

func (p *Provider) signRequest(req *http.Request, payload []byte) error {
	now := time.Now().UTC()
	datestamp := now.Format("20060102")
	amzdate := now.Format("20060102T150405Z")

	req.Header.Set("X-Amz-Date", amzdate)
	if p.sessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", p.sessionToken)
	}

	// Payload hash
	payloadHash := sha256Hex(payload)

	// Canonical request
	signedHeaders := "content-type;host;x-amz-date"
	if p.sessionToken != "" {
		signedHeaders += ";x-amz-security-token"
	}

	host := req.URL.Host
	if host == "" {
		host = req.Host
	}
	req.Header.Set("Host", host)

	canonicalHeaders := fmt.Sprintf(
		"content-type:%s\nhost:%s\nx-amz-date:%s\n",
		req.Header.Get("Content-Type"), host, amzdate,
	)
	if p.sessionToken != "" {
		canonicalHeaders += fmt.Sprintf("x-amz-security-token:%s\n", p.sessionToken)
		signedHeaders += ";x-amz-security-token"
		// Rebuild without duplicate
		signedHeaders = "content-type;host;x-amz-date;x-amz-security-token"
		canonicalHeaders = fmt.Sprintf(
			"content-type:%s\nhost:%s\nx-amz-date:%s\nx-amz-security-token:%s\n",
			req.Header.Get("Content-Type"), host, amzdate, p.sessionToken,
		)
	}

	canonicalURI := req.URL.Path
	if canonicalURI == "" {
		canonicalURI = "/"
	}

	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI,
		req.URL.RawQuery,
		canonicalHeaders,
		signedHeaders,
		payloadHash,
	}, "\n")

	// Credential scope
	credentialScope := fmt.Sprintf("%s/%s/%s/aws4_request", datestamp, p.region, service)

	// String to sign
	stringToSign := strings.Join([]string{
		algorithm,
		amzdate,
		credentialScope,
		sha256Hex([]byte(canonicalRequest)),
	}, "\n")

	// Signing key
	signingKey := deriveSigningKey(p.secretKey, datestamp, p.region, service)
	signature := hex.EncodeToString(hmacSHA256(signingKey, stringToSign))

	// Authorization header
	req.Header.Set("Authorization", fmt.Sprintf(
		"%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		algorithm, p.accessKey, credentialScope, signedHeaders, signature,
	))

	return nil
}

func deriveSigningKey(secretKey, date, region, svc string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secretKey), date)
	kRegion := hmacSHA256(kDate, region)
	kService := hmacSHA256(kRegion, svc)
	return hmacSHA256(kService, "aws4_request")
}

func hmacSHA256(key []byte, data string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(data))
	return h.Sum(nil)
}

func sha256Hex(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

// ─── Error handling ───────────────────────────────────────────────────────────

type bedrockError struct {
	Message string `json:"message"`
	Type    string `json:"__type"`
}

// ProviderError is a structured error returned by the Bedrock API.
type ProviderError struct {
	StatusCode int
	Message    string
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("bedrock: %s (status=%d)", e.Message, e.StatusCode)
}

func (e *ProviderError) HTTPStatus() int { return e.StatusCode }

func (p *Provider) parseError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	var be bedrockError
	if json.Unmarshal(body, &be) == nil && be.Message != "" {
		return &ProviderError{StatusCode: resp.StatusCode, Message: be.Message}
	}

	return &ProviderError{
		StatusCode: resp.StatusCode,
		Message:    fmt.Sprintf("unexpected status %d", resp.StatusCode),
	}
}
