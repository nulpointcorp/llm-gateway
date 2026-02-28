// Package apierr provides structured API error types and HTTP status mapping
// compatible with the OpenAI error format.
package apierr

import (
	"encoding/json"

	"github.com/valyala/fasthttp"
)

// ErrorType constants.
const (
	TypeProviderError     = "provider_error"
	TypeRateLimitError    = "rate_limit_error"
	TypeInvalidRequest    = "invalid_request_error"
	TypeAuthenticationErr = "authentication_error"
	TypeServerError       = "server_error"
)

// Code constants.
const (
	CodeRateLimitExceeded = "rate_limit_exceeded"
	CodeInvalidAPIKey     = "invalid_api_key"
	CodeInternalError     = "internal_error"
	CodeProviderError     = "provider_error"
	CodeRequestTimeout    = "request_timeout"
	CodeNotImplemented    = "not_implemented"
	CodeInvalidRequest    = "invalid_request"
)

// APIError is the structured error returned to clients.
type (
	APIError struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	}
	envelope struct {
		Error APIError `json:"error"`
	}
)

// Write writes the error as JSON to the fasthttp response with the given HTTP status.
func Write(ctx *fasthttp.RequestCtx, status int, message, errType, code string) {
	ctx.SetStatusCode(status)
	ctx.SetContentType("application/json")
	body, _ := json.Marshal(envelope{Error: APIError{
		Message: message,
		Type:    errType,
		Code:    code,
	}})
	ctx.SetBody(body)
}

// WriteProviderError maps a provider HTTP status to the appropriate gateway status.
//
//	Provider 429  → 429 + Retry-After: 60
//	Provider 5xx  → 502
//	Timeout       → 504
//	Default       → 502
func WriteProviderError(ctx *fasthttp.RequestCtx, providerStatus int, msg string) {
	switch {
	case providerStatus == fasthttp.StatusTooManyRequests:
		ctx.Response.Header.Set("Retry-After", "60")
		Write(ctx, fasthttp.StatusTooManyRequests, msg, TypeRateLimitError, CodeRateLimitExceeded)
	case providerStatus >= 500 && providerStatus < 600:
		Write(ctx, fasthttp.StatusBadGateway, msg, TypeProviderError, CodeProviderError)
	default:
		Write(ctx, fasthttp.StatusBadGateway, msg, TypeProviderError, CodeProviderError)
	}
}

// WriteTimeout writes a 504 timeout error.
func WriteTimeout(ctx *fasthttp.RequestCtx) {
	Write(ctx, fasthttp.StatusGatewayTimeout, "provider request timed out", TypeProviderError, CodeRequestTimeout)
}

// WriteRateLimit writes a 429 rate limit error.
func WriteRateLimit(ctx *fasthttp.RequestCtx) {
	ctx.Response.Header.Set("Retry-After", "60")
	Write(ctx, fasthttp.StatusTooManyRequests, "rate limit exceeded", TypeRateLimitError, CodeRateLimitExceeded)
}
