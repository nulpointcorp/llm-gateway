package proxy

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

// failoverEvent records one failover attempt for observability.
type failoverEvent struct {
	From      string
	To        string
	Reason    string
	LatencyMs int64
}

// requestWithFailover tries the primary provider and, on retryable errors,
// walks through providers.DefaultFallbackOrder until one succeeds or
// g.maxRetries is exhausted.
//
// It skips providers whose circuit breaker is in the Open state.
// Returns the successful response, the name of the provider that served it,
// and nil — or nil, "", and an error if every candidate fails.
func (g *Gateway) requestWithFailover(
	ctx context.Context,
	req *providers.ProxyRequest,
	primary string,
	route string,
) (*providers.ProxyResponse, string, error) {

	candidates := buildCandidateList(primary)

	var lastErr error

	prevProvider := ""
	prevReason := ""
	havePrevFailure := false
	attempts := 0

	for _, name := range candidates {
		if attempts >= g.maxRetries {
			break
		}

		prov, ok := g.providers[name]
		if !ok {
			continue // provider not configured, skip
		}

		// Skip providers whose circuit breaker is open.
		if g.cb != nil && !g.cb.Allow(name) {
			g.log.WarnContext(ctx, "circuit_breaker_open",
				slog.String("request_id", req.RequestID),
				slog.String("provider", name),
			)
			if g.metrics != nil {
				g.metrics.RecordCircuitBreakerRejection(name, g.cb.StateLabel(name))
				g.metrics.SetCircuitBreaker(name, int64(g.cb.State(name)))
				g.metrics.ObserveUpstreamAttempt(name, route, "circuit_reject", 0)
			}
			continue
		}

		// We are switching to a different provider after a failure.
		if havePrevFailure && prevProvider != "" && prevProvider != name {
			if g.metrics != nil {
				g.metrics.RecordFailover(primary, prevProvider, name, prevReason)
			}
		}

		start := time.Now()
		resp, err := prov.Request(ctx, req)
		dur := time.Since(start)
		latencyMs := dur.Milliseconds()
		attempts++

		if err == nil {
			if g.metrics != nil {
				g.metrics.ObserveUpstreamAttempt(name, route, "success", dur)
			}
			// ── Success ───────────────────────────────────────────────────────
			if g.cb != nil {
				g.cb.RecordSuccess(name)
				if g.metrics != nil {
					g.metrics.SetCircuitBreaker(name, int64(g.cb.State(name)))
				}
			}
			if name != primary {
				g.log.InfoContext(ctx, "failover_success",
					slog.String("request_id", req.RequestID),
					slog.String("from", primary),
					slog.String("to", name),
					slog.Int64("latency_ms", latencyMs),
				)
				if g.metrics != nil {
					g.metrics.RecordFailoverSuccess(primary, name)
				}
			}
			return resp, name, nil
		}

		// ── Failure ───────────────────────────────────────────────────────────
		if g.cb != nil {
			g.cb.RecordFailure(name)
			if g.metrics != nil {
				g.metrics.SetCircuitBreaker(name, int64(g.cb.State(name)))
			}
		}

		reason := classifyError(err)
		if g.metrics != nil {
			g.metrics.ObserveUpstreamAttempt(name, route, reason, dur)
			g.metrics.RecordError(name, reason)
		}
		g.log.WarnContext(ctx, "provider_attempt_failed",
			slog.String("request_id", req.RequestID),
			slog.String("from", primary),
			slog.String("to", name),
			slog.String("reason", reason),
			slog.Int64("latency_ms", latencyMs),
			slog.String("error", err.Error()),
		)

		lastErr = err
		prevProvider = name
		prevReason = reason
		havePrevFailure = true

		// Non-retryable errors (4xx) abort failover immediately — further
		// providers are unlikely to return a different result for the same
		// request parameters.
		if !isRetryable(err) {
			break
		}
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("no providers available")
	}
	if g.metrics != nil {
		g.metrics.RecordFailoverExhausted(primary)
	}
	return nil, "", fmt.Errorf("failover: all providers failed after %d attempt(s): %w", attempts, lastErr)
}

// buildCandidateList returns an ordered slice starting with primary, followed
// by the remaining providers in DefaultFallbackOrder (deduped).
func buildCandidateList(primary string) []string {
	seen := map[string]bool{primary: true}
	out := []string{primary}
	for _, name := range providers.DefaultFallbackOrder {
		if !seen[name] {
			seen[name] = true
			out = append(out, name)
		}
	}
	return out
}

// isRetryable returns true for errors that should trigger provider failover.
//
//   - 5xx provider errors → retryable (infrastructure failure)
//   - context.DeadlineExceeded → retryable (timeout, different provider may be faster)
//   - 4xx provider errors → NOT retryable (bad request / auth — won't change)
//   - unknown errors → retryable (conservative default)
func isRetryable(err error) bool {
	if err == context.DeadlineExceeded {
		return true
	}
	if sc, ok := err.(providers.StatusCoder); ok {
		status := sc.HTTPStatus()
		return status >= 500 && status < 600
	}
	return true // unknown errors are treated as retryable
}

// classifyError converts an error into a short human-readable category string
// used in log fields and metrics labels.
func classifyError(err error) string {
	if err == context.DeadlineExceeded {
		return "timeout"
	}
	if sc, ok := err.(providers.StatusCoder); ok {
		return fmt.Sprintf("http_%d", sc.HTTPStatus())
	}
	return "unknown"
}
