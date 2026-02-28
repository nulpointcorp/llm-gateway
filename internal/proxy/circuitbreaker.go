package proxy

import (
	"sync"
	"time"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

// cbState represents the operational state of a per-provider circuit breaker.
//
//	cbClosed   — normal operation; all requests pass through.
//	cbOpen     — provider is failing; requests are rejected immediately.
//	cbHalfOpen — recovery probe; one request is allowed to mock the provider.
type cbState int

const (
	cbClosed   cbState = 0
	cbOpen     cbState = 1
	cbHalfOpen cbState = 2
)

// CBConfig holds circuit breaker tuning parameters. Zero values fall back to
// the package-level defaults defined in providers/provider.go.
type CBConfig struct {
	// ErrorThreshold is the number of failures within TimeWindow that trips
	// the breaker. Default: providers.CBErrorThreshold (5).
	ErrorThreshold int

	// TimeWindow is the rolling window for counting errors.
	// Default: providers.CBTimeWindow (60s).
	TimeWindow time.Duration

	// HalfOpenTimeout is how long the breaker stays open before allowing a
	// single probe request. Default: providers.CBHalfOpenTimeout (30s).
	HalfOpenTimeout time.Duration
}

func (c *CBConfig) errorThreshold() int {
	if c.ErrorThreshold > 0 {
		return c.ErrorThreshold
	}
	return providers.CBErrorThreshold
}

func (c *CBConfig) timeWindow() time.Duration {
	if c.TimeWindow > 0 {
		return c.TimeWindow
	}
	return providers.CBTimeWindow
}

func (c *CBConfig) halfOpenTimeout() time.Duration {
	if c.HalfOpenTimeout > 0 {
		return c.HalfOpenTimeout
	}
	return providers.CBHalfOpenTimeout
}

// providerCB holds per-provider circuit breaker state.
type providerCB struct {
	mu sync.Mutex

	state         cbState
	errorCount    int
	windowStart   time.Time // start of the current error-counting window
	openedAt      time.Time // when the breaker was tripped (for half-open timer)
	probeInflight bool      // true while a half-open probe is in flight
}

// CircuitBreaker manages independent circuit breakers for each LLM provider.
// It is safe for concurrent use from multiple goroutines.
type CircuitBreaker struct {
	mu       sync.RWMutex
	breakers map[string]*providerCB
	cfg      CBConfig
}

// NewCircuitBreaker creates a CircuitBreaker with default settings for every
// provider in providers.DefaultFallbackOrder.
func NewCircuitBreaker() *CircuitBreaker {
	return NewCircuitBreakerWithConfig(CBConfig{})
}

// NewCircuitBreakerWithConfig creates a CircuitBreaker with custom thresholds.
// Use this to apply values loaded from configuration.
func NewCircuitBreakerWithConfig(cfg CBConfig) *CircuitBreaker {
	cb := &CircuitBreaker{
		breakers: make(map[string]*providerCB),
		cfg:      cfg,
	}
	for _, name := range providers.DefaultFallbackOrder {
		cb.breakers[name] = &providerCB{
			state:       cbClosed,
			windowStart: time.Now(),
		}
	}
	return cb
}

// Allow reports whether the named provider should receive the next request.
//
//   - Closed  → always true.
//   - Open    → false, unless the half-open timeout has elapsed, in which case
//     the breaker transitions to HalfOpen and allows one probe.
//   - HalfOpen → true only if no probe is currently in flight.
//
// Returns true for unknown providers (the breaker is not tracking them yet).
func (cb *CircuitBreaker) Allow(provider string) bool {
	pcb := cb.get(provider)
	if pcb == nil {
		return true // unknown provider — optimistic allow
	}

	pcb.mu.Lock()
	defer pcb.mu.Unlock()

	switch pcb.state {
	case cbClosed:
		return true

	case cbOpen:
		if time.Since(pcb.openedAt) >= cb.cfg.halfOpenTimeout() {
			// Transition to half-open: allow exactly one probe request.
			pcb.state = cbHalfOpen
			pcb.probeInflight = true
			return true
		}
		return false

	case cbHalfOpen:
		if pcb.probeInflight {
			// A probe is already in flight — reject other requests.
			return false
		}
		pcb.probeInflight = true
		return true
	}

	return true
}

// RecordSuccess marks a successful response for provider and resets the
// breaker to Closed regardless of its previous state.
func (cb *CircuitBreaker) RecordSuccess(provider string) {
	pcb := cb.get(provider)
	if pcb == nil {
		return
	}

	pcb.mu.Lock()
	defer pcb.mu.Unlock()

	pcb.state = cbClosed
	pcb.errorCount = 0
	pcb.probeInflight = false
	pcb.windowStart = time.Now()
}

// RecordFailure increments the error counter for provider. When the counter
// reaches ErrorThreshold within TimeWindow the breaker opens.
func (cb *CircuitBreaker) RecordFailure(provider string) {
	pcb := cb.get(provider)
	if pcb == nil {
		return
	}

	pcb.mu.Lock()
	defer pcb.mu.Unlock()

	now := time.Now()

	// Reset counter when the rolling window has expired.
	if now.Sub(pcb.windowStart) > cb.cfg.timeWindow() {
		pcb.errorCount = 0
		pcb.windowStart = now
	}

	pcb.errorCount++
	pcb.probeInflight = false

	if pcb.errorCount >= cb.cfg.errorThreshold() {
		pcb.state = cbOpen
		pcb.openedAt = now
	}
}

// State returns the current cbState for provider (useful for metrics export).
func (cb *CircuitBreaker) State(provider string) cbState {
	pcb := cb.get(provider)
	if pcb == nil {
		return cbClosed
	}
	pcb.mu.Lock()
	defer pcb.mu.Unlock()
	return pcb.state
}

// StateLabel returns a human-readable state name: "closed", "open", or "half_open".
func (cb *CircuitBreaker) StateLabel(provider string) string {
	switch cb.State(provider) {
	case cbOpen:
		return "open"
	case cbHalfOpen:
		return "half_open"
	default:
		return "closed"
	}
}

func (cb *CircuitBreaker) get(provider string) *providerCB {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.breakers[provider]
}
