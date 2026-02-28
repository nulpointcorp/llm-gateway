package proxy

import (
	"testing"
	"time"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

func TestCircuitBreaker_InitialState(t *testing.T) {
	cb := NewCircuitBreaker()

	for _, name := range providers.DefaultFallbackOrder {
		if cb.State(name) != cbClosed {
			t.Errorf("provider %s should start closed, got %v", name, cb.State(name))
		}
		if cb.StateLabel(name) != "closed" {
			t.Errorf("provider %s label should be 'closed', got %s", name, cb.StateLabel(name))
		}
	}
}

func TestCircuitBreaker_AllowClosedState(t *testing.T) {
	cb := NewCircuitBreaker()
	if !cb.Allow("openai") {
		t.Error("closed breaker should allow requests")
	}
}

func TestCircuitBreaker_AllowUnknownProvider(t *testing.T) {
	cb := NewCircuitBreaker()
	if !cb.Allow("unknown-provider") {
		t.Error("unknown provider should be allowed")
	}
}

func TestCircuitBreaker_OpensAfterThreshold(t *testing.T) {
	cb := NewCircuitBreaker()

	for i := 0; i < providers.CBErrorThreshold-1; i++ {
		cb.RecordFailure("openai")
		if cb.State("openai") != cbClosed {
			t.Fatalf("should remain closed before threshold, iteration %d", i)
		}
	}

	// One more failure should trip it.
	cb.RecordFailure("openai")
	if cb.State("openai") != cbOpen {
		t.Error("should be open after reaching threshold")
	}
	if cb.StateLabel("openai") != "open" {
		t.Errorf("label should be 'open', got %s", cb.StateLabel("openai"))
	}
}

func TestCircuitBreaker_OpenRejectsRequests(t *testing.T) {
	cb := NewCircuitBreaker()

	for i := 0; i < providers.CBErrorThreshold; i++ {
		cb.RecordFailure("openai")
	}

	if cb.Allow("openai") {
		t.Error("open breaker should reject requests")
	}
}

func TestCircuitBreaker_SuccessResets(t *testing.T) {
	cb := NewCircuitBreaker()

	// Accumulate some failures (but not enough to trip).
	for i := 0; i < providers.CBErrorThreshold-1; i++ {
		cb.RecordFailure("openai")
	}

	cb.RecordSuccess("openai")

	if cb.State("openai") != cbClosed {
		t.Error("success should reset to closed")
	}

	// Should need full threshold again.
	for i := 0; i < providers.CBErrorThreshold-1; i++ {
		cb.RecordFailure("openai")
	}
	if cb.State("openai") != cbClosed {
		t.Error("should still be closed before new threshold")
	}
}

func TestCircuitBreaker_WindowReset(t *testing.T) {
	cb := NewCircuitBreaker()

	// Manually set the window start to the past so failures are outside window.
	pcb := cb.breakers["openai"]
	pcb.mu.Lock()
	pcb.windowStart = time.Now().Add(-providers.CBTimeWindow - time.Second)
	pcb.errorCount = providers.CBErrorThreshold - 1
	pcb.mu.Unlock()

	// This failure should reset the counter because the window expired.
	cb.RecordFailure("openai")

	if cb.State("openai") != cbClosed {
		t.Error("error counter should reset after window expires; breaker should stay closed")
	}
}

func TestCircuitBreaker_HalfOpenAfterTimeout(t *testing.T) {
	cb := NewCircuitBreaker()

	// Trip the breaker.
	for i := 0; i < providers.CBErrorThreshold; i++ {
		cb.RecordFailure("openai")
	}
	if cb.State("openai") != cbOpen {
		t.Fatal("expected open")
	}

	// Simulate time passing past half-open timeout.
	pcb := cb.breakers["openai"]
	pcb.mu.Lock()
	pcb.openedAt = time.Now().Add(-providers.CBHalfOpenTimeout - time.Second)
	pcb.mu.Unlock()

	// Allow should transition to half-open and permit one probe.
	if !cb.Allow("openai") {
		t.Error("should allow one probe in half-open state")
	}
	if cb.State("openai") != cbHalfOpen {
		t.Errorf("expected half_open, got %s", cb.StateLabel("openai"))
	}

	// Second request in half-open should be rejected (probe already in flight).
	if cb.Allow("openai") {
		t.Error("should reject second request while probe is in flight")
	}
}

func TestCircuitBreaker_HalfOpenSuccessCloses(t *testing.T) {
	cb := NewCircuitBreaker()

	// Trip + fast-forward to half-open.
	for i := 0; i < providers.CBErrorThreshold; i++ {
		cb.RecordFailure("openai")
	}
	pcb := cb.breakers["openai"]
	pcb.mu.Lock()
	pcb.openedAt = time.Now().Add(-providers.CBHalfOpenTimeout - time.Second)
	pcb.mu.Unlock()

	cb.Allow("openai") // transitions to half-open
	cb.RecordSuccess("openai")

	if cb.State("openai") != cbClosed {
		t.Error("success in half-open should close the breaker")
	}
	if !cb.Allow("openai") {
		t.Error("should allow requests after closing from half-open")
	}
}

func TestCircuitBreaker_HalfOpenFailureReopens(t *testing.T) {
	cb := NewCircuitBreaker()

	// Trip + fast-forward to half-open.
	for i := 0; i < providers.CBErrorThreshold; i++ {
		cb.RecordFailure("openai")
	}
	pcb := cb.breakers["openai"]
	pcb.mu.Lock()
	pcb.openedAt = time.Now().Add(-providers.CBHalfOpenTimeout - time.Second)
	pcb.mu.Unlock()

	cb.Allow("openai") // transitions to half-open

	// Probe fails — should reopen.
	cb.RecordFailure("openai")

	if cb.State("openai") != cbOpen {
		t.Error("failure in half-open should reopen the breaker")
	}
}

func TestCircuitBreaker_IndependentProviders(t *testing.T) {
	cb := NewCircuitBreaker()

	// Trip openai.
	for i := 0; i < providers.CBErrorThreshold; i++ {
		cb.RecordFailure("openai")
	}

	if cb.State("openai") != cbOpen {
		t.Error("openai should be open")
	}
	if cb.State("anthropic") != cbClosed {
		t.Error("anthropic should remain closed")
	}
	if !cb.Allow("anthropic") {
		t.Error("anthropic should still allow requests")
	}
}

func TestCircuitBreaker_RecordOnUnknownProvider(t *testing.T) {
	cb := NewCircuitBreaker()
	// Should not panic.
	cb.RecordSuccess("nonexistent")
	cb.RecordFailure("nonexistent")
	if cb.State("nonexistent") != cbClosed {
		t.Error("unknown provider state should default to closed")
	}
}

func TestCircuitBreaker_StateLabel(t *testing.T) {
	cb := NewCircuitBreaker()

	if cb.StateLabel("openai") != "closed" {
		t.Errorf("expected 'closed', got %s", cb.StateLabel("openai"))
	}

	// Trip it.
	for i := 0; i < providers.CBErrorThreshold; i++ {
		cb.RecordFailure("openai")
	}
	if cb.StateLabel("openai") != "open" {
		t.Errorf("expected 'open', got %s", cb.StateLabel("openai"))
	}

	// Fast-forward to half-open.
	pcb := cb.breakers["openai"]
	pcb.mu.Lock()
	pcb.openedAt = time.Now().Add(-providers.CBHalfOpenTimeout - time.Second)
	pcb.mu.Unlock()
	cb.Allow("openai")
	if cb.StateLabel("openai") != "half_open" {
		t.Errorf("expected 'half_open', got %s", cb.StateLabel("openai"))
	}
}
