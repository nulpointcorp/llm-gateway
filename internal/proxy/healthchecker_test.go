package proxy

import (
	"context"
	"fmt"
	"testing"

	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

// --- healthyProvider / failingHealthProvider ---------------------------------

type healthyProvider struct{ name string }

func (p *healthyProvider) Name() string { return p.name }
func (p *healthyProvider) Request(_ context.Context, _ *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	return nil, nil
}
func (p *healthyProvider) HealthCheck(_ context.Context) error { return nil }

type failingHealthProvider struct{ name string }

func (p *failingHealthProvider) Name() string { return p.name }
func (p *failingHealthProvider) Request(_ context.Context, _ *providers.ProxyRequest) (*providers.ProxyResponse, error) {
	return nil, nil
}
func (p *failingHealthProvider) HealthCheck(_ context.Context) error {
	return fmt.Errorf("health check failed")
}

// --- NewHealthChecker -------------------------------------------------------

func TestNewHealthChecker_PanicsOnNilContext(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for nil context")
		}
	}()
	NewHealthChecker(nil, nil, nil, nil)
}

func TestNewHealthChecker_RunsInitialProbe(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai": &healthyProvider{name: "openai"},
	}
	hc := NewHealthChecker(context.Background(), provs, nil, nil)
	defer hc.Close()

	snap := hc.Snapshot()
	if snap.Providers["openai"] != "ok" {
		t.Errorf("expected openai=ok after initial probe, got %s", snap.Providers["openai"])
	}
}

// --- Snapshot ---------------------------------------------------------------

func TestSnapshot_AllHealthy(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai":    &healthyProvider{name: "openai"},
		"anthropic": &healthyProvider{name: "anthropic"},
	}
	hc := NewHealthChecker(context.Background(), provs, func() bool { return true }, nil)
	defer hc.Close()

	snap := hc.Snapshot()
	if snap.Status != "ok" {
		t.Errorf("expected status=ok, got %s", snap.Status)
	}
	if snap.Cache != "ok" {
		t.Errorf("expected cache=ok, got %s", snap.Cache)
	}
	if snap.UptimeSeconds < 0 {
		t.Error("uptime should be non-negative")
	}
}

func TestSnapshot_DegradedProvider(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai":    &healthyProvider{name: "openai"},
		"anthropic": &failingHealthProvider{name: "anthropic"},
	}
	hc := NewHealthChecker(context.Background(), provs, nil, nil)
	defer hc.Close()

	snap := hc.Snapshot()
	if snap.Status != "degraded" {
		t.Errorf("expected status=degraded when a provider is down, got %s", snap.Status)
	}
	if snap.Providers["openai"] != "ok" {
		t.Errorf("openai should be ok, got %s", snap.Providers["openai"])
	}
	if snap.Providers["anthropic"] != "degraded" {
		t.Errorf("anthropic should be degraded, got %s", snap.Providers["anthropic"])
	}
}

func TestSnapshot_CacheDegraded(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai": &healthyProvider{name: "openai"},
	}
	hc := NewHealthChecker(context.Background(), provs, func() bool { return false }, nil)
	defer hc.Close()

	snap := hc.Snapshot()
	if snap.Cache != "degraded" {
		t.Errorf("expected cache=degraded, got %s", snap.Cache)
	}
}

func TestSnapshot_NilCacheProbe(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai": &healthyProvider{name: "openai"},
	}
	hc := NewHealthChecker(context.Background(), provs, nil, nil)
	defer hc.Close()

	snap := hc.Snapshot()
	// Nil cache probe means "not configured" → ok.
	if snap.Cache != "ok" {
		t.Errorf("expected cache=ok when probe is nil, got %s", snap.Cache)
	}
}

func TestSnapshot_DBDown(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai": &healthyProvider{name: "openai"},
	}
	hc := NewHealthChecker(context.Background(), provs, nil, nil)
	defer hc.Close()

	// Manually set DB to down.
	hc.dbStatus.set("down")

	snap := hc.Snapshot()
	if snap.Database != "down" {
		t.Errorf("expected database=down, got %s", snap.Database)
	}
	if snap.Status != "degraded" {
		t.Errorf("expected overall=degraded when DB is down, got %s", snap.Status)
	}
}

// --- ReadinessOK ------------------------------------------------------------

func TestReadinessOK_DBUp(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai": &healthyProvider{name: "openai"},
	}
	hc := NewHealthChecker(context.Background(), provs, nil, nil)
	defer hc.Close()

	// DB probe is nil → defaults to "ok".
	if !hc.ReadinessOK() {
		t.Error("readiness should be OK when DB is up")
	}
}

func TestReadinessOK_DBDown(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai": &healthyProvider{name: "openai"},
	}
	hc := NewHealthChecker(context.Background(), provs, nil, nil)
	defer hc.Close()

	hc.dbStatus.set("down")

	if hc.ReadinessOK() {
		t.Error("readiness should NOT be OK when DB is down")
	}
}

// --- componentStatus --------------------------------------------------------

func TestComponentStatus_DefaultUnknown(t *testing.T) {
	var cs componentStatus
	if cs.get() != "unknown" {
		t.Errorf("expected 'unknown' default, got %q", cs.get())
	}
}

func TestComponentStatus_SetGet(t *testing.T) {
	var cs componentStatus
	cs.set("ok")
	if cs.get() != "ok" {
		t.Errorf("expected 'ok', got %q", cs.get())
	}
	cs.set("degraded")
	if cs.get() != "degraded" {
		t.Errorf("expected 'degraded', got %q", cs.get())
	}
}

// --- Close ------------------------------------------------------------------

func TestHealthChecker_Close(t *testing.T) {
	provs := map[string]providers.Provider{
		"openai": &healthyProvider{name: "openai"},
	}
	hc := NewHealthChecker(context.Background(), provs, nil, nil)

	// Close should not hang.
	hc.Close()
}
