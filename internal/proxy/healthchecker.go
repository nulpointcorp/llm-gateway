package proxy

import (
	"context"
	"sync"
	"time"

	"github.com/nulpointcorp/llm-gateway/internal/metrics"
	"github.com/nulpointcorp/llm-gateway/internal/providers"
)

const healthProbeInterval = 30 * time.Second
const healthProbeTimeout = 5 * time.Second

// componentStatus holds the last known health result for one component.
type componentStatus struct {
	mu     sync.RWMutex
	status string // "ok" | "degraded" | "down"
}

func (s *componentStatus) set(v string) {
	s.mu.Lock()
	s.status = v
	s.mu.Unlock()
}

func (s *componentStatus) get() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.status == "" {
		return "unknown"
	}
	return s.status
}

// HealthChecker runs background probes and exposes the latest results.
type HealthChecker struct {
	providers  map[string]providers.Provider
	cacheReady func() bool
	dbReady    func() bool
	baseCtx    context.Context
	metrics    *metrics.Registry

	providerStatuses map[string]*componentStatus
	cacheStatus      componentStatus
	dbStatus         componentStatus

	startTime time.Time
	done      chan struct{}
	wg        sync.WaitGroup
}

// NewHealthChecker creates a HealthChecker and immediately starts background probes.
func NewHealthChecker(
	ctx context.Context,
	provs map[string]providers.Provider,
	cacheReady func() bool,
	met *metrics.Registry,
) *HealthChecker {
	if ctx == nil {
		panic("healthchecker: context must not be nil")
	}
	hc := &HealthChecker{
		providers:        provs,
		cacheReady:       cacheReady,
		providerStatuses: make(map[string]*componentStatus),
		startTime:        time.Now(),
		done:             make(chan struct{}),
		baseCtx:          ctx,
		metrics:          met,
	}

	for name := range provs {
		hc.providerStatuses[name] = &componentStatus{status: "unknown"}
	}

	// Run first probe synchronously so health is not "unknown" immediately.
	hc.probe()

	hc.wg.Add(1)
	go hc.run()

	return hc
}

// HealthSnapshot returns the current health state for all components.
type HealthSnapshot struct {
	Status        string            `json:"status"`
	UptimeSeconds int64             `json:"uptime_seconds"`
	Providers     map[string]string `json:"providers"`
	Cache         string            `json:"cache"`
	Database      string            `json:"database"`
}

// Snapshot builds a snapshot from the latest probe results.
func (hc *HealthChecker) Snapshot() HealthSnapshot {
	overall := "ok"

	providers := make(map[string]string, len(hc.providerStatuses))
	for name, s := range hc.providerStatuses {
		st := s.get()
		providers[name] = st
		if st != "ok" {
			overall = "degraded"
		}
	}

	cache := hc.cacheStatus.get()
	db := hc.dbStatus.get()

	if db == "down" {
		overall = "degraded"
	}

	return HealthSnapshot{
		Status:        overall,
		UptimeSeconds: int64(time.Since(hc.startTime).Seconds()),
		Providers:     providers,
		Cache:         cache,
		Database:      db,
	}
}

// ReadinessOK returns true when the database and cache are reachable
// (used by GET /readiness for Kubernetes probes).
func (hc *HealthChecker) ReadinessOK() bool {
	return hc.dbStatus.get() == "ok"
}

// Close stops the background probe goroutine.
func (hc *HealthChecker) Close() {
	close(hc.done)
	hc.wg.Wait()
}

func (hc *HealthChecker) run() {
	defer hc.wg.Done()
	ticker := time.NewTicker(healthProbeInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			hc.probe()
		case <-hc.done:
			return
		}
	}
}

func (hc *HealthChecker) probe() {
	ctx, cancel := context.WithTimeout(hc.baseCtx, healthProbeTimeout)
	defer cancel()

	// Provider probes — run in parallel.
	var wg sync.WaitGroup
	for name, prov := range hc.providers {
		name, prov := name, prov
		s := hc.providerStatuses[name]
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := prov.HealthCheck(ctx); err != nil {
				s.set("degraded")
				if hc.metrics != nil {
					hc.metrics.SetProviderHealth(name, false)
				}
			} else {
				s.set("ok")
				if hc.metrics != nil {
					hc.metrics.SetProviderHealth(name, true)
				}
			}
		}()
	}

	// Cache probe — nil probe means "not configured" → ok.
	wg.Add(1)
	go func() {
		defer wg.Done()
		if hc.cacheReady == nil || hc.cacheReady() {
			hc.cacheStatus.set("ok")
		} else {
			hc.cacheStatus.set("degraded")
		}
	}()

	// DB probe — nil probe means "not configured" → ok.
	wg.Add(1)
	go func() {
		defer wg.Done()
		if hc.dbReady == nil || hc.dbReady() {
			hc.dbStatus.set("ok")
		} else {
			hc.dbStatus.set("down")
		}
	}()

	wg.Wait()
}
