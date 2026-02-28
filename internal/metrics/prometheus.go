// Package metrics provides a Prometheus metrics registry for the gateway.
//
// All metrics are scoped to a private registry (not the global default) so
// they don't interfere with host-level metrics when embedded in other
// applications. The /metrics HTTP handler is exposed via Handler().
package metrics

import (
	"strconv"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttpadaptor"
)

// Registry holds all exported metrics.
type Registry struct {
	reg *prometheus.Registry

	// gateway_inflight_requests
	inFlight prometheus.Gauge

	// gateway_http_requests_total{route,status}
	httpRequestsTotal *prometheus.CounterVec

	// gateway_http_request_duration_seconds{route}
	httpDuration *prometheus.HistogramVec

	// gateway_http_request_size_bytes{route}
	httpReqSize *prometheus.HistogramVec

	// gateway_http_response_size_bytes{route,status}
	httpRespSize *prometheus.HistogramVec

	// gateway_requests_total{provider, status}
	requestsTotal *prometheus.CounterVec

	// gateway_latency_ms_total{provider} — sum of latency in ms (derive avg externally)
	latencyTotal *prometheus.CounterVec

	// gateway_request_duration_seconds{provider,route,cache}
	requestDuration *prometheus.HistogramVec

	// gateway_upstream_attempts_total{provider,route,outcome}
	upstreamAttempts *prometheus.CounterVec

	// gateway_upstream_attempt_duration_seconds{provider,route,outcome}
	upstreamDuration *prometheus.HistogramVec

	// cache_hits_total / cache_misses_total
	cacheHits   prometheus.Counter
	cacheMisses prometheus.Counter

	// gateway_cache_operations_total{op,result}
	cacheOps *prometheus.CounterVec

	// provider_errors_total{provider, error_type}
	providerErrors *prometheus.CounterVec

	// circuit_breaker_state{provider} — 0=closed, 1=open, 2=half-open
	circuitBreakerState *prometheus.GaugeVec

	// gateway_circuit_breaker_transitions_total{provider,to_state}
	cbTransitions *prometheus.CounterVec

	// gateway_circuit_breaker_rejections_total{provider,state}
	cbRejections *prometheus.CounterVec

	// gateway_failover_events_total{primary,from,to,reason}
	failoverEvents *prometheus.CounterVec

	// gateway_failover_success_total{primary,to}
	failoverSuccess *prometheus.CounterVec

	// gateway_failover_exhausted_total{primary}
	failoverExhausted *prometheus.CounterVec

	// gateway_ratelimit_total{result}
	rateLimitTotal *prometheus.CounterVec

	// gateway_tokens_total{provider,route,direction,cache}
	tokensTotal *prometheus.CounterVec

	// gateway_provider_health{provider}
	providerHealth *prometheus.GaugeVec

	// gateway_build_info{version}
	buildInfo *prometheus.GaugeVec

	cbMu        sync.Mutex
	lastCBState map[string]float64

	metricsHandler fasthttp.RequestHandler
}

func New() *Registry {
	reg := prometheus.NewRegistry()

	// Baseline runtime metrics even with a private registry.
	reg.MustRegister(prometheus.NewGoCollector())
	reg.MustRegister(prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))

	r := &Registry{
		reg: reg,
		lastCBState: make(map[string]float64),

		inFlight: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "gateway_inflight_requests",
			Help: "Current number of in-flight HTTP requests handled by the gateway",
		}),

		httpRequestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_http_requests_total",
				Help: "Total number of HTTP requests handled by the gateway",
			},
			[]string{"route", "status"},
		),

		httpDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "gateway_http_request_duration_seconds",
				Help:    "HTTP request duration in seconds (end-to-end, includes cache + upstream)",
				Buckets: []float64{0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60},
			},
			[]string{"route"},
		),

		httpReqSize: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "gateway_http_request_size_bytes",
				Help:    "HTTP request body size in bytes",
				Buckets: prometheus.ExponentialBuckets(256, 2, 12), // 256B .. ~512KB
			},
			[]string{"route"},
		),

		httpRespSize: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "gateway_http_response_size_bytes",
				Help:    "HTTP response body size in bytes",
				Buckets: prometheus.ExponentialBuckets(256, 2, 14), // 256B .. ~2MB
			},
			[]string{"route", "status"},
		),

		requestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_requests_total",
				Help: "Total number of proxy requests",
			},
			[]string{"provider", "status"},
		),

		latencyTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_latency_ms_total",
				Help: "Sum of latency in ms (compute avg externally)",
			},
			[]string{"provider"},
		),

		requestDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "gateway_request_duration_seconds",
				Help:    "End-to-end request duration (gateway perspective) in seconds",
				Buckets: []float64{0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60},
			},
			[]string{"provider", "route", "cache"},
		),

		upstreamAttempts: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_upstream_attempts_total",
				Help: "Total upstream provider attempts (includes failovers)",
			},
			[]string{"provider", "route", "outcome"},
		),

		upstreamDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "gateway_upstream_attempt_duration_seconds",
				Help:    "Upstream provider attempt duration in seconds",
				Buckets: []float64{0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60},
			},
			[]string{"provider", "route", "outcome"},
		),

		cacheHits: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "cache_hits_total",
			Help: "Total cache hits",
		}),

		cacheMisses: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "cache_misses_total",
			Help: "Total cache misses",
		}),

		cacheOps: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_cache_operations_total",
				Help: "Cache operations by type and result",
			},
			[]string{"op", "result"},
		),

		providerErrors: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "provider_errors_total",
				Help: "Total provider errors by type",
			},
			[]string{"provider", "error_type"},
		),

		circuitBreakerState: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "circuit_breaker_state",
				Help: "Circuit breaker state (0=closed,1=open,2=half-open)",
			},
			[]string{"provider"},
		),

		cbTransitions: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_circuit_breaker_transitions_total",
				Help: "Circuit breaker transitions to a new state",
			},
			[]string{"provider", "to_state"},
		),

		cbRejections: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_circuit_breaker_rejections_total",
				Help: "Requests rejected due to circuit breaker state",
			},
			[]string{"provider", "state"},
		),

		failoverEvents: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_failover_events_total",
				Help: "Failover events between providers (emitted when switching to a different provider)",
			},
			[]string{"primary", "from", "to", "reason"},
		),

		failoverSuccess: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_failover_success_total",
				Help: "Successful failovers (request served by non-primary provider)",
			},
			[]string{"primary", "to"},
		),

		failoverExhausted: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_failover_exhausted_total",
				Help: "Requests that exhausted failover attempts without success",
			},
			[]string{"primary"},
		),

		rateLimitTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_ratelimit_total",
				Help: "Rate limit decisions",
			},
			[]string{"result"},
		),

		tokensTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_tokens_total",
				Help: "Token usage totals derived from upstream usage fields",
			},
			[]string{"provider", "route", "direction", "cache"},
		),

		providerHealth: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "gateway_provider_health",
				Help: "Provider health status (1=ok, 0=degraded)",
			},
			[]string{"provider"},
		),

		buildInfo: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "gateway_build_info",
				Help: "Build information",
			},
			[]string{"version"},
		),
	}

	reg.MustRegister(
		r.inFlight,
		r.httpRequestsTotal,
		r.httpDuration,
		r.httpReqSize,
		r.httpRespSize,
		r.requestsTotal,
		r.latencyTotal,
		r.requestDuration,
		r.upstreamAttempts,
		r.upstreamDuration,
		r.cacheHits,
		r.cacheMisses,
		r.cacheOps,
		r.providerErrors,
		r.circuitBreakerState,
		r.cbTransitions,
		r.cbRejections,
		r.failoverEvents,
		r.failoverSuccess,
		r.failoverExhausted,
		r.rateLimitTotal,
		r.tokensTotal,
		r.providerHealth,
		r.buildInfo,
	)

	h := promhttp.HandlerFor(reg, promhttp.HandlerOpts{})
	r.metricsHandler = fasthttpadaptor.NewFastHTTPHandler(h)

	return r
}

func (r *Registry) RecordRequest(provider string, statusCode int, latencyMs int64) {
	r.requestsTotal.WithLabelValues(provider, strconv.Itoa(statusCode)).Inc()
	r.latencyTotal.WithLabelValues(provider).Add(float64(latencyMs))
}

func (r *Registry) IncInFlight() { r.inFlight.Inc() }
func (r *Registry) DecInFlight() { r.inFlight.Dec() }

// ObserveHTTP records end-to-end HTTP metrics.
func (r *Registry) ObserveHTTP(route string, statusCode int, dur time.Duration, reqBytes, respBytes int) {
	status := strconv.Itoa(statusCode)
	r.httpRequestsTotal.WithLabelValues(route, status).Inc()
	r.httpDuration.WithLabelValues(route).Observe(dur.Seconds())
	if reqBytes >= 0 {
		r.httpReqSize.WithLabelValues(route).Observe(float64(reqBytes))
	}
	if respBytes >= 0 {
		r.httpRespSize.WithLabelValues(route, status).Observe(float64(respBytes))
	}
}

// ObserveGatewayRequest records per-provider request latency and cache status.
func (r *Registry) ObserveGatewayRequest(provider, route, cache string, dur time.Duration) {
	r.requestDuration.WithLabelValues(provider, route, cache).Observe(dur.Seconds())
}

// ObserveUpstreamAttempt records one upstream provider attempt.
func (r *Registry) ObserveUpstreamAttempt(provider, route, outcome string, dur time.Duration) {
	r.upstreamAttempts.WithLabelValues(provider, route, outcome).Inc()
	r.upstreamDuration.WithLabelValues(provider, route, outcome).Observe(dur.Seconds())
}

func (r *Registry) RecordFailover(primary, from, to, reason string) {
	r.failoverEvents.WithLabelValues(primary, from, to, reason).Inc()
}

func (r *Registry) RecordFailoverSuccess(primary, to string) {
	r.failoverSuccess.WithLabelValues(primary, to).Inc()
}

func (r *Registry) RecordFailoverExhausted(primary string) {
	r.failoverExhausted.WithLabelValues(primary).Inc()
}

func (r *Registry) RecordRateLimit(result string) {
	r.rateLimitTotal.WithLabelValues(result).Inc()
}

func (r *Registry) CacheGetHit() {
	r.cacheHits.Inc()
	r.cacheOps.WithLabelValues("get", "hit").Inc()
}

func (r *Registry) CacheGetMiss() {
	r.cacheMisses.Inc()
	r.cacheOps.WithLabelValues("get", "miss").Inc()
}

func (r *Registry) CacheGetBypass() {
	r.cacheOps.WithLabelValues("get", "bypass").Inc()
}

func (r *Registry) CacheSetOK() {
	r.cacheOps.WithLabelValues("set", "ok").Inc()
}

func (r *Registry) CacheSetError() {
	r.cacheOps.WithLabelValues("set", "error").Inc()
}

func (r *Registry) AddTokens(provider, route string, inputTokens, outputTokens int, cached bool) {
	cache := "miss"
	if cached {
		cache = "hit"
	}
	if inputTokens > 0 {
		r.tokensTotal.WithLabelValues(provider, route, "input", cache).Add(float64(inputTokens))
	}
	if outputTokens > 0 {
		r.tokensTotal.WithLabelValues(provider, route, "output", cache).Add(float64(outputTokens))
	}
	if inputTokens+outputTokens > 0 {
		r.tokensTotal.WithLabelValues(provider, route, "total", cache).Add(float64(inputTokens + outputTokens))
	}
}

func (r *Registry) SetProviderHealth(provider string, ok bool) {
	if ok {
		r.providerHealth.WithLabelValues(provider).Set(1)
		return
	}
	r.providerHealth.WithLabelValues(provider).Set(0)
}

func (r *Registry) SetBuildInfo(version string) {
	// Gauge is used so the time series always exists.
	r.buildInfo.WithLabelValues(version).Set(1)
}

func (r *Registry) RecordError(provider, errType string) {
	r.providerErrors.WithLabelValues(provider, errType).Inc()
}

// SetCircuitBreaker sets the circuit breaker state gauge and increments a
// transition counter when the state changes.
func (r *Registry) SetCircuitBreaker(provider string, state int64) {
	r.circuitBreakerState.WithLabelValues(provider).Set(float64(state))

	r.cbMu.Lock()
	prev, ok := r.lastCBState[provider]
	if !ok || prev != float64(state) {
		r.lastCBState[provider] = float64(state)
		toState := strconv.FormatInt(state, 10)
		r.cbTransitions.WithLabelValues(provider, toState).Inc()
	}
	r.cbMu.Unlock()
}

func (r *Registry) RecordCircuitBreakerRejection(provider, state string) {
	r.cbRejections.WithLabelValues(provider, state).Inc()
}
func (r *Registry) Handler() fasthttp.RequestHandler {
	return r.metricsHandler
}
func (r *Registry) PromRegistry() *prometheus.Registry { return r.reg }
