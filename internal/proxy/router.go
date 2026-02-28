package proxy

import (
	"encoding/json"
	"time"

	"github.com/fasthttp/router"
	"github.com/valyala/fasthttp"
)

// RouteHandler is a fasthttp handler function.
type RouteHandler = fasthttp.RequestHandler

// ManagementRoutes holds optional management API handler functions
// that are registered alongside the proxy routes.
type ManagementRoutes struct {
	Metrics RouteHandler
}

// Start starts the HTTP server on addr (e.g. ":8080").
// Pass nil for routes to start in proxy-only mode.
func (g *Gateway) Start(addr string) error {
	return g.StartWithRoutes(addr, nil)
}

// StartWithRoutes starts the HTTP server with optional management routes.
func (g *Gateway) StartWithRoutes(addr string, mgmt *ManagementRoutes) error {
	r := router.New()

	r.POST("/v1/chat/completions", g.handleChatCompletions)
	r.POST("/v1/completions", g.handleCompletions)
	r.POST("/v1/embeddings", g.handleEmbeddings)
	r.GET("/health", g.handleHealth)
	r.GET("/readiness", g.handleReadiness)

	if mgmt != nil && mgmt.Metrics != nil {
		r.GET("/metrics", mgmt.Metrics)
	}

	handler := applyMiddleware(r.Handler,
		recovery,
		requestID,
		timing,
		corsHandler(g.corsOrigins),
		securityHeaders,
	)

	srv := &fasthttp.Server{
		Handler:      handler,
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 60 * time.Second,
	}

	return srv.ListenAndServe(addr)
}

func (g *Gateway) handleChatCompletions(ctx *fasthttp.RequestCtx) {
	g.dispatchChat(ctx)
}

func (g *Gateway) handleCompletions(ctx *fasthttp.RequestCtx) {
	g.dispatchChat(ctx)
}

func (g *Gateway) handleEmbeddings(ctx *fasthttp.RequestCtx) {
	g.dispatchEmbeddings(ctx)
}

func (g *Gateway) handleHealth(ctx *fasthttp.RequestCtx) {
	if g.health == nil {
		writeJSON(ctx, map[string]any{"status": "ok", "version": "0.1.0"})
		return
	}
	snap := g.health.Snapshot()
	writeJSON(ctx, snap)
}

func (g *Gateway) handleReadiness(ctx *fasthttp.RequestCtx) {
	if g.health == nil || g.health.ReadinessOK() {
		writeJSON(ctx, map[string]string{"status": "ok"})
		return
	}
	ctx.SetStatusCode(fasthttp.StatusServiceUnavailable)
	writeJSON(ctx, map[string]string{"status": "unavailable"})
}

func writeJSON(ctx *fasthttp.RequestCtx, v any) {
	ctx.SetContentType("application/json")
	data, _ := json.Marshal(v)
	ctx.SetBody(data)
}
