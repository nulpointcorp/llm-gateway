package proxy

import (
	"log/slog"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/valyala/fasthttp"
)

// recovery catches panics in any handler and returns a 500 without crashing
// the server process. The panic value is logged at ERROR level.
func recovery(next fasthttp.RequestHandler) fasthttp.RequestHandler {
	return func(ctx *fasthttp.RequestCtx) {
		defer func() {
			if r := recover(); r != nil {
				slog.Error("handler_panic",
					slog.Any("panic", r),
					slog.String("path", string(ctx.Path())),
					slog.String("method", string(ctx.Method())),
				)
				ctx.ResetBody()
				ctx.SetStatusCode(fasthttp.StatusInternalServerError)
				ctx.SetContentType("application/json")
				ctx.SetBodyString(`{"error":{"message":"internal server error","type":"server_error","code":"internal_error"}}`)
			}
		}()
		next(ctx)
	}
}

// requestID ensures every request has an X-Request-ID header. If the client
// does not supply one a UUID v4 is generated. The ID is also stored in the
// request context under the key "request_id" for downstream handlers.
func requestID(next fasthttp.RequestHandler) fasthttp.RequestHandler {
	return func(ctx *fasthttp.RequestCtx) {
		id := string(ctx.Request.Header.Peek("X-Request-ID"))
		if id == "" {
			id = uuid.New().String()
		}
		ctx.Response.Header.Set("X-Request-ID", id)
		ctx.SetUserValue("request_id", id)
		next(ctx)
	}
}

// timing records the total handler duration in the X-Response-Time response
// header. The value uses Go's default Duration string format (e.g. "2.5ms").
func timing(next fasthttp.RequestHandler) fasthttp.RequestHandler {
	return func(ctx *fasthttp.RequestCtx) {
		start := time.Now()
		next(ctx)
		ctx.Response.Header.Set("X-Response-Time", time.Since(start).String())
	}
}

// securityHeaders adds HTTP security headers recommended by OWASP to every
// response. These headers have no effect on the API functionality but harden
// the server against common web attacks.
func securityHeaders(next fasthttp.RequestHandler) fasthttp.RequestHandler {
	return func(ctx *fasthttp.RequestCtx) {
		next(ctx)
		h := &ctx.Response.Header
		h.Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("X-Frame-Options", "DENY")
		// X-XSS-Protection is deprecated; set to 0 and rely on CSP instead.
		h.Set("X-XSS-Protection", "0")
		// API-only CSP: no HTML resources served, so deny everything.
		h.Set("Content-Security-Policy", "default-src 'none'")
		h.Set("Referrer-Policy", "no-referrer")
		h.Set("Permissions-Policy", "geolocation=(), camera=(), microphone=()")
	}
}

// corsHandler returns a CORS middleware configured for the given allowed origins.
//
//   - nil or []string{"*"} → Access-Control-Allow-Origin: *  (open)
//   - specific origins      → joined with ", "  (strict allowlist)
//
// OPTIONS preflight requests are answered with 204 No Content and no body.
func corsHandler(origins []string) func(fasthttp.RequestHandler) fasthttp.RequestHandler {
	origin := "*"
	if len(origins) > 0 && !(len(origins) == 1 && origins[0] == "*") {
		origin = strings.Join(origins, ", ")
	}
	return func(next fasthttp.RequestHandler) fasthttp.RequestHandler {
		return func(ctx *fasthttp.RequestCtx) {
			ctx.Response.Header.Set("Access-Control-Allow-Origin", origin)
			ctx.Response.Header.Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
			ctx.Response.Header.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Request-ID")

			if string(ctx.Method()) == fasthttp.MethodOptions {
				ctx.SetStatusCode(fasthttp.StatusNoContent)
				return
			}
			next(ctx)
		}
	}
}

// applyMiddleware wraps h with the given middleware chain. The first middleware
// in the slice becomes the outermost wrapper (executes first on request,
// last on response). This matches the conventional "left-to-right" ordering:
//
//	applyMiddleware(h, mw1, mw2) → mw1(mw2(h))
func applyMiddleware(h fasthttp.RequestHandler, mws ...func(fasthttp.RequestHandler) fasthttp.RequestHandler) fasthttp.RequestHandler {
	for i := len(mws) - 1; i >= 0; i-- {
		h = mws[i](h)
	}
	return h
}
