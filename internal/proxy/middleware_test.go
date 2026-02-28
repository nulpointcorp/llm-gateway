package proxy

import (
	"testing"

	"github.com/valyala/fasthttp"
)

// --- recovery middleware ----------------------------------------------------

func TestRecovery_NoPanic(t *testing.T) {
	handler := recovery(func(ctx *fasthttp.RequestCtx) {
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetBodyString("ok")
	})

	ctx := &fasthttp.RequestCtx{}
	handler(ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200, got %d", ctx.Response.StatusCode())
	}
}

func TestRecovery_CatchesPanic(t *testing.T) {
	handler := recovery(func(ctx *fasthttp.RequestCtx) {
		panic("mock panic")
	})

	ctx := &fasthttp.RequestCtx{}
	handler(ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusInternalServerError {
		t.Errorf("expected 500, got %d", ctx.Response.StatusCode())
	}
	if string(ctx.Response.Header.ContentType()) != "application/json" {
		t.Errorf("expected application/json content type, got %s",
			string(ctx.Response.Header.ContentType()))
	}
	body := string(ctx.Response.Body())
	if !containsStr(body, "internal server error") {
		t.Errorf("expected error body to contain 'internal server error', got: %s", body)
	}
}

// --- requestID middleware ---------------------------------------------------

func TestRequestID_GeneratesWhenMissing(t *testing.T) {
	handler := requestID(func(ctx *fasthttp.RequestCtx) {
		id, _ := ctx.UserValue("request_id").(string)
		if id == "" {
			t.Error("request_id should be generated")
		}
	})

	ctx := &fasthttp.RequestCtx{}
	handler(ctx)

	respID := string(ctx.Response.Header.Peek("X-Request-ID"))
	if respID == "" {
		t.Error("X-Request-ID response header should be set")
	}
}

func TestRequestID_PreservesExisting(t *testing.T) {
	handler := requestID(func(ctx *fasthttp.RequestCtx) {
		id, _ := ctx.UserValue("request_id").(string)
		if id != "custom-id-123" {
			t.Errorf("expected preserved ID, got %s", id)
		}
	})

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.Set("X-Request-ID", "custom-id-123")
	handler(ctx)

	respID := string(ctx.Response.Header.Peek("X-Request-ID"))
	if respID != "custom-id-123" {
		t.Errorf("expected 'custom-id-123' in response, got %s", respID)
	}
}

// --- timing middleware ------------------------------------------------------

func TestTiming_SetsHeader(t *testing.T) {
	handler := timing(func(ctx *fasthttp.RequestCtx) {
		ctx.SetStatusCode(fasthttp.StatusOK)
	})

	ctx := &fasthttp.RequestCtx{}
	handler(ctx)

	rt := string(ctx.Response.Header.Peek("X-Response-Time"))
	if rt == "" {
		t.Error("X-Response-Time header should be set")
	}
}

// --- securityHeaders middleware ---------------------------------------------

func TestSecurityHeaders_AllSet(t *testing.T) {
	handler := securityHeaders(func(ctx *fasthttp.RequestCtx) {
		ctx.SetStatusCode(fasthttp.StatusOK)
	})

	ctx := &fasthttp.RequestCtx{}
	handler(ctx)

	expected := map[string]string{
		"Strict-Transport-Security": "max-age=31536000; includeSubDomains",
		"X-Content-Type-Options":    "nosniff",
		"X-Frame-Options":           "DENY",
		"X-XSS-Protection":          "0",
		"Content-Security-Policy":   "default-src 'none'",
		"Referrer-Policy":           "no-referrer",
	}

	for header, want := range expected {
		got := string(ctx.Response.Header.Peek(header))
		if got != want {
			t.Errorf("header %s: expected %q, got %q", header, want, got)
		}
	}

	pp := string(ctx.Response.Header.Peek("Permissions-Policy"))
	if pp == "" {
		t.Error("Permissions-Policy header should be set")
	}
}

// --- corsHandler middleware -------------------------------------------------

func TestCORS_Wildcard(t *testing.T) {
	handler := corsHandler(nil)(func(ctx *fasthttp.RequestCtx) {
		ctx.SetStatusCode(fasthttp.StatusOK)
	})

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod("GET")
	handler(ctx)

	origin := string(ctx.Response.Header.Peek("Access-Control-Allow-Origin"))
	if origin != "*" {
		t.Errorf("expected wildcard origin, got %q", origin)
	}
}

func TestCORS_WildcardExplicit(t *testing.T) {
	handler := corsHandler([]string{"*"})(func(ctx *fasthttp.RequestCtx) {
		ctx.SetStatusCode(fasthttp.StatusOK)
	})

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod("GET")
	handler(ctx)

	origin := string(ctx.Response.Header.Peek("Access-Control-Allow-Origin"))
	if origin != "*" {
		t.Errorf("expected wildcard, got %q", origin)
	}
}

func TestCORS_SpecificOrigins(t *testing.T) {
	origins := []string{"https://app.nulpoint.com", "https://dashboard.nulpoint.com"}
	handler := corsHandler(origins)(func(ctx *fasthttp.RequestCtx) {
		ctx.SetStatusCode(fasthttp.StatusOK)
	})

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod("GET")
	handler(ctx)

	got := string(ctx.Response.Header.Peek("Access-Control-Allow-Origin"))
	expected := "https://app.nulpoint.com, https://dashboard.nulpoint.com"
	if got != expected {
		t.Errorf("expected %q, got %q", expected, got)
	}
}

func TestCORS_PreflightReturns204(t *testing.T) {
	handler := corsHandler(nil)(func(ctx *fasthttp.RequestCtx) {
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetBodyString("should not be reached")
	})

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod("OPTIONS")
	handler(ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusNoContent {
		t.Errorf("preflight should return 204, got %d", ctx.Response.StatusCode())
	}
	if len(ctx.Response.Body()) != 0 {
		t.Error("preflight should have empty body")
	}
}

func TestCORS_AllowedHeaders(t *testing.T) {
	handler := corsHandler(nil)(func(ctx *fasthttp.RequestCtx) {
		ctx.SetStatusCode(fasthttp.StatusOK)
	})

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod("GET")
	handler(ctx)

	allowHeaders := string(ctx.Response.Header.Peek("Access-Control-Allow-Headers"))
	for _, h := range []string{"Authorization", "Content-Type", "X-Request-ID"} {
		if !containsStr(allowHeaders, h) {
			t.Errorf("expected %q in Allow-Headers, got %q", h, allowHeaders)
		}
	}
}

func TestCORS_AllowedMethods(t *testing.T) {
	handler := corsHandler(nil)(func(ctx *fasthttp.RequestCtx) {
		ctx.SetStatusCode(fasthttp.StatusOK)
	})

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod("GET")
	handler(ctx)

	methods := string(ctx.Response.Header.Peek("Access-Control-Allow-Methods"))
	for _, m := range []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"} {
		if !containsStr(methods, m) {
			t.Errorf("expected %q in Allow-Methods, got %q", m, methods)
		}
	}
}

// --- applyMiddleware --------------------------------------------------------

func TestApplyMiddleware_Order(t *testing.T) {
	var order []string

	mw1 := func(next fasthttp.RequestHandler) fasthttp.RequestHandler {
		return func(ctx *fasthttp.RequestCtx) {
			order = append(order, "mw1-before")
			next(ctx)
			order = append(order, "mw1-after")
		}
	}
	mw2 := func(next fasthttp.RequestHandler) fasthttp.RequestHandler {
		return func(ctx *fasthttp.RequestCtx) {
			order = append(order, "mw2-before")
			next(ctx)
			order = append(order, "mw2-after")
		}
	}

	handler := applyMiddleware(func(ctx *fasthttp.RequestCtx) {
		order = append(order, "handler")
	}, mw1, mw2)

	ctx := &fasthttp.RequestCtx{}
	handler(ctx)

	// mw1 is outermost, mw2 is inner.
	expected := []string{"mw1-before", "mw2-before", "handler", "mw2-after", "mw1-after"}
	if len(order) != len(expected) {
		t.Fatalf("expected %d calls, got %d: %v", len(expected), len(order), order)
	}
	for i, v := range expected {
		if order[i] != v {
			t.Errorf("position %d: expected %q, got %q", i, v, order[i])
		}
	}
}

func TestApplyMiddleware_NoMiddlewares(t *testing.T) {
	called := false
	handler := applyMiddleware(func(ctx *fasthttp.RequestCtx) {
		called = true
	})

	ctx := &fasthttp.RequestCtx{}
	handler(ctx)

	if !called {
		t.Error("handler should be called even with no middlewares")
	}
}

// --- helper -----------------------------------------------------------------

func containsStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
