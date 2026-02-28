package ratelimit_test

import (
	"context"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/nulpointcorp/llm-gateway/internal/ratelimit"
	"github.com/redis/go-redis/v9"
)

func newTestRedis(t *testing.T) (*redis.Client, func()) {
	t.Helper()
	mr, err := miniredis.Run()
	if err != nil {
		t.Fatalf("miniredis: %v", err)
	}
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	return client, func() {
		client.Close()
		mr.Close()
	}
}

func TestRPMLimiter_AllowsUnderLimit(t *testing.T) {
	rdb, cleanup := newTestRedis(t)
	defer cleanup()

	const limit = 10
	limiter := ratelimit.NewRPMLimiter(rdb, limit)
	ctx := context.Background()

	for i := 0; i < limit; i++ {
		allowed, err := limiter.Allow(ctx)
		if err != nil {
			t.Fatalf("unexpected error at iteration %d: %v", i, err)
		}
		if !allowed {
			t.Fatalf("expected allowed=true at iteration %d", i)
		}
	}
}

func TestRPMLimiter_BlocksOverLimit(t *testing.T) {
	rdb, cleanup := newTestRedis(t)
	defer cleanup()

	const limit = 3
	limiter := ratelimit.NewRPMLimiter(rdb, limit)
	ctx := context.Background()

	for i := 0; i < limit; i++ {
		allowed, err := limiter.Allow(ctx)
		if err != nil {
			t.Fatalf("unexpected error at iteration %d: %v", i, err)
		}
		if !allowed {
			t.Fatalf("expected allowed=true at iteration %d", i)
		}
	}

	// The (limit+1)th request must be blocked.
	allowed, err := limiter.Allow(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if allowed {
		t.Error("expected allowed=false after limit exceeded")
	}
}

func TestRPMLimiter_DegradedGracefully_WhenRedisDown(t *testing.T) {
	rdb, cleanup := newTestRedis(t)
	// Close Redis before making any calls — limiter must allow requests.
	cleanup()

	limiter := ratelimit.NewRPMLimiter(rdb, 5)
	ctx := context.Background()

	allowed, err := limiter.Allow(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !allowed {
		t.Error("expected allowed=true when Redis is unavailable (graceful degradation)")
	}
}
