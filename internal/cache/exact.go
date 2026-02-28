// Package cache provides Redis-backed exact-match caching.
//
// Key format: SHA-256(workspace_id + provider + model + temperature + messages_json)
//
// Graceful degradation: when Redis is unavailable, Get returns (nil, false)
// and Set returns nil so the proxy never fails due to a missing cache.
package cache

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/redis/go-redis/v9"
)

const defaultCacheTimeout = 500 * time.Millisecond

// ExactCache is a Redis-backed cache that implements the Cache interface.
//
// All operations degrade gracefully when Redis is unavailable:
//   - Get returns (nil, false) on any error.
//   - Set returns nil even on error (silent degradation keeps proxy alive).
//   - Delete returns the underlying error so callers can log/handle it.
type ExactCache struct {
	client       *redis.Client
	queryTimeout time.Duration
}

// NewExactCacheFromClient wraps an existing Redis client in an ExactCache.
// The caller owns the client lifecycle (creation and Close).
func NewExactCacheFromClient(redisCli *redis.Client) *ExactCache {
	return &ExactCache{client: redisCli, queryTimeout: defaultCacheTimeout}
}

// NewExactCacheFromURL parses redisURL, creates a Redis client, verifies the
// connection with a PING, and returns an ExactCache.
// Returns an error if the URL is invalid or the initial ping fails.
func NewExactCacheFromURL(ctx context.Context, redisURL string) (*ExactCache, error) {
	if ctx == nil {
		return nil, fmt.Errorf("cache: context must not be nil")
	}

	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("cache: parse url: %w", err)
	}

	cli := redis.NewClient(opts)

	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if err := cli.Ping(pingCtx).Err(); err != nil {
		_ = cli.Close()
		return nil, fmt.Errorf("cache: ping: %w", err)
	}

	return &ExactCache{client: cli, queryTimeout: defaultCacheTimeout}, nil
}

// Get retrieves the value for key from Redis.
// Returns (data, true) on a hit and (nil, false) on a miss or any error.
// Redis errors are logged at WARN level but not propagated.
func (c *ExactCache) Get(ctx context.Context, key string) ([]byte, bool) {
	ctx, cancel := context.WithTimeout(ctx, c.queryTimeout)
	defer cancel()

	val, err := c.client.Get(ctx, key).Bytes()
	if err != nil {
		if !errors.Is(err, redis.Nil) {
			slog.WarnContext(ctx, "cache_get_error",
				slog.String("key", key),
				slog.String("error", err.Error()),
			)
		}
		return nil, false
	}

	return val, true
}

// Set stores value under key with the given TTL.
// Returns nil even on Redis error — graceful degradation keeps the proxy
// functioning when the cache layer is unavailable.
func (c *ExactCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, c.queryTimeout)
	defer cancel()

	if err := c.client.Set(ctx, key, value, ttl).Err(); err != nil {
		slog.WarnContext(ctx, "cache_set_error",
			slog.String("key", key),
			slog.String("error", err.Error()),
		)
	}

	return nil // always nil — degrade gracefully
}

// Delete removes key from Redis.
// Returns the underlying error so callers can decide how to handle it.
func (c *ExactCache) Delete(ctx context.Context, key string) error {
	ctx, cancel := context.WithTimeout(ctx, c.queryTimeout)
	defer cancel()

	if err := c.client.Del(ctx, key).Err(); err != nil {
		return fmt.Errorf("cache: DEL %s: %w", key, err)
	}

	return nil
}

// Close releases the Redis connection pool.
func (c *ExactCache) Close() error {
	return c.client.Close()
}
