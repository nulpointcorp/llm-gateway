// Package ratelimit implements per-workspace and per-key rate limiting using
// Redis sliding window counters with atomic Lua scripts.
package ratelimit

import (
	"context"
	"time"

	"github.com/redis/go-redis/v9"
)

// slidingWindowScript is an atomic Lua script that implements a sliding window
// rate limiter using a sorted set.
// KEYS[1] = Redis key
// ARGV[1] = current unix timestamp (nanoseconds as string)
// ARGV[2] = window size in nanoseconds
// ARGV[3] = limit (max requests per window)
// Returns: 1 if allowed, 0 if rate limited.
var slidingWindowScript = redis.NewScript(`
		local key    = KEYS[1]
		local now    = tonumber(ARGV[1])
		local window = tonumber(ARGV[2])
		local limit  = tonumber(ARGV[3])
		
		-- Remove expired entries.
		redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
		
		local count = redis.call('ZCARD', key)
		if count >= limit then
			return 0
		end
		
		-- Add current request with a unique member (now + random suffix).
		local member = tostring(now) .. tostring(math.random(1, 1000000))
		redis.call('ZADD', key, now, member)
		redis.call('PEXPIRE', key, math.ceil(window / 1000000))  -- window is in ns; PEXPIRE wants ms
		return 1
`)

const (
	rateLimitKey = "ratelimit:ws:rpm"
)

// RPMLimiter checks a global requests-per-minute limit using a Redis sliding window.
type RPMLimiter struct {
	rdb      *redis.Client
	rpmLimit int
}

// NewRPMLimiter creates a new RPMLimiter with the given global RPM limit.
// rpmLimit must be > 0; values ≤ 0 will block every request.
func NewRPMLimiter(rdb *redis.Client, rpmLimit int) *RPMLimiter {
	return &RPMLimiter{rdb: rdb, rpmLimit: rpmLimit}
}

// Allow returns true if the current request is within the rate limit.
func (r *RPMLimiter) Allow(ctx context.Context) (bool, error) {
	return r.check(ctx, rateLimitKey, r.rpmLimit)
}

func (r *RPMLimiter) check(ctx context.Context, key string, limit int) (bool, error) {
	now := time.Now().UnixNano()
	window := time.Minute.Nanoseconds()

	result, err := slidingWindowScript.Run(ctx, r.rdb,
		[]string{key},
		now, window, limit,
	).Int()
	if err != nil {
		// Redis unavailable — allow request (graceful degradation).
		return true, nil
	}

	return result == 1, nil
}
