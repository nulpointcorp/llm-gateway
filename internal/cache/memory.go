// Package cache provides caching implementations for the LLM gateway.
//
// Two backends are available:
//   - ExactCache  — Redis-backed, recommended for production clusters.
//   - MemoryCache — in-process TTL cache, zero external dependencies.
//     Ideal for single-instance deployments or local development.
//
// Both implement the Cache interface so they are fully interchangeable.
package cache

import (
	"context"
	"sync"
	"time"
)

// memItem stores a cached value together with its expiry time.
type memItem struct {
	data      []byte
	expiresAt time.Time
}

// MemoryCache is a simple in-process cache with per-entry TTL.
//
// It is safe for concurrent use. A background goroutine periodically
// removes expired entries to prevent unbounded memory growth.
//
// Use this backend when Redis is not available — for local development,
// single-instance deployments, or integration tests. For distributed
// (multi-replica) deployments use ExactCache (Redis) instead so that
// all replicas share the same cache.
type MemoryCache struct {
	mu    sync.RWMutex
	items map[string]memItem

	done chan struct{}
}

// NewMemoryCache creates a MemoryCache and starts the background cleanup loop.
// The cleanup goroutine stops when ctx is cancelled or Close is called.
func NewMemoryCache(ctx context.Context) *MemoryCache {
	c := &MemoryCache{
		items: make(map[string]memItem),
		done:  make(chan struct{}),
	}
	go c.cleanup(ctx)
	return c
}

// Get returns the cached value for key. Returns (nil, false) on a miss or if
// the entry has expired. Expired entries are removed lazily on access.
func (c *MemoryCache) Get(_ context.Context, key string) ([]byte, bool) {
	c.mu.RLock()
	item, ok := c.items[key]
	c.mu.RUnlock()

	if !ok {
		return nil, false
	}

	if time.Now().After(item.expiresAt) {
		// Lazy expiry — remove the stale entry without blocking reads.
		c.mu.Lock()
		delete(c.items, key)
		c.mu.Unlock()
		return nil, false
	}

	return item.data, true
}

// Set stores value under key for the duration of ttl.
// A zero or negative ttl is treated as a 1-hour TTL.
func (c *MemoryCache) Set(_ context.Context, key string, value []byte, ttl time.Duration) error {
	if ttl <= 0 {
		ttl = time.Hour
	}

	c.mu.Lock()
	c.items[key] = memItem{
		data:      value,
		expiresAt: time.Now().Add(ttl),
	}
	c.mu.Unlock()

	return nil
}

// Delete removes key from the cache. Returns nil if the key did not exist.
func (c *MemoryCache) Delete(_ context.Context, key string) error {
	c.mu.Lock()
	delete(c.items, key)
	c.mu.Unlock()
	return nil
}

// Len returns the number of entries currently held in the cache
// (including entries that may have expired but not yet been evicted).
func (c *MemoryCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}

// Close stops the background cleanup goroutine.
func (c *MemoryCache) Close() {
	close(c.done)
}

// cleanup runs every 5 minutes and evicts all expired entries.
func (c *MemoryCache) cleanup(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.evictExpired()
		case <-ctx.Done():
			return
		case <-c.done:
			return
		}
	}
}

func (c *MemoryCache) evictExpired() {
	now := time.Now()

	c.mu.Lock()
	for k, v := range c.items {
		if now.After(v.expiresAt) {
			delete(c.items, k)
		}
	}
	c.mu.Unlock()
}
