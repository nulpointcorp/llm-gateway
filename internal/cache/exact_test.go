package cache

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
)

// newTestCache starts a miniredis server and returns an ExactCache backed by
// it plus a cleanup function that stops the server.
func newTestCache(t *testing.T) (*ExactCache, *miniredis.Miniredis) {
	t.Helper()

	mr := miniredis.RunT(t)

	c, err := NewExactCacheFromURL(context.Background(), "redis://"+mr.Addr())
	if err != nil {
		t.Fatalf("NewExactCacheFromURL: %v", err)
	}

	t.Cleanup(func() { _ = c.Close() })

	return c, mr
}

// TestGetMiss verifies that Get returns (nil, false) when the key is absent.
func TestGetMiss(t *testing.T) {
	c, _ := newTestCache(t)

	data, ok := c.Get(context.Background(), "nonexistent-key")
	if ok {
		t.Fatal("expected cache miss, got hit")
	}
	if data != nil {
		t.Fatalf("expected nil data on miss, got %v", data)
	}
}

// TestSetAndGetHit verifies that a value written with Set can be read back.
func TestSetAndGetHit(t *testing.T) {
	c, _ := newTestCache(t)

	key := "mock-key"
	want := []byte(`{"answer":42}`)

	if err := c.Set(context.Background(), key, want, time.Hour); err != nil {
		t.Fatalf("Set: %v", err)
	}

	got, ok := c.Get(context.Background(), key)
	if !ok {
		t.Fatal("expected cache hit, got miss")
	}
	if string(got) != string(want) {
		t.Fatalf("Get returned %q, want %q", got, want)
	}
}

// TestTTLIsSet verifies that the TTL is actually stored in Redis by advancing
// miniredis time past the TTL and confirming the key expires.
func TestTTLIsSet(t *testing.T) {
	c, mr := newTestCache(t)

	key := "ttl-key"
	ttl := 10 * time.Second

	if err := c.Set(context.Background(), key, []byte("payload"), ttl); err != nil {
		t.Fatalf("Set: %v", err)
	}

	// Confirm the key is present before expiry.
	if _, ok := c.Get(context.Background(), key); !ok {
		t.Fatal("key should exist before TTL expires")
	}

	// Advance miniredis clock beyond the TTL.
	mr.FastForward(ttl + time.Second)

	// The key must be gone now.
	if _, ok := c.Get(context.Background(), key); ok {
		t.Fatal("key should have expired after TTL")
	}
}

// TestDelete verifies that Delete removes an existing key.
func TestDelete(t *testing.T) {
	c, _ := newTestCache(t)

	key := "delete-key"
	if err := c.Set(context.Background(), key, []byte("to-be-deleted"), time.Hour); err != nil {
		t.Fatalf("Set: %v", err)
	}

	if err := c.Delete(context.Background(), key); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	if _, ok := c.Get(context.Background(), key); ok {
		t.Fatal("key should be gone after Delete")
	}
}

// TestDeleteMissingKey verifies that deleting a non-existent key does not
// return an error.
func TestDeleteMissingKey(t *testing.T) {
	c, _ := newTestCache(t)

	if err := c.Delete(context.Background(), "ghost-key"); err != nil {
		t.Fatalf("Delete of missing key returned error: %v", err)
	}
}

// TestGracefulDegradationGet verifies that Get returns (nil, false) when Redis
// is unreachable instead of panicking or returning an error to the caller.
func TestGracefulDegradationGet(t *testing.T) {
	mr := miniredis.RunT(t)
	addr := mr.Addr()

	c, err := NewExactCacheFromURL(context.Background(), "redis://"+addr)
	if err != nil {
		t.Fatalf("NewExactCacheFromURL: %v", err)
	}
	defer func() { _ = c.Close() }()

	// Take the server down.
	mr.Close()

	data, ok := c.Get(context.Background(), "any-key")
	if ok {
		t.Fatal("expected miss when Redis is down, got hit")
	}
	if data != nil {
		t.Fatalf("expected nil data when Redis is down, got %v", data)
	}
}

// TestGracefulDegradationSet verifies that Set returns nil (not an error) when
// Redis is unreachable so the proxy request is not aborted.
func TestGracefulDegradationSet(t *testing.T) {
	mr := miniredis.RunT(t)
	addr := mr.Addr()

	c, err := NewExactCacheFromURL(context.Background(), "redis://"+addr)
	if err != nil {
		t.Fatalf("NewExactCacheFromURL: %v", err)
	}
	defer func() { _ = c.Close() }()

	// Take the server down.
	mr.Close()

	err = c.Set(context.Background(), "any-key", []byte("value"), time.Hour)
	if err != nil {
		t.Fatalf("Set must return nil on Redis error for graceful degradation, got: %v", err)
	}
}

// TestNewExactCacheInvalidURL verifies that an invalid Redis URL is rejected.
func TestNewExactCacheInvalidURL(t *testing.T) {
	_, err := NewExactCacheFromURL(context.Background(), "not-a-valid-url")
	if err == nil {
		t.Fatal("expected error for invalid URL, got nil")
	}
}

// TestCacheImplementsInterface is a compile-time assertion that ExactCache
// satisfies the Cache interface.
func TestCacheImplementsInterface(t *testing.T) {
	var _ Cache = (*ExactCache)(nil)
}
