// Package logger implements a non-blocking, batched request logger.
//
// Log entries are written to an internal buffered channel and flushed in
// batches by a background goroutine — so logging never blocks the proxy hot
// path. If the channel fills up (> 10 000 entries), new entries are dropped
// and counted in DroppedLogs.
package logger

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

const (
	channelBuffer = 10_000
	batchSize     = 100
	flushInterval = time.Second
)

type RequestLog struct {
	ID           uuid.UUID
	Provider     string
	Model        string
	InputTokens  uint32
	OutputTokens uint32
	LatencyMs    uint16
	Status       uint16
	Cached       bool
	CreatedAt    time.Time
}

type Logger struct {
	ch        chan RequestLog
	done      chan struct{}
	closeOnce sync.Once
	wg        sync.WaitGroup

	droppedLogs int64

	baseCtx context.Context
	log     *slog.Logger
}

func New(ctx context.Context, slogger *slog.Logger) (*Logger, error) {
	if ctx == nil {
		return nil, fmt.Errorf("logger: context must not be nil")
	}
	if slogger == nil {
		slogger = slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		}))
	}

	l := &Logger{
		ch:      make(chan RequestLog, channelBuffer),
		done:    make(chan struct{}),
		baseCtx: ctx,
		log:     slogger,
	}

	l.wg.Add(1)
	go l.run()

	return l, nil
}

func (l *Logger) Log(entry RequestLog) {
	select {
	case l.ch <- entry:
	default:
		atomic.AddInt64(&l.droppedLogs, 1)
	}
}

func (l *Logger) DroppedLogs() int64 {
	return atomic.LoadInt64(&l.droppedLogs)
}

func (l *Logger) Close() error {
	l.closeOnce.Do(func() {
		close(l.done)
	})
	l.wg.Wait()
	return nil
}

func (l *Logger) run() {
	defer l.wg.Done()

	ticker := time.NewTicker(flushInterval)
	defer ticker.Stop()

	batch := make([]RequestLog, 0, batchSize)

	flush := func(ctx context.Context) {
		if len(batch) == 0 {
			return
		}
		for _, e := range batch {
			l.log.InfoContext(ctx, "request",
				slog.String("id", e.ID.String()),
				slog.String("provider", e.Provider),
				slog.String("model", e.Model),
				slog.Uint64("input_tokens", uint64(e.InputTokens)),
				slog.Uint64("output_tokens", uint64(e.OutputTokens)),
				slog.Uint64("latency_ms", uint64(e.LatencyMs)),
				slog.Uint64("status", uint64(e.Status)),
				slog.Bool("cached", e.Cached),
				slog.Time("created_at", normalizeTime(e.CreatedAt)),
			)
		}
		batch = batch[:0]
	}

	for {
		select {
		case entry := <-l.ch:
			batch = append(batch, entry)
			if len(batch) >= batchSize {
				flush(l.baseCtx)
			}

		case <-ticker.C:
			flush(l.baseCtx)

		case <-l.done:
			for {
				select {
				case entry := <-l.ch:
					batch = append(batch, entry)
					if len(batch) >= batchSize {
						flush(l.baseCtx)
					}
				default:
					flush(l.baseCtx)
					return
				}
			}
		}
	}
}

func normalizeTime(t time.Time) time.Time {
	if t.IsZero() {
		return time.Now().UTC()
	}
	return t.UTC()
}
