BINARY   := gateway
BUILD_DIR := bin
MODULE   := github.com/nulpoint/gateway
VERSION  := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS  := -s -w -X main.version=$(VERSION)

.PHONY: build run test test-race test-short lint bench \
        docker-up docker-down docker-logs docker-build \
        help

## ── Development ──────────────────────────────────────────────────────────────

build:                ## Build the binary to ./bin/gateway
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=0 go build -trimpath -ldflags="$(LDFLAGS)" -o $(BUILD_DIR)/$(BINARY) ./cmd/gateway

run:                  ## Run the gateway locally (reads .env if present)
	@if [ -f .env ]; then set -a && . ./.env && set +a; fi && \
	 CACHE_MODE=$${CACHE_MODE:-memory} go run ./cmd/gateway

test:                 ## Run all unit tests
	go test ./...

test-race:            ## Run tests with the race detector enabled
	go test -race ./...

test-short:           ## Run fast tests only (skip latency SLA tests)
	go test -short ./...

test-cover:           ## Generate HTML coverage report
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

lint:                 ## Run golangci-lint
	golangci-lint run ./...

bench:                ## Run proxy latency benchmark (30s)
	go test -bench=BenchmarkProxy -benchtime=30s -benchmem ./internal/proxy/

## ── Docker ───────────────────────────────────────────────────────────────────

docker-build:         ## Build the Docker image
	docker build -t nulpoint/gateway:$(VERSION) .

docker-up:            ## Start with in-memory cache (no Redis)
	docker compose up -d

docker-up-redis:      ## Start with Redis cache
	docker compose --profile redis up -d

docker-down:          ## Stop all services
	docker compose down

docker-logs:          ## Tail logs from all services
	docker compose logs -f

## ── Misc ─────────────────────────────────────────────────────────────────────

help:                 ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
