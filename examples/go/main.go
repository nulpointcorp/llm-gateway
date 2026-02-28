// nulpoint Gateway — Go examples
//
// Prerequisites:
//
//	go get github.com/openai/openai-go/v3
//
// Run the gateway first:
//
//	OPENAI_API_KEY=sk-... make run
//
// Then run this file:
//
//	go run examples/go/main.go
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func main() {
	// Point the OpenAI Go client at the local gateway.
	// The api_key value is forwarded to the provider but is not validated by the gateway itself.
	client := openai.NewClient(
		option.WithBaseURL("http://localhost:8080/v1"),
	)

	ctx := context.Background()

	if err := basicChat(ctx, client); err != nil {
		log.Fatalf("basicChat: %v", err)
	}
	if err := multiTurnChat(ctx, client); err != nil {
		log.Fatalf("multiTurnChat: %v", err)
	}
	if err := streamingChat(ctx, client); err != nil {
		log.Fatalf("streamingChat: %v", err)
	}
	if err := useAnthropicModel(ctx, client); err != nil {
		log.Fatalf("useAnthropicModel: %v", err)
	}
	if err := demonstrateCaching(ctx, client); err != nil {
		log.Fatalf("demonstrateCaching: %v", err)
	}
	if err := embeddings(ctx, client); err != nil {
		log.Fatalf("embeddings: %v", err)
	}
}

// basicChat sends a single-turn completion to GPT-4o.
func basicChat(ctx context.Context, client openai.Client) error {
	fmt.Println("=== Basic chat ===")

	resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: openai.ChatModelGPT4o,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("What is the capital of France?"),
		},
	})
	if err != nil {
		return err
	}

	fmt.Println(resp.Choices[0].Message.Content)
	fmt.Printf("Tokens used: %d\n\n", resp.Usage.TotalTokens)
	return nil
}

// multiTurnChat sends a conversation with prior context.
func multiTurnChat(ctx context.Context, client openai.Client) error {
	fmt.Println("=== Multi-turn chat ===")

	resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: openai.ChatModelGPT4o,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage("You are a helpful assistant who responds briefly."),
			openai.UserMessage("My name is Alice."),
			openai.AssistantMessage("Nice to meet you, Alice!"),
			openai.UserMessage("What's my name?"),
		},
	})
	if err != nil {
		return err
	}

	fmt.Println(resp.Choices[0].Message.Content)
	fmt.Println()
	return nil
}

// streamingChat streams tokens as they arrive using server-sent events.
func streamingChat(ctx context.Context, client openai.Client) error {
	fmt.Println("=== Streaming chat ===")

	stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
		Model: openai.ChatModelGPT4o,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Count slowly from 1 to 5."),
		},
	})

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			fmt.Print(chunk.Choices[0].Delta.Content)
		}
	}
	if err := stream.Err(); err != nil && !errors.Is(err, io.EOF) {
		return err
	}

	fmt.Println()
	return nil
}

// useAnthropicModel routes to Anthropic Claude by specifying a Claude model name.
// The gateway resolves the provider from the model field automatically.
func useAnthropicModel(ctx context.Context, client openai.Client) error {
	fmt.Println("=== Anthropic Claude ===")

	resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: openai.ChatModel("claude-3-5-sonnet"),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello! What model are you?"),
		},
		MaxTokens: openai.Int(128),
	})
	if err != nil {
		return err
	}

	fmt.Println(resp.Choices[0].Message.Content)
	fmt.Println()
	return nil
}

// embeddings embeds text using the /v1/embeddings endpoint.
// Supported models: text-embedding-3-small, text-embedding-3-large,
// text-embedding-ada-002 (OpenAI), mistral-embed, text-embedding-004 (Gemini).
func embeddings(ctx context.Context, client openai.Client) error {
	fmt.Println("=== Embeddings ===")

	// Single input.
	resp, err := client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Model: openai.EmbeddingModelTextEmbedding3Small,
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: []string{"The quick brown fox jumps over the lazy dog."},
		},
	})
	if err != nil {
		return err
	}
	vec := resp.Data[0].Embedding
	fmt.Printf("Single input → dims: %d, first 4: %.4f %.4f %.4f %.4f\n",
		len(vec), vec[0], vec[1], vec[2], vec[3])

	// Batch input — one vector is returned per string.
	resp, err = client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Model: openai.EmbeddingModelTextEmbedding3Small,
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: []string{
				"Hello world",
				"How are you?",
				"Embeddings are useful.",
			},
		},
	})
	if err != nil {
		return err
	}
	fmt.Printf("Batch input  → %d vectors, dims: %d\n", len(resp.Data), len(resp.Data[0].Embedding))
	fmt.Printf("Tokens used: %d\n\n", resp.Usage.TotalTokens)
	return nil
}

// demonstrateCaching shows how the gateway caches identical non-streaming requests.
// The second call returns instantly from cache (check X-Cache: HIT in gateway logs).
func demonstrateCaching(ctx context.Context, client openai.Client) error {
	fmt.Println("=== Response caching ===")

	const question = "What is the tallest mountain on Earth?"

	for i := range 2 {
		resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model: openai.ChatModelGPT4o,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(question),
			},
		})
		if err != nil {
			return err
		}
		content := resp.Choices[0].Message.Content
		if len(content) > 60 {
			content = content[:60] + "..."
		}
		fmt.Printf("Request %d: %s\n", i+1, content)
	}

	fmt.Println()
	return nil
}
