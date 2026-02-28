"""
nulpoint Gateway — Python examples
====================================

Prerequisites:
    pip install openai

Run the gateway first:
    OPENAI_API_KEY=sk-... make run

Then run this file:
    python examples/python/basic_usage.py
"""

from openai import OpenAI

# Point the OpenAI client at the local gateway.
# The api_key is passed through to the provider but can be any non-empty string.
client = OpenAI(
    api_key="any-string",           # gateway handles auth with providers
    base_url="http://localhost:8080/v1",
)


def basic_chat() -> None:
    """Single-turn chat completion."""
    print("=== Basic chat ===")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print(response.choices[0].message.content)
    print(f"Tokens used: {response.usage.total_tokens}")
    print()


def multi_turn_chat() -> None:
    """Multi-turn conversation."""
    print("=== Multi-turn chat ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant who responds briefly."},
        {"role": "user",   "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user",   "content": "What's my name?"},
    ]
    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    print(response.choices[0].message.content)
    print()


def streaming_chat() -> None:
    """Streaming response — tokens are printed as they arrive."""
    print("=== Streaming chat ===")
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Count slowly from 1 to 5."}],
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print("\n")


def use_anthropic_model() -> None:
    """Route to Anthropic Claude by specifying a Claude model name."""
    print("=== Anthropic Claude ===")
    response = client.chat.completions.create(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello! What model are you?"}],
        max_tokens=128,
    )
    print(response.choices[0].message.content)
    print()


def demonstrate_caching() -> None:
    """
    The gateway caches identical non-streaming requests.
    The second call returns instantly from cache (check X-Cache: HIT header).
    """
    print("=== Response caching ===")
    question = "What is the tallest mountain on Earth?"

    for i in range(2):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
        )
        # The underlying HTTP response carries an X-Cache header:
        # first request → MISS, second → HIT.
        print(f"Request {i + 1}: {response.choices[0].message.content[:60]}...")
    print()


def embeddings() -> None:
    """
    Embed text using the /v1/embeddings endpoint.
    Supported models: text-embedding-3-small, text-embedding-3-large,
    text-embedding-ada-002 (OpenAI), mistral-embed, text-embedding-004 (Gemini).
    """
    print("=== Embeddings ===")

    # Single input — plain string
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="The quick brown fox jumps over the lazy dog.",
    )
    vec = response.data[0].embedding
    print(f"Single input → dims: {len(vec)}, first 4: {[round(v, 4) for v in vec[:4]]}")

    # Batch input — list of strings; one vector is returned per item
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=["Hello world", "How are you?", "Embeddings are useful."],
    )
    print(f"Batch input  → {len(response.data)} vectors, dims: {len(response.data[0].embedding)}")
    print(f"Tokens used: {response.usage.total_tokens}")
    print()


if __name__ == "__main__":
    basic_chat()
    multi_turn_chat()
    streaming_chat()
    use_anthropic_model()
    demonstrate_caching()
    embeddings()
