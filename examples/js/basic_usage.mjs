/**
 * nulpoint Gateway — JavaScript / Node.js examples
 * ===================================================
 *
 * Prerequisites:
 *   npm install openai
 *
 * Run the gateway first:
 *   OPENAI_API_KEY=sk-... make run
 *
 * Then run this file:
 *   node examples/js/basic_usage.mjs
 */

import OpenAI from "openai";

// Point the OpenAI client at the local gateway.
const client = new OpenAI({
  apiKey: "any-string", // gateway forwards auth to providers
  baseURL: "http://localhost:8080/v1",
});

// ── Basic chat completion ────────────────────────────────────────────────────
async function basicChat() {
  console.log("=== Basic chat ===");

  const response = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "What is the capital of Japan?" }],
  });

  console.log(response.choices[0].message.content);
  console.log(`Tokens used: ${response.usage.total_tokens}\n`);
}

// ── Multi-turn conversation ──────────────────────────────────────────────────
async function multiTurnChat() {
  console.log("=== Multi-turn chat ===");

  const response = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system",    content: "You are a concise assistant." },
      { role: "user",      content: "My favourite colour is blue." },
      { role: "assistant", content: "Got it!" },
      { role: "user",      content: "What is my favourite colour?" },
    ],
  });

  console.log(response.choices[0].message.content, "\n");
}

// ── Streaming response ───────────────────────────────────────────────────────
async function streamingChat() {
  console.log("=== Streaming chat ===");

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Count from 1 to 5, one number per line." }],
    stream: true,
  });

  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content ?? "";
    process.stdout.write(content);
  }
  console.log("\n");
}

// ── Use Anthropic Claude ─────────────────────────────────────────────────────
async function useAnthropicModel() {
  console.log("=== Anthropic Claude ===");

  const response = await client.chat.completions.create({
    model: "claude-3-5-sonnet",
    messages: [{ role: "user", content: "What model are you running on?" }],
    max_tokens: 128,
  });

  console.log(response.choices[0].message.content, "\n");
}

// ── Failover demo ────────────────────────────────────────────────────────────
// If the primary provider fails, the gateway automatically retries the next
// one in the fallback chain (openai → anthropic → gemini → mistral).
// No code changes needed — just set multiple provider API keys.
async function failoverDemo() {
  console.log("=== Failover (transparent) ===");
  console.log(
    "The gateway automatically routes to the next available provider.\n" +
    "Enable multiple provider keys to see failover in action.\n"
  );
}

// ── Embeddings ────────────────────────────────────────────────────────────────
// Supported models: text-embedding-3-small, text-embedding-3-large,
// text-embedding-ada-002 (OpenAI), mistral-embed, text-embedding-004 (Gemini).
async function embeddings() {
  console.log("=== Embeddings ===");

  // Single input — plain string
  const single = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: "The quick brown fox jumps over the lazy dog.",
  });
  const vec = single.data[0].embedding;
  const first4 = vec.slice(0, 4).map((v) => v.toFixed(4)).join(", ");
  console.log(`Single input → dims: ${vec.length}, first 4: ${first4}`);

  // Batch input — array of strings; one vector returned per item
  const batch = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: ["Hello world", "How are you?", "Embeddings are useful."],
  });
  console.log(`Batch input  → ${batch.data.length} vectors, dims: ${batch.data[0].embedding.length}`);
  console.log(`Tokens used: ${batch.usage.total_tokens}\n`);
}

// ── Run all examples ─────────────────────────────────────────────────────────
try {
  await basicChat();
  await multiTurnChat();
  await streamingChat();
  await useAnthropicModel();
  await failoverDemo();
  await embeddings();
} catch (err) {
  console.error("Error:", err.message);
  process.exit(1);
}
