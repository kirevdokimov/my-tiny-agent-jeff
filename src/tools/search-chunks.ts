import type OpenAI from "openai";
import { observe } from "@langfuse/tracing";
import { searchChunks } from "../make-embeddings";

/** Tool definition to hand to the chat completions API. */
export const searchChunksToolDefinition: OpenAI.Chat.Completions.ChatCompletionTool = {
    type: "function",
    function: {
        name: "search_chunks",
        description:
            "Semantic search over the user's personal notes (Russian and English). " +
            "Use this to find relevant notes before answering questions about the user's life, " +
            "thoughts, plans, finances, or past conversations. Returns the most relevant note chunks.",
        parameters: {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: "Natural-language search query.",
                },
                limit: {
                    type: "number",
                    description: "How many chunks to return (default 5, max 20).",
                },
            },
            required: ["query"],
            additionalProperties: false,
        },
    },
};

const MAX_CHUNK_CHARS = 2_000;

const formatResult = (result: { path: string; text: string; score: number }, index: number): string => {
    const text =
        result.text.length > MAX_CHUNK_CHARS
            ? result.text.slice(0, MAX_CHUNK_CHARS) + "\n...[truncated]"
            : result.text;
    return `[${index + 1}] score=${result.score.toFixed(3)} path=${result.path}\n${text}`;
};

/** Executes a semantic search and returns a compact, LLM-readable list. */
export const searchChunksToolExecute = observe(
    async (query: string, limit = 5): Promise<string> => {
        if (typeof query !== "string" || query.trim().length === 0) {
            return "Error: `query` must be a non-empty string.";
        }

        const cappedLimit = Math.min(20, Math.max(1, Math.floor(limit) || 5));
        const results = await searchChunks(query, cappedLimit);

        if (results.length === 0) {
            return "No relevant chunks found. The embedding cache may be empty — run `bun run make-embeddings` to build it.";
        }

        return results.map(formatResult).join("\n\n---\n\n");
    },
    { name: "search_chunks", asType: "tool" },
);
