import { appendFile, mkdir, readFile, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { join } from "node:path";
import OpenAI from "openai";

const DATA_DIR = "./external-data";
const CACHE_FILE = "./out/embeddings-cache.jsonl";
const MODEL = "qwen/qwen3-embedding-8b";
const BASE_URL = "https://openrouter.ai/api/v1";
const BATCH_SIZE = 32;

const API_KEY = process.env.API_KEY;

const openai = new OpenAI({ baseURL: BASE_URL, apiKey: API_KEY });

/** Recursively list every file path under `dir`. */
const listFiles = async (dir: string): Promise<string[]> => {
    const entries = await readdir(dir, { withFileTypes: true, recursive: true });
    return entries
        .filter((entry) => entry.isFile())
        .map((entry) => join(entry.parentPath, entry.name));
};

/**
 * Splits text into chunks at an empty line (or a run of empty lines).
 * Each returned chunk is trimmed and guaranteed to be non-empty.
 */
const chunkText = (text: string): string[] => {
    return text
        .split(/\n\s*\n/)
        .map((chunk) => chunk.trim())
        .filter((chunk) => chunk.length > 0);
};

type FileChunks = {
    path: string;
    chunks: string[];
};

/** Read every file, one after another, and split its content into chunks. */
const readAllFiles = async (dir: string): Promise<FileChunks[]> => {
    const files: FileChunks[] = [];

    for (const filePath of await listFiles(dir)) {
        const content = await readFile(filePath, "utf8");
        files.push({ path: filePath, chunks: chunkText(content) });
    }

    return files;
};

type Chunk = {
    path: string;
    text: string;
    /** Stable id: model + content, so changing model or text re-embeds. */
    key: string;
};

const chunkKey = (text: string): string =>
    createHash("sha256").update(MODEL).update("\u0000").update(text).digest("hex");

/** L2-normalize so cosine similarity is just a dot product. */
const normalize = (vector: number[]): number[] => {
    let sum = 0;
    for (const value of vector) sum += value * value;
    const norm = Math.sqrt(sum) || 1;
    return vector.map((value) => value / norm);
};

// Embeddings are stored as base64 Float32 (4 bytes/dim) instead of JSON numbers,
// which is ~2x smaller and exact. 4096 dims -> ~16 KB per chunk.
const encodeVector = (vector: number[]): string =>
    Buffer.from(Float32Array.from(vector).buffer).toString("base64");

const decodeVector = (base64: string): Float32Array => {
    const buffer = Buffer.from(base64, "base64");
    return new Float32Array(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
};

type CacheRecord = {
    key: string;
    model: string;
    path: string;
    text: string;
    embedding: string;
};

const loadCache = async (file: string): Promise<Map<string, Float32Array>> => {
    const cache = new Map<string, Float32Array>();
    if (!existsSync(file)) return cache;

    const content = await readFile(file, "utf8");
    for (const line of content.split("\n")) {
        if (!line.trim()) continue;
        const record = JSON.parse(line) as CacheRecord;
        cache.set(record.key, decodeVector(record.embedding));
    }
    return cache;
};

/** One embeddings request; retries transient failures with backoff. */
const embedTexts = async (texts: string[], attempt = 1): Promise<number[][]> => {
    try {
        const response = await openai.embeddings.create({ model: MODEL, input: texts });
        return response.data
            .sort((a, b) => a.index - b.index)
            .map((item) => item.embedding);
    } catch (error) {
        if (attempt >= 3) throw error;
        const delayMs = 1000 * 2 ** (attempt - 1);
        console.warn(`embed request failed (attempt ${attempt}), retrying in ${delayMs}ms`);
        await Bun.sleep(delayMs);
        return embedTexts(texts, attempt + 1);
    }
};

// # File prep and Chinking
const files = await readAllFiles(DATA_DIR);
const chunks: Chunk[] = files.flatMap((file) =>
    file.chunks.map((text) => ({ path: file.path, text, key: chunkKey(text) })),
);

const cache = await loadCache(CACHE_FILE);

// Chunks that still need an embedding: not cached, and deduped within this run.
const pending = new Map<string, string>();
for (const chunk of chunks) {
    if (!cache.has(chunk.key) && !pending.has(chunk.key)) {
        pending.set(chunk.key, chunk.text);
    }
}

console.log(`${chunks.length} chunks across ${files.length} files`);
console.log(`${cache.size} cached, ${pending.size} to embed`);

const pathByKey = new Map(chunks.map((chunk) => [chunk.key, chunk.path]));

if (pending.size > 0) {
    await mkdir(join(CACHE_FILE, ".."), { recursive: true });

    const entries = [...pending.entries()];
    for (let i = 0; i < entries.length; i += BATCH_SIZE) {
        const batch = entries.slice(i, i + BATCH_SIZE);
        const vectors = await embedTexts(batch.map(([, text]) => text));

        const lines: string[] = [];
        batch.forEach(([key, text], index) => {
            const vector = normalize(vectors[index] ?? []);
            cache.set(key, Float32Array.from(vector));

            const record: CacheRecord = {
                key,
                model: MODEL,
                path: pathByKey.get(key) ?? "",
                text,
                embedding: encodeVector(vector),
            };
            lines.push(JSON.stringify(record));
        });

        // Append per batch so partial progress survives a crash.
        await appendFile(CACHE_FILE, lines.join("\n") + "\n");
        console.log(`embedded ${Math.min(i + BATCH_SIZE, entries.length)}/${entries.length}`);
    }
}

const dims = cache.values().next().value?.length ?? 0;
console.log(`cache: ${CACHE_FILE} — ${cache.size} vectors x ${dims} dims`);
console.log(`ready: ${chunks.length} chunks, ${chunks.filter((c) => cache.has(c.key)).length} resolvable`);
