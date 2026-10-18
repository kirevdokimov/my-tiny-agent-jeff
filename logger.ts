import type { FileHandle } from "node:fs/promises";
import { mkdir, open } from "node:fs/promises";
import { dirname } from "node:path";

const pad = (n: number) => String(n).padStart(2, "0");

/** YYYY-MM-DD-HHMM-mm */
function stamp(input: number | Date = Date.now()): string {
    const d = input instanceof Date ? input : new Date(input);
    return [
        d.getFullYear(),
        pad(d.getMonth() + 1),   // month is 0-based → +1
        pad(d.getDate()),
        `${pad(d.getHours())}${pad(d.getMinutes())}`, // HHMM
        pad(d.getMinutes()),     // -mm
    ].join("-");
}

/**
 * Like stamp(), but strictly increasing inside this process: clock steps and
 * same-millisecond calls still yield distinct values.
 */
let lastStamp = 0;
function stampUnique(input: number | Date = Date.now()): string {
    const now = input instanceof Date ? input.getTime() : input;
    lastStamp = now > lastStamp ? now : lastStamp + 1;
    return stamp(lastStamp);
}

/** Make any value serialize to exactly one line of JSON (a JSONL record). */
function toLine(value: unknown): string {
    let json: string | undefined;
    try {
        // No indentation: embedded newlines inside strings are escaped as \n,
        // so the record can never span multiple lines.
        json = JSON.stringify(value, replacer);
    } catch (error) {
        // Circular references, huge nesting, etc.
        json = JSON.stringify({ unserializable: true, reason: String(error) });
    }
    // JSON.stringify(undefined) / functions / symbols return undefined.
    return (json ?? JSON.stringify({ value: String(value) })) + "\n";
}

function replacer(_key: string, value: unknown): unknown {
    if (typeof value === "bigint") return value.toString();
    if (typeof value === "function") return `[Function ${value.name || "anonymous"}]`;
    if (value instanceof Error) {
        return {
            ...(value as unknown as Record<string, unknown>), // own props like `code`
            name: value.name,
            message: value.message,
            stack: value.stack,
            ...(value.cause !== undefined ? { cause: value.cause } : {}),
        };
    }
    return value;
}

export class Logger {
    /** Serializes writes so records never interleave and order is preserved. */
    private queue: Promise<void> = Promise.resolve();

    static async start(sessionId: string, fileName = `out/${stampUnique()}-${sessionId}.jsonl`) {
        await mkdir(dirname(fileName), { recursive: true });
        const fileHandle = await open(fileName, "a");
        const logger = new Logger(fileHandle, sessionId);
        logger.trace({ sessionId, event: "session-start" });
        return logger;
    }

    constructor(private fileHandle: FileHandle, public readonly sessionId: string) { }

    public trace = (object: unknown): Promise<void> => {
        const line = toLine(object);
        // Chain onto the previous write; swallow its rejection so one failure
        // doesn't poison every later record.
        const next = this.queue
            .catch(() => { })
            .then(() => this.fileHandle.write(line))
            .then(() => undefined);
        this.queue = next;
        return next;
    };

    public close = async () => {
        await this.queue.catch(() => { });
        await this.fileHandle.close();
    };
}