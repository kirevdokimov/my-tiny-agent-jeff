import OpenAI from 'openai';
import * as readline from "node:readline/promises";
import { NodeSDK } from "@opentelemetry/sdk-node";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { observeOpenAI } from "@langfuse/openai";
import { observe, propagateAttributes, startActiveObservation } from "@langfuse/tracing";
import { Logger } from './logger';

/** One id for the whole run: Langfuse session id + local JSONL log file. */
const sessionId = crypto.randomUUID();

const logger = await Logger.start(sessionId)

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

const BASE_URL = "https://openrouter.ai/api/v1"
const API_KEY = process.env.API_KEY
const MODEL = "deepseek/deepseek-v4.1-flash"

// Send traces to Langfuse via OpenTelemetry.
// Credentials come from LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_BASE_URL.
const langfuseSpanProcessor = new LangfuseSpanProcessor();
new NodeSDK({ spanProcessors: [langfuseSpanProcessor] }).start();

const openai = observeOpenAI(
    new OpenAI({
        baseURL: BASE_URL,
        apiKey: API_KEY,
    }),
    { generationName: "chat-completion" },
);

type ORChatMessage = OpenAI.Chat.Completions.ChatCompletion['choices'][number]['message'] & {
    reasoning_details?: unknown;
};

type Message = OpenAI.Chat.Completions.ChatCompletionMessageParam & { reasoning_details?: unknown }

const messages: Message[] = []



const bashToolDefinition: OpenAI.Chat.Completions.ChatCompletionTool = {
    type: "function",
    function: {
        name: "bash",
        description: "Executes shell command",
        parameters: {
            type: "object",
            properties: {
                command: {
                    type: "string",
                    description: "The shell command to run"
                },
                timeout_ms: {
                    type: "number",
                    description: "Optional timeout in milliseconds (default 10000)"
                },
            },
            required: ["command"],
            additionalProperties: false,
        }
    }
}

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
    bashToolDefinition
]

const bashToolExecute = async (command: string, timeoutMs: number = 10_000): Promise<string> => {
    // Guard against a missing/invalid timeout from the model: an undefined or 0
    // delay makes setTimeout fire almost immediately and SIGTERM the child.
    const effectiveTimeoutMs =
        Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : 10_000;

    // Spawn a process
    const proc = Bun.spawn(["bash", "-c", command], {
        cwd: process.cwd(),
        stdout: "pipe",
        stderr: "pipe",
        env: process.env,
    });

    const timer = setTimeout(() => proc.kill(), effectiveTimeoutMs);

    const [stdout, stderr] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
    ]);

    // Wait for process to finish
    const exitCode = await proc.exited;
    clearTimeout(timer);

    // Collect the output
    const out = [
        stdout.trim(),
        stderr.trim() ? `[stderr]\n${stderr.trim()}` : "",
        `[exit code: ${exitCode}]`,
    ].filter(Boolean).join("\n");

    // cap output so we don't blow the context window
    const MAX = 10_000;
    return out.length > MAX ? out.slice(0, MAX) + `\n...[truncated ${out.length - MAX}
 chars]` : out;

}

// Wrap the tool so each invocation is recorded as a "tool" observation.
const tracedBashToolExecute = observe(bashToolExecute, { name: "bash", asType: "tool" });

const oneTurn = async (askHuman: boolean): Promise<{ exit: true } | { askHuman: boolean }> => {

    // Most of the turns ask human the question, 
    // but some turns are needed after tool call with no user interruption
    if (askHuman) {
        const userMessage = await rl.question("user: ");

        if (userMessage === "exit") return { exit: true }

        const message: Message = {
            role: 'user',
            content: userMessage
        }

        messages.push(message)
        logger.trace(message)
    }

    const completion = await openai.chat.completions.create({
        model: MODEL,
        messages,
        tools,
    });

    const response = completion.choices[0]?.message as ORChatMessage

    const assistantMessage: Message = {
        role: 'assistant' as const,
        content: response.content,
        tool_calls: response.tool_calls,
        reasoning_details: response.reasoning_details, // Pass back unmodified
    }

    messages.push(assistantMessage)
    logger.trace(assistantMessage)

    // If no calls just out the response
    if (!response.tool_calls?.length) {
        console.log(`assistant: ${response.content}`)
        return { askHuman: true }
    }

    for (const call of response.tool_calls) {
        // Only one type is supported at the moment by SDK
        if (call.type !== "function") continue

        const { name, arguments: rawArgs } = call.function
        const args = JSON.parse(rawArgs)

        if (name === "bash" && args.command !== undefined) {
            const result = await tracedBashToolExecute(args.command, args.timeout_ms)
            console.log(`[tool call] bash ${rawArgs}. Response ${result}`)

            // Push tool response to the history
            const toolMessage: Message = {
                role: "tool",
                tool_call_id: call.id,
                content: result,
            }
            messages.push(toolMessage);
            logger.trace(toolMessage)
        }

        // run next turn without human
        return { askHuman: false }


    }
    return { askHuman: true };
}

const loop = async () => {
    let askHuman = true;
    while (true) {
        const endOfTurn = await oneTurn(askHuman)
        if ("exit" in endOfTurn) {
            return
        }
        askHuman = endOfTurn.askHuman
    }
}

async function main() {

    try {
        // Group the whole session under one trace; the LLM generations and bash
        // tool spans created inside are automatically nested under it.
        await propagateAttributes(
            {
                traceName: "jeff-agent",
                sessionId,
                tags: ["my-tiny-agent-jeff"],
            },
            async () => {
                await startActiveObservation("agent-loop", async (span) => {
                    await loop();
                    span.update({ output: "session ended" });
                });
            },
        );
    } finally {
        // The SDK batches events in the background; always flush before exiting,
        // even if the loop threw, otherwise pending spans are lost.
        await langfuseSpanProcessor.forceFlush();
        rl.close();
        logger.close()
    }
}

main();
