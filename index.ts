import OpenAI from 'openai';
import * as readline from "node:readline/promises";

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

const BASE_URL = "https://openrouter.ai/api/v1"
const API_KEY = process.env.API_KEY
const MODEL = "deepseek/deepseek-v4.1-flash"

const openai = new OpenAI({
    baseURL: BASE_URL,
    apiKey: API_KEY,
});

type ORChatMessage = OpenAI.Chat.Completions.ChatCompletion['choices'][number]['message'] & {
    reasoning_details?: unknown;
};

const messages: (OpenAI.Chat.Completions.ChatCompletionMessageParam & { reasoning_details?: unknown })[] = []



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
                required: ["command"],
                additionalProperties: false,
            }
        }
    }
}

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
    bashToolDefinition
]

const bashToolExecute = async (command: string, timeoutMs: 10_000): Promise<string> => {
    // Spawn a process
    const proc = Bun.spawn(["bash", "-c", command], {
        cwd: process.cwd(),
        stdout: "pipe",
        stderr: "pipe",
        env: process.env,
    });

    const timer = setTimeout(() => proc.kill(), timeoutMs);

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

const oneTurn = async (askHuman: boolean): Promise<{ exit: true } | { askHuman: boolean }> => {

    // Most of the turns ask human the question, 
    // but some turns are needed after tool call with no user interruption
    if (askHuman) {
        const userMessage = await rl.question("user: ");

        if (userMessage === "exit") return { exit: true }

        messages.push({
            role: 'user',
            content: userMessage
        })
    }

    const completion = await openai.chat.completions.create({
        model: MODEL,
        messages,
        tools,
    });

    const response = completion.choices[0]?.message as ORChatMessage

    messages.push({
        role: 'assistant' as const,
        content: response.content,
        tool_calls: response.tool_calls,
        reasoning_details: response.reasoning_details, // Pass back unmodified
    })

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
            const result = await bashToolExecute(args.command, args.timeout_ms)
            console.log(`[tool call] bash ${rawArgs}`)

            // Push tool response to the history
            messages.push({
                role: "tool",
                tool_call_id: call.id,
                content: result,
            });
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

    await loop()

    rl.close();
}

main();
