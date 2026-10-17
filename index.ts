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


const loop = async () => {
    while (true) {
        const userMessage = await rl.question("user: ");

        if (userMessage === "exit") return

        messages.push({
            role: 'user',
            content: userMessage
        })

        const completion = await openai.chat.completions.create({
            model: MODEL,
            messages,
        });

        const response = completion.choices[0]?.message as ORChatMessage

        messages.push({
            role: 'assistant' as const,
            content: response.content,
            reasoning_details: response.reasoning_details, // Pass back unmodified
        })

        console.log(`assistant: ${response.content}`)
    }
}

async function main() {

    await loop()
    
    rl.close();
}

main();
