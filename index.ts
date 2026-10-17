import OpenAI from 'openai';

const BASE_URL = "https://openrouter.ai/api/v1"
const API_KEY = process.env.API_KEY
const MODEL = "deepseek/deepseek-v4.1-flash"

const openai = new OpenAI({
    baseURL: BASE_URL,
    apiKey: API_KEY,
});

async function main() {
    const completion = await openai.chat.completions.create({
        model: MODEL,
        messages: [
            {
                role: 'user',
                content: 'Who is Marie Curie?',
            },
        ],
    });

    // Extract the assistant message with reasoning_details
    type ORChatMessage = (typeof completion)['choices'][number]['message'] & {
        reasoning_details?: unknown;
    };
    const response = completion.choices[0]?.message as ORChatMessage;


    console.log(response);
}

main();
