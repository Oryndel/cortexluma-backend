import express from 'express';
import cors from 'cors';
import 'dotenv/config'; 
import fetch from 'node-fetch'; 
import { GoogleGenAI } from "@google/genai";

// --- Configuration and Initialization ---

// CRITICAL CHECK: Ensure API Key is available
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
    console.error("FATAL ERROR: GEMINI_API_KEY is missing. Please ensure your environment variable is set.");
    // The server will proceed but API calls will fail immediately.
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });
const app = express();
const port = process.env.PORT || 3000;

// Set hard limits for fast response (Fail-Fast Validation)
const MAX_HISTORY_LENGTH = 50; // Max turns of history
const MAX_PROMPT_LENGTH = 15000; // Max characters in the current turn

// Middleware Setup
// FIX: Explicitly set CORS origin to allow your frontend URL (GitHub Pages) and local testing
app.use(cors({
    origin: ['https://oryndel.github.io', 'http://localhost:3000', 'https://oryndel.github.io/cortexluma'],
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type'],
}));
app.use(express.json({ limit: '50mb' }));

// -----------------------------------------------------------------
// HEALTH CHECK / ROOT ENDPOINT
// -----------------------------------------------------------------
app.get('/', (req, res) => {
    res.status(200).json({
        status: 'OK',
        message: 'CortexLuma AI Backend is running. Use /api/stream-chat or /api/generate-image via POST.',
        version: '1.0.2'
    });
});

// -----------------------------------------------------------------
// STREAMING CHAT ENDPOINT
// -----------------------------------------------------------------
app.post('/api/stream-chat', async (req, res) => {
    // Destructure all fields from the frontend payload
    const { history, prompt, imageParts, temperature, maxOutputTokens, systemInstruction } = req.body; 

    // 1. Fail-Fast Validation
    if (!GEMINI_API_KEY) {
        return res.status(500).send("Error: API Key not configured on the server.");
    }
    if (!history || !Array.isArray(history) || typeof prompt !== 'string') {
        return res.status(400).send('Invalid request: `history` and `prompt` are required.');
    }
    if (history.length > MAX_HISTORY_LENGTH || prompt.length > MAX_PROMPT_LENGTH) {
        return res.status(400).send('Request exceeds maximum length limit.');
    }
    
    // Set headers for streaming response
    res.writeHead(200, {
        'Content-Type': 'text/plain; charset=utf-8',
        'Transfer-Encoding': 'chunked',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    });

    try {
        const contents = [...history]; 

        // Add the current user prompt (and optional image(s))
        const userParts = [{ text: prompt }];
        
        // Handle imageParts array (multi-image support)
        if (imageParts && Array.isArray(imageParts)) {
            imageParts.forEach(part => {
                if (part.inlineData) {
                    userParts.push(part.inlineData); // Push the inlineData part
                }
            });
        }
        contents.push({ role: 'user', parts: userParts });

        // Configuration for fast streaming and grounding
        const model = 'gemini-2.5-flash';
        
        // Use dynamic settings from the request body
        const defaultInstruction = "You are CortexLuma, a helpful and concise AI assistant. Answer the user's questions clearly, and provide relevant sources if using Google Search grounding.";
        
        const generationConfig = {
            // Use systemInstruction from the body, falling back to the default
            systemInstruction: systemInstruction || defaultInstruction, 
            // Use temperature from the body, falling back to 0.2
            temperature: temperature !== undefined ? parseFloat(temperature) : 0.2, 
            // Use maxOutputTokens from the body, or undefined to use model default
            maxOutputTokens: maxOutputTokens !== undefined ? parseInt(maxOutputTokens) : undefined 
        };

        const responseStream = await ai.models.generateContentStream({
            model: model,
            contents: contents,
            config: {
                ...generationConfig,
                tools: [{ google_search: {} }] // Enable Google Search grounding
            }
        });

        // Stream the response chunks directly to the client
        for await (const chunk of responseStream) {
            if (chunk.text) {
                // Write the text content
                res.write(chunk.text);
            }
            
            // Collect grounding sources from the final chunk
            if (chunk.groundingMetadata?.groundingAttributions) {
                const sources = chunk.groundingMetadata.groundingAttributions.map(attr => ({
                    uri: attr.web?.uri,
                    title: attr.web?.title
                })).filter(source => source.uri); // Filter out any incomplete sources

                // Send a special marker for the end of the text and sources
                res.write(`\n--SOURCES--\n${JSON.stringify(sources)}`);
            }
        }
        
        // End the response stream
        res.end();

    } catch (error) {
        console.error('Chat Streaming Backend Error:', error.message);
        // Send the error message back to the client as plain text before ending the stream
        res.write(`\n--ERROR--\n${error.message}`);
        res.end();
    }
});


// -----------------------------------------------------------------
// IMAGE GENERATION ENDPOINT (No changes needed, but included for completeness)
// -----------------------------------------------------------------
const IMAGEN_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict";

app.post('/api/generate-image', async (req, res) => {
    const { prompt } = req.body;

    if (!GEMINI_API_KEY) {
        return res.status(500).json({ error: "Error: API Key not configured on the server." });
    }
    if (!prompt || typeof prompt !== 'string' || prompt.length < 5) {
        return res.status(400).json({ error: 'Invalid prompt. Please provide a descriptive prompt (min 5 chars).' });
    }
    
    // Construct the payload for the Imagen API
    const payload = {
        instances: {
            prompt: prompt
        },
        parameters: {
            "sampleCount": 1,
            "aspectRatio": "1:1" 
        } 
    };
    
    try {
        const fetchResponse = await fetch(IMAGEN_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Goog-Api-Key': GEMINI_API_KEY },
            body: JSON.stringify(payload)
        });

        if (!fetchResponse.ok) {
            const errorBody = await fetchResponse.json();
            console.error('External Image API Error Response:', errorBody);
            throw new Error(`External Image API Error (${fetchResponse.status}): ${errorBody.error?.message || 'Unknown error from Imagen API'}`);
        }
        
        const result = await fetchResponse.json();
        const base64Data = result?.predictions?.[0]?.bytesBase64Encoded;

        if (base64Data) {
            const imageUrl = `data:image/png;base64,${base64Data}`;
            res.json({ imageUrl: imageUrl });
        } else {
            console.error('Image generation response missing data:', result);
            res.status(500).json({ error: 'Image generation failed: No image data received.' });
        }

    } catch (error) {
        console.error('Image Generation Backend Error:', error.message);
        res.status(500).json({ error: `Image generation failed: ${error.message}` });
    }
});


app.listen(port, () => {
    console.log(`CortexLuma Backend listening on port ${port}`);
});
