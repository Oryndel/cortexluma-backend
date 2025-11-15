import express from 'express';
import cors from 'cors';
import 'dotenv/config'; 
import { GoogleGenAI } from "@google/genai";
import fetch from 'node-fetch';

// Initialize Gemini Client
// NOTE: Ensure your .env file has GEMINI_API_KEY="YOUR_KEY"
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const app = express();
const port = process.env.PORT || 3000;

// Set hard limits for fast response (Fail-Fast Validation)
const MAX_HISTORY_LENGTH = 50; // Max turns of history
const MAX_TOKEN_ESTIMATE = 10000; // Max characters/tokens in the current turn

// Increase limit for JSON body to allow for large Base64 image strings (up to 50MB)
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// -----------------------------------------------------------------
// HEALTH CHECK / ROOT ENDPOINT
// -----------------------------------------------------------------
app.get('/', (req, res) => {
    res.status(200).json({
        status: 'OK',
        message: 'CortexLuma AI Backend is running. Use /api/stream-chat or /api/generate-image via POST.',
        version: '1.0.1'
    });
});

// -----------------------------------------------------------------
// STREAMING CHAT ENDPOINT (Optimized for Maximum Speed - Fail Fast)
// Removed all exponential backoff and retry logic to ensure minimum latency.
// -----------------------------------------------------------------
app.post('/api/stream-chat', async (req, res) => {
    const { history, prompt, image, user_id } = req.body;

    // 1. Fail-Fast Validation
    if (!history || !Array.isArray(history) || typeof prompt !== 'string') {
        return res.status(400).json({ error: 'Invalid request: `history` and `prompt` are required.' });
    }
    if (history.length > MAX_HISTORY_LENGTH || prompt.length > MAX_TOKEN_ESTIMATE) {
        return res.status(400).json({ error: 'Request exceeds maximum history or prompt length limit.' });
    }
    
    // Set headers for streaming response
    res.writeHead(200, {
        'Content-Type': 'text/plain; charset=utf-8',
        'Transfer-Encoding': 'chunked',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    });

    try {
        const contents = [...history]; // Copy existing history

        // Add the current user prompt (and optional image)
        const userParts = [{ text: prompt }];
        if (image && image.data && image.mimeType) {
            // Add image part for multimodal requests
            userParts.push({ inlineData: { data: image.data, mimeType: image.mimeType } });
        }
        contents.push({ role: 'user', parts: userParts });

        // Configuration for fast streaming and grounding
        const model = 'gemini-2.5-flash';
        const generationConfig = {
            systemInstruction: "You are CortexLuma, a helpful and concise AI assistant. Answer the user's questions clearly, and provide relevant sources if using Google Search grounding.",
            temperature: 0.2
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
                res.write(chunk.text);
            }
            
            // Collect grounding sources from the final chunk
            if (chunk.groundingMetadata?.groundingAttributions) {
                const sources = chunk.groundingMetadata.groundingAttributions.map(attr => ({
                    uri: attr.web.uri,
                    title: attr.web.title
                }));
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
// IMAGE GENERATION ENDPOINT (Analysis: imagen-4.0-generate-001)
// Already configured for MAX SPEED (Fail-Fast)
// -----------------------------------------------------------------
const IMAGEN_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict";

app.post('/api/generate-image', async (req, res) => {
    const { prompt } = req.body;

    if (!prompt || typeof prompt !== 'string' || prompt.length < 5) {
        return res.status(400).json({ error: 'Invalid prompt. Please provide a descriptive prompt.' });
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
    
    // MAX SPEED: Removed the exponential backoff retry loop. Fail fast and directly.
    try {
        const fetchResponse = await fetch(IMAGEN_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!fetchResponse.ok) {
            // Throw a specific error if the external API returns a non-200 status
            const errorBody = await fetchResponse.json();
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
        // Use proper status code for internal/external errors
        res.status(500).json({ error: `Image generation failed: ${error.message}` });
    }
});


app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});
