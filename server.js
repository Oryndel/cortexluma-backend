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
// STREAMING CHAT ENDPOINT (Analysis: gemini-2.5-flash with Google Search)
// Optimized for maximum streaming speed and aggressive retry.
// -----------------------------------------------------------------
app.post('/api/stream-chat', async (req, res) => {
    res.writeHead(200, {
        'Content-Type': 'text/plain; charset=utf-8',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Transfer-Encoding': 'chunked'
    });

    const { 
        history, 
        imageParts, 
        temperature, 
        maxOutputTokens, 
        systemInstruction,
        safetySettings
    } = req.body;
    
    // FAIL-FAST VALIDATION 
    if (!history || history.length === 0) {
        // FIX: Added a check for history.length > 0
        res.write('Error: Chat history is required.');
        return res.end();
    }
    if (history.length > MAX_HISTORY_LENGTH) {
        res.write(`Error: History is too long (Max ${MAX_HISTORY_LENGTH} turns).`);
        return res.end();
    }

    const lastUserTurn = history[history.length - 1];
    // FIX: Ensure lastUserTurn and its parts exist before accessing properties
    if (!lastUserTurn || !lastUserTurn.parts || lastUserTurn.parts.length === 0) {
         res.write('Error: Last user turn is incomplete.');
         return res.end();
    }
    
    const userPromptText = lastUserTurn.parts.find(p => p.text)?.text || '';
    if (userPromptText.length > MAX_TOKEN_ESTIMATE) {
        res.write(`Error: Prompt is too long (Max ${MAX_TOKEN_ESTIMATE} characters).`);
        return res.end();
    }

    // Combine history parts and new image parts for the API call
    const contents = [...history];

    // Append image parts to the last user turn if present
    // NOTE: The client sends the user prompt as the last item of history AND the imageParts separately.
    // The client-side logic updates the history *before* sending. We just need to ensure the imageParts 
    // are correctly merged into the last user turn (which already contains the text part from history).
    if (imageParts && imageParts.length > 0) {
        // The last history item is the user's turn with the text part.
        // We append the new image parts to this last item's parts array.
        contents[contents.length - 1].parts = [
            ...contents[contents.length - 1].parts,
            // FIX: Ensure the imageParts array is spread correctly
            ...imageParts 
        ];
    }
    
    // AGGRESSIVE RETRY LOGIC (2 attempts, no backoff delay)
    const maxRetries = 2;

    for (let i = 0; i < maxRetries; i++) {
        try {
            const stream = await ai.models.generateContentStream({
                model: "gemini-2.5-flash", 
                contents: contents, 
                config: {
                    tools: [{ googleSearch: {} }], 
                    temperature: temperature !== undefined ? parseFloat(temperature) : 0.7, 
                    maxOutputTokens: maxOutputTokens !== undefined ? parseInt(maxOutputTokens) : 2048,
                    systemInstruction: systemInstruction || "You are CortexLuma, a powerful, witty, and highly helpful AI assistant built by Google. You excel at real-time data retrieval and coding tasks. Keep your answers concise, clear, and engaging.",
                    safetySettings: safetySettings
                }
            });

            for await (const chunk of stream) {
                const text = chunk.text;
                if (text) {
                    res.write(text);
                }
            }
            break; // Success, break the retry loop

        } catch (error) {
            console.error(`Attempt ${i + 1}: Gemini API Error:`, error);
            if (i === maxRetries - 1) {
                // Final error message on failure
                res.write(`Critical Error: AI API failed after ${maxRetries} attempts. Details: ${error.message}`);
                break;
            }
            // Immediate retry for speed (no delay)
        }
    }

    res.end(); 
});


// -----------------------------------------------------------------
// IMAGE GENERATION ENDPOINT (Analysis: Imagen 3.0 via HTTP API)
// Optimized for FAIL-FAST processing.
// -----------------------------------------------------------------
app.post('/api/generate-image', async (req, res) => {
    const IMAGEN_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key=${process.env.GEMINI_API_KEY}`;
    
    if (!process.env.GEMINI_API_KEY) {
        return res.status(500).json({ error: 'GEMINI_API_KEY not set in environment variables.' });
    }

    const { prompt } = req.body;
    if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required for image generation.' });
    }

    const payload = { 
        instances: { prompt: prompt }, 
        parameters: { 
            "sampleCount": 1,
            "outputMimeType": "image/png",
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
            // FIX: Log the more detailed error body from the Imagen API
            console.error('External Imagen API returned non-OK status. Error body:', errorBody); 
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
