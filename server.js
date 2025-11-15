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
        version: '1.0.2'
    });
});

// -----------------------------------------------------------------
// STREAMING CHAT ENDPOINT (Analysis: gemini-2.5-flash with Google Search)
// Optimized for maximum streaming speed and aggressive retry on initial connection.
// -----------------------------------------------------------------
app.post('/api/stream-chat', async (req, res) => {
    // 1. Initial validation
    const { history, prompt, base64Image } = req.body;

    if (!prompt) {
        return res.status(400).json({ error: "Prompt is required." });
    }

    if (prompt.length > MAX_PROMPT_LENGTH) {
        return res.status(400).json({ error: `Prompt exceeds maximum length of ${MAX_PROMPT_LENGTH} characters.` });
    }

    // 2. Construct chat history
    const chatHistory = history ? [...history].slice(-MAX_HISTORY_LENGTH) : [];

    // Add current user prompt (and image, if provided)
    const userParts = [{ text: prompt }];

    if (base64Image) {
        try {
            // base64Image is expected to be a data URL (e.g., 'data:image/jpeg;base64,...')
            const [mimeTypePart, base64Data] = base64Image.split(';base64,');
            const mimeType = mimeTypePart.split(':')[1];
            
            if (!mimeType || !base64Data) {
                throw new Error("Invalid base64 image format.");
            }

            userParts.push({
                inlineData: {
                    mimeType: mimeType,
                    data: base64Data
                }
            });
        } catch (e) {
            console.error("Image parsing error:", e.message);
            return res.status(400).json({ error: "Invalid image data provided." });
        }
    }

    chatHistory.push({ role: "user", parts: userParts });

    // 3. Configure response headers for streaming
    res.setHeader('Content-Type', 'text/plain'); // Sending raw text chunks
    res.setHeader('Transfer-Encoding', 'chunked');
    res.setHeader('Cache-Control', 'no-cache');
    res.status(200);

    const MAX_RETRIES = 3;
    let retries = 0;
    let responseStream;

    while (retries < MAX_RETRIES) {
        try {
            // 4. Call Gemini API with Exponential Backoff
            responseStream = await ai.models.generateContentStream({
                model: "gemini-2.5-flash",
                contents: chatHistory,
                config: {
                    // Enable Google Search grounding tool
                    tools: [{ googleSearch: {} }],
                },
            });

            // If successful, break the retry loop
            break; 

        } catch (error) {
            retries++;
            if (retries >= MAX_RETRIES) {
                console.error(`FATAL: Gemini API call failed after ${MAX_RETRIES} retries.`, error.message);
                // Clean up response stream and close connection on fatal error
                res.end(`[STREAM_ERROR] Gemini API failed: ${error.message}`);
                return;
            }
            // Implement exponential backoff delay
            const delay = Math.pow(2, retries) * 1000;
            console.warn(`Gemini API connection failed (Attempt ${retries}/${MAX_RETRIES}). Retrying in ${delay}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }

    if (!responseStream) {
        // This case should not happen if the retry loop is correctly structured, but acts as a safeguard.
        return res.end("[STREAM_ERROR] Could not initialize Gemini stream.");
    }
    
    // 5. Stream the response chunks to the client
    try {
        for await (const chunk of responseStream) {
            if (chunk.text) {
                res.write(chunk.text);
            }

            // Stream citations when available (only included in the final chunk)
            if (chunk.candidates?.[0]?.groundingMetadata?.groundingAttributions) {
                const attributions = chunk.candidates[0].groundingMetadata.groundingAttributions;
                const sources = attributions.map(attr => ({
                    uri: attr.web?.uri,
                    title: attr.web?.title,
                })).filter(source => source.uri && source.title);

                if (sources.length > 0) {
                    // Send a special JSON marker for the frontend to parse
                    // The frontend must parse any line starting with [METADATA]
                    res.write(`\n[METADATA]${JSON.stringify({ sources: sources })}`);
                }
            }
        }
    } catch (error) {
        console.error("Error during streaming:", error.message);
        res.end(`\n[STREAM_ERROR] Streaming connection lost: ${error.message}`);
    } finally {
        // 6. End the response stream
        res.end();
    }
});


// -----------------------------------------------------------------
// IMAGE GENERATION ENDPOINT (Imagen 4.0)
// Uses node-fetch to call the external service
// -----------------------------------------------------------------
const IMAGEN_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict";

app.post('/api/generate-image', async (req, res) => {
    const { prompt } = req.body;

    if (!prompt) {
        return res.status(400).json({ error: "Prompt is required for image generation." });
    }

    const payload = {
        instances: {
            prompt: prompt
        },
        parameters: {
            "sampleCount": 1,
            "aspectRatio": "1:1" 
        } 
    };
    
    // NOTE: This endpoint uses a single, non-retried fetch call for fast feedback.
    try {
        const fetchResponse = await fetch(IMAGEN_API_URL, {
            method: 'POST',
            // CRITICAL: We pass the API key via a header for Imagen API
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
    console.log(`Server listening on port ${port} (http://localhost:${port})`);
});
