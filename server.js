import express from 'express';
import cors from 'cors';
import 'dotenv/config'; 
import { GoogleGenAI } from "@google/genai";
import fetch from 'node-fetch'; // Required for making external API calls like Imagen

// Initialize Gemini Client
// NOTE: Ensure your .env file has GEMINI_API_KEY="YOUR_KEY"
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const app = express();
const port = process.env.PORT || 3000;

// Increase limit for JSON body to allow for large Base64 image strings (up to 50MB)
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// -----------------------------------------------------------------
// HEALTH CHECK / ROOT ENDPOINT (NEW)
// -----------------------------------------------------------------
app.get('/', (req, res) => {
    // This route responds to the browser/Render health check on the base URL.
    res.status(200).json({
        status: 'OK',
        message: 'CortexLuma AI Backend is running. Use /api/stream-chat or /api/generate-image via POST.',
        version: '1.0.0'
    });
});

// -----------------------------------------------------------------
// STREAMING CHAT ENDPOINT (Analysis: gemini-2.5-flash with Google Search)
// Handles multi-turn chat, multi-image analysis, and configurable settings.
// -----------------------------------------------------------------
app.post('/api/stream-chat', async (req, res) => {
    // Set headers for streaming (Server-Sent Events configuration for plain text chunks)
    res.writeHead(200, {
        'Content-Type': 'text/plain; charset=utf-8',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Transfer-Encoding': 'chunked'
    });

    // Destructure new config parameters and multiple images
    const { 
        history, 
        imageParts, 
        temperature, 
        maxOutputTokens, 
        systemInstruction,
        safetySettings
    } = req.body;
    
    if (!history || history.length === 0) {
        res.write('Error: Chat history is required.');
        return res.end();
    }
    
    // The last part of the history must be the user's current prompt (part of the last turn)
    const lastUserTurn = history[history.length - 1];
    
    if (lastUserTurn.role !== 'user' || !lastUserTurn.parts || lastUserTurn.parts.length === 0) {
        res.write('Error: Invalid last chat turn format.');
        return res.end();
    }

    // Combine history parts and new image parts for the API call
    const contents = [...history];

    // Append image parts to the last user turn if present
    if (imageParts && imageParts.length > 0) {
        contents[contents.length - 1].parts = [
            ...contents[contents.length - 1].parts,
            ...imageParts
        ];
    }
    
    let delay = 1000; // Initial delay of 1 second
    const maxRetries = 3;

    for (let i = 0; i < maxRetries; i++) {
        try {
            const stream = await ai.models.generateContentStream({
                model: "gemini-2.5-flash", 
                contents: contents, 
                config: {
                    // FEATURE: Google Search grounding
                    tools: [{ googleSearch: {} }], 
                    
                    // FEATURE: Temperature (Creativity)
                    temperature: temperature !== undefined ? parseFloat(temperature) : 0.7, 
                    
                    // FEATURE: Max Output Tokens
                    maxOutputTokens: maxOutputTokens !== undefined ? parseInt(maxOutputTokens) : 2048,
                    
                    // FEATURE: System Instruction/Personality
                    systemInstruction: systemInstruction || "You are CortexLuma, a powerful, witty, and highly helpful AI assistant built by Google. You excel at real-time data retrieval and coding tasks. Keep your answers concise, clear, and engaging.",
                    
                    // FEATURE: Safety Settings
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
                // Send a clearer error back to the client on final failure
                res.write(`Error: An internal API error occurred after ${maxRetries} attempts. Details: ${error.message}`);
                break;
            }
            // Wait and retry
            await new Promise(resolve => setTimeout(resolve, delay));
            delay *= 2; 
        }
    }

    res.end(); 
});


// -----------------------------------------------------------------
// IMAGE GENERATION ENDPOINT (Analysis: Imagen 3.0 via HTTP API)
// -----------------------------------------------------------------
app.post('/api/generate-image', async (req, res) => {
    // This is for the Imagen model which requires an external API call
    const IMAGEN_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key=${process.env.GEMINI_API_KEY}`;
    
    // Note: The GEMINI_API_KEY is used for Imagen 3.0 via the HTTP API.
    if (!process.env.GEMINI_API_KEY) {
        return res.status(500).json({ error: 'GEMINI_API_KEY not set in environment variables.' });
    }

    const { prompt } = req.body;
    if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required for image generation.' });
    }

    // Imagen 3.0 parameters
    const payload = { 
        instances: { prompt: prompt }, 
        parameters: { 
            "sampleCount": 1,
            "outputMimeType": "image/png",
            "aspectRatio": "1:1" // Default to square
        } 
    };
    
    let delay = 1000; // Initial delay of 1 second
    const maxRetries = 3;
    let fetchResponse;

    try {
        for (let i = 0; i < maxRetries; i++) {
            try {
                fetchResponse = await fetch(IMAGEN_API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (fetchResponse.ok) {
                    break; // Success, exit retry loop
                }

                // If HTTP status is not 200, throw to trigger backoff/retry
                if (i === maxRetries - 1) {
                    const errorBody = await fetchResponse.json();
                    throw new Error(`External Image API Error (${fetchResponse.status}): ${errorBody.error?.message || 'Unknown error'}`);
                }
            } catch (error) {
                if (i === maxRetries - 1) throw error; // Re-throw if it's the last attempt
                // Non-HTTP errors (e.g., network issues) also trigger backoff
                await new Promise(resolve => setTimeout(resolve, delay));
                delay *= 2; 
            }
        }
        
        const result = await fetchResponse.json();
        
        // Extract the base64 data as per instruction
        const base64Data = result?.predictions?.[0]?.bytesBase64Encoded;

        if (base64Data) {
            // Respond with the base64 data wrapped in a data URL as JSON
            const imageUrl = `data:image/png;base64,${base64Data}`;
            res.json({ imageUrl: imageUrl });
        } else {
            // Log full API response if data is missing for debugging
            console.error('Image generation response missing data:', result);
            res.status(500).json({ error: 'Image generation failed: No image data received.' });
        }

    } catch (error) {
        console.error('Image Generation Backend Error:', error.message);
        // Ensure the response is always JSON for the frontend to handle
        res.status(500).json({ error: `Image generation failed: ${error.message}` });
    }
});


app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});
