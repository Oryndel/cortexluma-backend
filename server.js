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
        systemInstruction 
    } = req.body;
    
    if (!history || history.length === 0) {
        res.write('Error: Chat history is required.');
        return res.end();
    }

    // Combine chat history and uploaded images into one contents array
    const contents = [...history];

    if (imageParts && imageParts.length > 0) {
        // Find the last user message and append the new image parts to it
        const lastUserIndex = contents.length - 1;
        if (contents[lastUserIndex] && contents[lastUserIndex].role === 'user') {
            contents[lastUserIndex].parts.push(...imageParts);
        } else {
            // This should not happen in a typical chat flow, but as a fallback:
            contents.push({ role: 'user', parts: imageParts });
        }
    }
    
    // Add the current user query (the last item in history)
    // The history array passed from the client should already contain the new user message.
    
    const safetySettings = [
        { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
        { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
        { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
        { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_MEDIUM_AND_ABOVE" }
    ];

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
    } catch (error) {
        console.error("Gemini API Error:", error);
        // Send a clearer error back to the client
        res.write(`Error: An internal API error occurred. Please check the backend's GEMINI_API_KEY and logs. Details: ${error.message}`);
    } finally {
        res.end(); 
    }
});

// -----------------------------------------------------------------
// IMAGE GENERATION ENDPOINT (Model: imagen-3.0-generate-002)
// This uses a direct fetch to the API as per required implementation instructions.
// -----------------------------------------------------------------
app.post('/api/generate-image', async (req, res) => {
    try {
        const { prompt } = req.body;
        if (!prompt) {
            // Respond with a 400 Bad Request if the prompt is missing
            return res.status(400).json({ error: 'Prompt is required for image generation.' });
        }

        // Configuration for the Imagen 3.0 API call
        const apiKey = process.env.GEMINI_API_KEY || "";
        const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key=${apiKey}`;

        const payload = { 
            instances: [{ prompt: prompt }], 
            parameters: { "sampleCount": 1 } 
        };

        // Retry mechanism for API calls (Exponential Backoff)
        let fetchResponse;
        const maxRetries = 3;
        let delay = 1000; 

        for (let i = 0; i < maxRetries; i++) {
            try {
                fetchResponse = await fetch(apiUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (fetchResponse.ok) {
                    break; // Success, exit loop
                } else if (fetchResponse.status === 429 || fetchResponse.status >= 500) {
                    // Too Many Requests or Server Error, retry after delay
                    if (i === maxRetries - 1) {
                         // Last attempt failed
                         throw new Error(`API failed after ${maxRetries} retries with status: ${fetchResponse.status}`);
                    }
                    console.warn(`Retry attempt ${i + 1} for Imagen API after status ${fetchResponse.status}. Delaying for ${delay}ms...`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    delay *= 2; // Exponential backoff
                } else {
                    // Non-retryable error (e.g., 400 Bad Request, 401 Unauthorized)
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
