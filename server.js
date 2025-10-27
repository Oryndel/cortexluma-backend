import express from 'express';
import cors from 'cors';
import 'dotenv/config'; 
import { GoogleGenAI } from "@google/genai";
import fetch from 'node-fetch'; // Required for calling Freepik API

// --- Initialization ---
// NOTE: Ensure your .env file has GEMINI_API_KEY="YOUR_KEY"
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const app = express();
const port = process.env.PORT || 3000;

// Increase limit for JSON body to allow for large Base64 image strings (up to 50MB)
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// -----------------------------------------------------------------
// 1. GEMINI STREAMING CHAT ENDPOINT
// Handles multi-turn chat, multi-image analysis, and configurable settings (System Instruction, Temp, Max Tokens).
// -----------------------------------------------------------------
app.post('/api/stream-chat', async (req, res) => {
    // Set headers for streaming (Server-Sent Events configuration for plain text chunks)
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
        systemInstruction 
    } = req.body;
    
    if (!history || history.length === 0) {
        res.write('Error: The chat history cannot be empty.');
        return res.end();
    }
    
    // Combine history and new image parts for the contents array
    const contents = [...history];

    if (imageParts && imageParts.length > 0) {
        // Find the last user message and append the new image parts to it
        const lastUserMessage = contents[contents.length - 1];
        if (lastUserMessage && lastUserMessage.role === 'user') {
            // NOTE: The frontend should already include the text part of the last message
            lastUserMessage.parts.push(...imageParts);
        } else {
            // Fallback: This should ideally not happen if history is structured correctly
            contents.push({ role: 'user', parts: imageParts });
        }
    }
    
    // Define standard safety settings
    const safetySettings = [
        { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
        { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
        { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
        { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
    ];


    try {
        // Start streaming response from Gemini
        const stream = await ai.models.generateContentStream({
            model: "gemini-2.5-flash", 
            contents: contents, 
            config: {
                tools: [{ googleSearch: {} }], // Enable Google Search grounding
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
    } catch (error) {
        console.error("Gemini API Error:", error);
        // Send a clearer error back to the client
        res.write(`Error: An internal API error occurred. Please check the backend's GEMINI_API_KEY and logs. Details: ${error.message}`);
    } finally {
        res.end(); 
    }
});


// -----------------------------------------------------------------
// 2. FREEPIK AI IMAGE GENERATION ENDPOINT
// Handles the /image command from the frontend.
// -----------------------------------------------------------------
app.post('/api/generate-image', async (req, res) => {
    const { prompt } = req.body;
    
    if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required' });
    }

    try {
        const FREEPIK_API_KEY = process.env.FREEPIK_API_KEY; 
        
        if (!FREEPIK_API_KEY) {
            return res.status(500).json({ error: 'Freepik API key is missing. Please set FREEPIK_API_KEY in your .env file.' });
        }

        const response = await fetch('https://api.freepik.com/v1/ai/text-to-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-freepik-api-key': FREEPIK_API_KEY, 
            },
            body: JSON.stringify({
                prompt: prompt,
                // Recommended parameters for high quality
                negative_prompt: "text, watermark, poor quality, bad anatomy, ugly, cartoon, blurry, low resolution, unrefined",
                image: { size: "widescreen_16_9" }, // Landscape ratio
                styling: { style: "photorealistic" }, 
                num_images: 1
            })
        });

        const data = await response.json();

        if (response.ok && data.results && data.results.length > 0) {
            // Freepik returns an array of results, we take the first image's URL
            const imageUrl = data.results[0].image_url; 
            res.json({ imageUrl });
        } else {
            console.error("Freepik API Error Response:", data);
            res.status(500).json({ error: data.message || 'Freepik image generation failed. Check backend logs for details.' });
        }

    } catch (error) {
        console.error('Freepik Backend Error:', error);
        res.status(500).json({ error: 'Internal server error while calling Freepik API.' });
    }
});

app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});
