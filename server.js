import express from 'express';
import cors from 'cors';
import 'dotenv/config'; 
import { GoogleGenAI } from "@google/genai";

// Initialize Gemini Client
// NOTE: Ensure your .env file has GEMINI_API_KEY="YOUR_KEY"
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const app = express();
const port = process.env.PORT || 3000;

// Increase limit for JSON body to allow for large Base64 image strings (up to 50MB)
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// -----------------------------------------------------------------
// 1. STREAMING CHAT ENDPOINT (Analysis: gemini-2.5-flash)
// Handles multi-turn text chat and multi-modal image analysis.
// -----------------------------------------------------------------
app.post('/api/stream-chat', async (req, res) => {
    // Set headers for streaming (Server-Sent Events configuration for plain text chunks)
    res.writeHead(200, {
        'Content-Type': 'text/plain; charset=utf-8',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Transfer-Encoding': 'chunked'
    });

    const { history, imagePart } = req.body;
    
    if (!history || history.length === 0) {
        res.write("Error: Conversation history is empty.");
        return res.end();
    }

    let contents = history; 
    
    // Check if an image was uploaded in this turn (Multi-modal)
    if (imagePart) {
        const lastUserMessage = contents.pop(); 
        const multiModalMessage = {
            role: "user",
            parts: [
                ...lastUserMessage.parts, 
                imagePart                  
            ]
        };
        contents.push(multiModalMessage);
    }
    
    try {
        const stream = await ai.models.generateContentStream({
            model: "gemini-2.5-flash", 
            contents: contents, 
        });

        for await (const chunk of stream) {
            const text = chunk.text;
            if (text) {
                res.write(text);
            }
        }
    } catch (error) {
        console.error("Gemini API Error:", error);
        res.write(`Error: An internal API error occurred: ${error.message}`);
    } finally {
        res.end(); 
    }
});

// -----------------------------------------------------------------
// 2. IMAGE GENERATION ENDPOINT (Creation: imagen-3.0-generate-002)
// This is a non-streaming endpoint for creating images.
// -----------------------------------------------------------------
app.post('/api/generate-image', async (req, res) => {
    const { prompt } = req.body;

    if (!prompt) {
        return res.status(400).send({ error: "Prompt is required for image generation." });
    }

    try {
        const response = await ai.models.generateImages({
            model: 'imagen-3.0-generate-002', // The dedicated image generation model
            prompt: prompt,
            config: {
                numberOfImages: 1,
                outputMimeType: 'image/jpeg',
                aspectRatio: '1:1', // You can change this to '16:9', '4:3', etc.
            }
        });

        const image = response.generatedImages[0];
        
        // Respond with the Base64 image data and mime type
        res.send({ 
            base64Image: image.image.imageBytes, 
            mimeType: image.image.mimeType 
        });
        
    } catch (error) {
        console.error("Imagen API Error:", error);
        res.status(500).send({ 
            error: "Image generation failed.", 
            details: error.message 
        });
    }
});

app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});
