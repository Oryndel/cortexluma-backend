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
// STREAMING CHAT ENDPOINT (Analysis: gemini-2.5-flash)
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

// NOTE: The /api/generate-image endpoint has been completely removed.

app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});
