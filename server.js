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
app.use(express.json({ limit: '200mb' }));

// -----------------------------------------------------------------
// STREAMING TEXT/MULTI-MODAL CHAT ENDPOINT (Uses gemini-2.5-flash)
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

    // Start with the existing conversation history
    let contents = history; 
    
    // Check if an image was uploaded in this turn
    if (imagePart) {
        // The last message in the history array is the new user text prompt.
        // We pop it to modify it and add the image part to make it multi-modal.
        const lastUserMessage = contents.pop(); 
        
        const multiModalMessage = {
            role: "user",
            parts: [
                ...lastUserMessage.parts, // The text prompt (e.g., "What is this?")
                imagePart                  // The Base64 image data
            ]
        };
        // Push the combined multi-modal message back into the contents array
        contents.push(multiModalMessage);
    }
    
    try {
        const stream = await ai.models.generateContentStream({
            model: "gemini-2.5-flash", 
            contents: contents, // Sends the modified history (with or without image)
        });

        // Read the stream and write chunks directly to the response
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
        // Ensure the response is closed after the stream is finished or an error occurs
        res.end(); 
    }
});

app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});
