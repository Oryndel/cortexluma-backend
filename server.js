import express from 'express';
import cors from 'cors';
import 'dotenv/config'; // To load environment variables (like GEMINI_API_KEY)
import { GoogleGenAI } from "@google/genai";

// Initialize Gemini Client
// NOTE: Make sure you have a .env file with GEMINI_API_KEY="YOUR_KEY"
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Streaming Endpoint: Handles multi-turn chat history
app.post('/api/stream-chat', async (req, res) => {
    // Set headers for streaming (Server-Sent Events configuration)
    res.writeHead(200, {
        'Content-Type': 'text/plain; charset=utf-8',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Transfer-Encoding': 'chunked'
    });

    const { history } = req.body;
    
    if (!history || history.length === 0) {
        res.write("Error: Conversation history is empty.");
        return res.end();
    }

    try {
        // Use gemini-2.5-flash for high speed and efficient streaming
        const stream = await ai.models.generateContentStream({
            model: "gemini-2.5-flash", 
            contents: history, // Sends the full history array for multi-turn context
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
        // Ensure error is sent to the frontend for display
        res.write(`Error: An internal API error occurred: ${error.message}`);
    } finally {
        // Close the response connection
        res.end(); 
    }
});

app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});
