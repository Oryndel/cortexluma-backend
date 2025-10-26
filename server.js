import express from 'express';
import cors from 'cors';
import 'dotenv/config'; // To load GEMINI_API_KEY
import { GoogleGenAI } from "@google/genai";

// Initialize Gemini Client
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// New Streaming Endpoint (FASTEST, FULLY WORKING FEATURE)
app.post('/api/stream-chat', async (req, res) => {
    // Set headers for streaming (Server-Sent Events)
    res.writeHead(200, {
        'Content-Type': 'text/plain; charset=utf-8', // Plain text for streaming chunks
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
        // Use gemini-2.5-flash for speed (FASTER ANSWER)
        const stream = await ai.models.generateContentStream({
            model: "gemini-2.5-flash", 
            contents: history, // Send the full history array
        });

        // Read the stream and write chunks directly to the response
        for await (const chunk of stream) {
            // Ensure text exists before writing
            const text = chunk.text;
            if (text) {
                res.write(text);
            }
        }
    } catch (error) {
        console.error("Gemini API Error:", error);
        res.write(`Error: An internal API error occurred: ${error.message}`);
    } finally {
        // Close the response connection
        res.end(); 
    }
});

app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});
