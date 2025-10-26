import express from 'express';
import cors from 'cors';
import 'dotenv/config'; 
import { GoogleGenAI } from "@google/genai";

// Initialize Gemini Client
// NOTE: Ensure your .env file has GEMINI_API_KEY="YOUR_KEY"
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// -----------------------------------------------------------------
// 1. STREAMING TEXT CHAT ENDPOINT (Uses gemini-2.5-flash)
// -----------------------------------------------------------------
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
        res.write(`Error: An internal API error occurred: ${error.message}`);
    } finally {
        res.end(); 
    }
});

// -----------------------------------------------------------------
// 2. IMAGE GENERATION ENDPOINT (Uses imagen-3.0-generate-002)
// -----------------------------------------------------------------
app.post('/api/generate-image', async (req, res) => {
    const { prompt } = req.body;

    if (!prompt) {
        return res.status(400).json({ error: "Image prompt is required." });
    }

    try {
        const response = await ai.models.generateImages({
            model: "imagen-3.0-generate-002", // Dedicated, high-quality Imagen model
            prompt: prompt,
            config: {
                numberOfImages: 1,
                aspectRatio: "1:1", // Square is standard, other options: "16:9", "4:3", etc.
                outputMimeType: "image/png" // Ensure the output is PNG
            },
        });

        // Extract Base64 image data and MIME type
        const firstImage = response.generatedImages[0];
        const base64Image = firstImage.image.imageBytes; 
        const mimeType = firstImage.image.mimeType;

        res.json({
            image: {
                data: base64Image,
                mimeType: mimeType,
            }
        });

    } catch (error) {
        console.error("Imagen API Error:", error);
        // Respond with a more user-friendly error
        res.status(500).json({ error: `Image generation failed. This could be due to a safety violation in the prompt or an API error. Details: ${error.message}` });
    }
});


app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});
