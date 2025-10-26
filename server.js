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
        res.write("Error: Conversation history is empty.");
        return res.end();
    }

    let contents = history; 
    
    // Handle MULTIPLE image parts (up to 4 supported by frontend)
    if (imageParts && imageParts.length > 0) {
        const lastUserMessage = contents.pop(); 
        const multiModalMessage = {
            role: "user",
            parts: [
                // Spread the user's text part(s)
                ...lastUserMessage.parts.filter(p => p.text), 
                // Spread the array of image parts
                ...imageParts 
            ]
        };
        contents.push(multiModalMessage);
    }
    
    // Define Safety Settings (Moderate Blocking)
    const safetySettings = [
      { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
      { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
      { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
      { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
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

app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});
