// server.js
// Simple Express backend that forwards chat requests to Google Gemini with streaming
// Uses environment variable GEMINI_API_KEY (never hardcode your key)

import express from "express";
import cors from "cors";
import fetch from "node-fetch"; // Still needed for standard server checks/errors
import dotenv from "dotenv";

dotenv.config();

const app = express();
// Enable streaming for better performance
app.use(cors());
app.use(express.json());

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
  console.warn(
    "WARNING: GEMINI_API_KEY is not set. Set process.env.GEMINI_API_KEY before starting the server."
  );
}

app.get("/", (req, res) => {
  res.send("✅ Gemini AI backend is running.");
});

// --- NEW STREAMING ENDPOINT ---
app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: "message is required" });

    // 1. Set headers for streaming
    res.setHeader('Content-Type', 'text/plain'); // Sending raw text chunks
    res.setHeader('Transfer-Encoding', 'chunked');
    res.flushHeaders(); // Send headers immediately

    // 2. Use the streaming API endpoint
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContentStream?key=${GEMINI_API_KEY}`;

    const body = {
      contents: [{ parts: [{ text: message }] }],
    };

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text();
      console.error("Gemini API returned non-200:", response.status, text);
      // For streaming, we can't change the status code after flushing headers, 
      // so we send the error message as the final chunk and close the stream.
      res.write(`ERROR: Upstream API error (${response.status}). Details: ${text}`);
      return res.end();
    }
    
    // 3. Process the streaming response from Gemini
    if (response.body) {
      // Use the response body as a ReadableStream
      for await (const chunk of response.body) {
        // The Gemini stream sends newline-separated JSON objects
        const lines = chunk.toString().split('\n');
        
        for (const line of lines) {
          if (line.trim().startsWith('{')) {
            try {
              const data = JSON.parse(line);
              // Extract the text part from the chunk and send it to the client
              const text = data?.candidates?.[0]?.content?.parts?.[0]?.text || "";
              if (text) {
                // Send the raw text chunk to the client
                res.write(text);
              }
            } catch (e) {
              // Ignore malformed JSON or empty lines
            }
          }
        }
      }
    }

    // 4. End the response once the stream is complete
    res.end();

  } catch (err) {
    console.error("Server error:", err);
    res.write(`ERROR: Internal server error: ${err.message}`);
    res.end();
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
