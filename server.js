// server.js
// Simple Express backend that forwards chat requests to Google Gemini
// Uses environment variable GEMINI_API_KEY (never hardcode your key)

import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import dotenv from "dotenv";

dotenv.config(); // Loads .env for local development (ignored in git)

const app = express();
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

app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: "message is required" });

    // FIX: Ensure the correct model ID is used for the non-streaming endpoint.
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`;

    const body = {
      contents: [{ parts: [{ text: message }] }],
      // ADDITION for Speed: Limit the output length to force a concise response.
      generationConfig: {
        maxOutputTokens: 200, // This is a good balance for quick chat replies.
      },
    };

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text();
      // Log the original API error for debugging
      console.error("Gemini API returned non-200:", response.status, text);
      return res
        .status(502) 
        // Returning the status code and details back to the client
        .json({ error: "Upstream API error", status: response.status, details: text });
    }

    const data = await response.json();
    // Safely extract the reply text
    const reply = data?.candidates?.[0]?.content?.parts?.[0]?.text || "";

    return res.json({ reply, raw: data });
  } catch (err) {
    console.error("Server error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
