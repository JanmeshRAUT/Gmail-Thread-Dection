import express from "express";
import { createServer as createViteServer } from "vite";
import { google } from "googleapis";
import { GoogleGenAI, Type } from "@google/genai";
import cookieSession from "cookie-session";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

console.log("🚀 Starting server...");

dotenv.config();

console.log("📝 Environment loaded");
console.log("GOOGLE_CLIENT_ID:", process.env.GOOGLE_CLIENT_ID ? "✓ Set" : "✗ Missing");
console.log("GOOGLE_CLIENT_SECRET:", process.env.GOOGLE_CLIENT_SECRET ? "✓ Set" : "✗ Missing");
console.log("GEMINI_API_KEY:", process.env.GEMINI_API_KEY ? "✓ Set" : "✗ Missing");
console.log("APP_URL:", process.env.APP_URL);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.set('trust proxy', 1);

app.use((req, res, next) => {
  console.log(`${req.method} ${req.url}`);
  next();
});

app.use(express.json());
app.use(
  cookieSession({
    name: "session",
    keys: [process.env.GOOGLE_CLIENT_SECRET || "secret"],
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
    secure: process.env.NODE_ENV === "production",
    sameSite: process.env.NODE_ENV === "production" ? "none" : "lax",
    httpOnly: true,
  })
);

const appUrl = process.env.APP_URL;
if (!appUrl) {
  console.error('CRITICAL: APP_URL environment variable is missing. OAuth will fail.');
}

const redirectUri = `${appUrl}/auth/google/callback`;
console.log('Using Redirect URI:', redirectUri);

const oauth2Client = new google.auth.OAuth2(
  process.env.GOOGLE_CLIENT_ID,
  process.env.GOOGLE_CLIENT_SECRET,
  redirectUri
);

// Auth Routes
app.get("/api/auth/url", (req, res) => {
  const url = oauth2Client.generateAuthUrl({
    access_type: "offline",
    scope: [
      "https://www.googleapis.com/auth/gmail.readonly",
      "https://www.googleapis.com/auth/userinfo.email",
    ],
    redirect_uri: redirectUri,
    prompt: "consent",
  });
  res.json({ url });
});

app.get("/auth/google/callback", async (req, res) => {
  const { code } = req.query;
  try {
    const { tokens } = await oauth2Client.getToken(code as string);
    req.session!.tokens = tokens;
    res.send(`
      <html>
        <body>
          <script>
            if (window.opener) {
              window.opener.postMessage({ type: 'OAUTH_AUTH_SUCCESS' }, '*');
              window.close();
            } else {
              window.location.href = '/';
            }
          </script>
          <p>Authentication successful. This window should close automatically.</p>
        </body>
      </html>
    `);
  } catch (error) {
    console.error("Error exchanging code for tokens", error);
    res.status(500).send("Authentication failed");
  }
});

app.get("/api/auth/status", (req, res) => {
  res.json({ isAuthenticated: !!req.session?.tokens });
});

app.post("/api/auth/logout", (req, res) => {
  req.session = null;
  res.json({ success: true });
});

// Gmail API Routes
app.get("/api/gmail/messages", async (req, res) => {
  console.log("📧 /api/gmail/messages called");
  console.log("Session tokens:", req.session?.tokens ? "✓ Present" : "✗ Missing");
  
  if (!req.session?.tokens) {
    console.log("❌ No tokens in session");
    return res.status(401).json({ error: "Unauthorized" });
  }

  oauth2Client.setCredentials(req.session.tokens);
  const gmail = google.gmail({ version: "v1", auth: oauth2Client });

  try {
    console.log("🔍 Fetching messages from Gmail...");
    const response = await gmail.users.messages.list({
      userId: "me",
      maxResults: 20,
    });

    console.log(`✓ Found ${response.data.messages?.length || 0} messages`);

    const messages = await Promise.all(
      (response.data.messages || []).map(async (msg) => {
        const detail = await gmail.users.messages.get({
          userId: "me",
          id: msg.id!,
        });
        return detail.data;
      })
    );

    console.log(`✓ Fetched ${messages.length} message details`);
    res.json(messages);
  } catch (error: any) {
    console.error("❌ Error fetching messages:", error?.message);
    console.error("Full error:", error);
    res.status(500).json({ 
      error: "Failed to fetch messages",
      details: error?.message
    });
  }
});

app.get("/api/gmail/messages/:id", async (req, res) => {
  if (!req.session?.tokens) return res.status(401).json({ error: "Unauthorized" });

  oauth2Client.setCredentials(req.session.tokens);
  const gmail = google.gmail({ version: "v1", auth: oauth2Client });

  try {
    const response = await gmail.users.messages.get({
      userId: "me",
      id: req.params.id,
    });
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch message" });
  }
});

app.get("/api/gmail/attachment/:messageId/:attachmentId", async (req, res) => {
  if (!req.session?.tokens) return res.status(401).json({ error: "Unauthorized" });

  oauth2Client.setCredentials(req.session.tokens);
  const gmail = google.gmail({ version: "v1", auth: oauth2Client });

  try {
    const response = await gmail.users.messages.attachments.get({
      userId: "me",
      messageId: req.params.messageId,
      id: req.params.attachmentId,
    });
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch attachment" });
  }
});

// Gemini Analysis Endpoint
app.post("/api/analyze", async (req, res) => {
  console.log("🧠 /api/analyze called");
  console.log("Request body:", req.body ? Object.keys(req.body) : "empty");
  
  const { content, attachments } = req.body;
  
  if (!content) {
    console.error("❌ Missing content");
    return res.status(400).json({ error: "Missing email content" });
  }

  if (!process.env.GEMINI_API_KEY) {
    console.error("❌ GEMINI_API_KEY is not set");
    return res.status(500).json({ error: "Gemini API key not configured" });
  }

  try {
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const model = "gemini-3-flash-preview";
    
    const analysisPrompt = `Analyze the following email for BOTH security threats AND spam classification.

Return ONLY a JSON object with these exact fields:
{
  "riskLevel": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
  "summary": "Brief summary of security findings",
  "threats": ["threat1", "threat2"],
  "recommendation": "Recommended security action",
  "isSpam": true or false,
  "spamScore": 0 to 100,
  "spamReason": "Why this is or isn't spam"
}

Email Content:
${content}

Attachments:
${(attachments || []).map((a: any) => `- ${a.filename} (${a.mimeType})`).join('\n') || 'None'}`;

    console.log("🔍 Calling Gemini API with model:", model);
    const response = await ai.models.generateContent({
      model,
      contents: [{ role: "user", parts: [{ text: analysisPrompt }] }],
    });

    console.log("✓ Gemini API response received");
    const responseText = response.text || "{}";
    console.log("Response text (first 300 chars):", responseText.substring(0, 300));
    
    // Try to extract JSON if it's wrapped in markdown code blocks
    let jsonText = responseText;
    const jsonMatch = responseText.match(/```(?:json)?\n?([\s\S]*?)\n?```/);
    if (jsonMatch) {
      jsonText = jsonMatch[1];
    }
    
    const analysis = JSON.parse(jsonText);
    console.log("✓ Parsed analysis:", analysis);
    res.json(analysis);
  } catch (error: any) {
    console.error("❌ Error analyzing email:", error?.message);
    console.error("Full error:", JSON.stringify(error, null, 2));
    res.status(500).json({ 
      error: "Failed to analyze email",
      details: error?.message
    });
  }
});

// Vite middleware for development
async function startServer() {
  try {
    if (process.env.NODE_ENV !== "production") {
      const vite = await createViteServer({
        server: { middlewareMode: true },
        appType: "spa",
      });
      app.use(vite.middlewares);
    } else {
      app.use(express.static(path.join(__dirname, "dist")));
      app.get("*", (req, res) => {
        res.sendFile(path.join(__dirname, "dist", "index.html"));
      });
    }

    app.listen(PORT, "0.0.0.0", () => {
      console.log(`Server running on http://localhost:${PORT}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer().catch(error => {
  console.error('Startup error:', error);
  process.exit(1);
});
