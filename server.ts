import express from "express";
import { createServer as createViteServer } from "vite";
import { google } from "googleapis";
import cookieSession from "cookie-session";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import crypto from "crypto";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const APP_URL      = process.env.APP_URL || "http://localhost:3000";
const CLIENT_ID    = process.env.GOOGLE_CLIENT_ID;
const CLIENT_SEC   = process.env.GOOGLE_CLIENT_SECRET;
const SESSION_SEC  = process.env.SESSION_SECRET || crypto.randomUUID(); // separate from OAuth secret
const PORT         = 3000;

// ── Startup diagnostics ────────────────────────────────────────────────────────
console.log("GOOGLE_CLIENT_ID:",     CLIENT_ID  ? "✓ Set" : "✗ Missing");
console.log("GOOGLE_CLIENT_SECRET:", CLIENT_SEC  ? "✓ Set" : "✗ Missing");
console.log("SESSION_SECRET:",       process.env.SESSION_SECRET ? "✓ Set" : "⚠ Auto-generated (add to .env)");
console.log("APP_URL:",              APP_URL);

if (!CLIENT_ID || !CLIENT_SEC) {
  console.error("CRITICAL: Missing Google OAuth credentials — auth will fail");
}

const REDIRECT_URI = `${APP_URL}/auth/google/callback`;
console.log("Redirect URI:", REDIRECT_URI);

// ── Express app ────────────────────────────────────────────────────────────────
const app = express();
app.set("trust proxy", 1);
app.use(express.json());

// ── Session (separate secret from OAuth credentials) ──────────────────────────
app.use(
  cookieSession({
    name:    "session",
    keys:    [SESSION_SEC],
    maxAge:  24 * 60 * 60 * 1000,
    secure:  process.env.NODE_ENV === "production",
    sameSite: process.env.NODE_ENV === "production" ? "none" : "lax",
    httpOnly: true,
  })
);

// ── Request logger (dev only) ─────────────────────────────────────────────────
if (process.env.NODE_ENV !== "production") {
  app.use((req, _res, next) => {
    console.log(`${req.method} ${req.url}`);
    next();
  });
}

// ── OAuth2 client ──────────────────────────────────────────────────────────────
const oauth2Client = new google.auth.OAuth2(CLIENT_ID, CLIENT_SEC, REDIRECT_URI);

// ── Auth: generate URL with state param (CSRF protection) ─────────────────────
app.get("/api/auth/url", (req, res) => {
  const state = crypto.randomUUID();
  req.session!.oauthState = state;               // store state to verify on callback

  const url = oauth2Client.generateAuthUrl({
    access_type:  "offline",
    scope: [
      "https://www.googleapis.com/auth/gmail.readonly",
      "https://www.googleapis.com/auth/userinfo.email",
    ],
    redirect_uri: REDIRECT_URI,
    prompt:       "consent",
    state,                                        // FIX: include state for CSRF protection
  });
  res.json({ url });
});

// ── Auth: OAuth callback ───────────────────────────────────────────────────────
app.get("/auth/google/callback", async (req, res) => {
  const { code, state } = req.query;

  // FIX: Verify state to prevent CSRF
  if (!state || state !== req.session?.oauthState) {
    console.error("❌ OAuth state mismatch — possible CSRF attempt");
    res.status(403).send("Authentication failed: invalid state parameter");
    return;
  }
  req.session!.oauthState = undefined;           // clear after use

  try {
    const { tokens } = await oauth2Client.getToken(code as string);
    req.session!.tokens = tokens;

    // FIX: postMessage with explicit targetOrigin instead of wildcard '*'
    res.send(`
      <html>
        <body>
          <script>
            if (window.opener) {
              window.opener.postMessage({ type: 'OAUTH_AUTH_SUCCESS' }, ${JSON.stringify(APP_URL)});
              window.close();
            } else {
              window.location.href = '/';
            }
          </script>
          <p>Authentication successful. This window should close automatically.</p>
        </body>
      </html>
    `);
  } catch (err) {
    console.error("Error exchanging OAuth code:", err);
    res.status(500).send("Authentication failed");
  }
});

// ── Auth: status ───────────────────────────────────────────────────────────────
app.get("/api/auth/status", (req, res) => {
  res.json({ isAuthenticated: !!req.session?.tokens });
});

// ── Auth: logout ───────────────────────────────────────────────────────────────
app.post("/api/auth/logout", (req, res) => {
  req.session = null;
  res.json({ success: true });
});

// ── Gmail: list messages ───────────────────────────────────────────────────────
app.get("/api/gmail/messages", async (req, res) => {
  if (!req.session?.tokens) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  oauth2Client.setCredentials(req.session.tokens);
  const gmail = google.gmail({ version: "v1", auth: oauth2Client });

  try {
    const listResp = await gmail.users.messages.list({
      userId:     "me",
      maxResults: 20,
    });

    const msgs = listResp.data.messages || [];
    const messages = await Promise.all(
      msgs.map(async (msg) => {
        const detail = await gmail.users.messages.get({ userId: "me", id: msg.id! });
        return detail.data;
      })
    );

    res.json(messages);
  } catch (err: any) {
    console.error("Error fetching messages:", err?.message);
    res.status(500).json({ error: "Failed to fetch messages", details: err?.message });
  }
});

// ── Gmail: single message ──────────────────────────────────────────────────────
app.get("/api/gmail/messages/:id", async (req, res) => {
  if (!req.session?.tokens) return res.status(401).json({ error: "Unauthorized" });

  oauth2Client.setCredentials(req.session.tokens);
  const gmail = google.gmail({ version: "v1", auth: oauth2Client });

  try {
    const resp = await gmail.users.messages.get({ userId: "me", id: req.params.id });
    res.json(resp.data);
  } catch {
    res.status(500).json({ error: "Failed to fetch message" });
  }
});

// ── Gmail: attachment ──────────────────────────────────────────────────────────
app.get("/api/gmail/attachment/:messageId/:attachmentId", async (req, res) => {
  if (!req.session?.tokens) return res.status(401).json({ error: "Unauthorized" });

  oauth2Client.setCredentials(req.session.tokens);
  const gmail = google.gmail({ version: "v1", auth: oauth2Client });

  try {
    const resp = await gmail.users.messages.attachments.get({
      userId:    "me",
      messageId: req.params.messageId,
      id:        req.params.attachmentId,
    });
    res.json(resp.data);
  } catch {
    res.status(500).json({ error: "Failed to fetch attachment" });
  }
});

// ── Analysis: proxy to Python LSTM API (retry on backend only) ────────────────
app.post("/api/analyze", async (req, res) => {
  const { content, attachments } = req.body;

  if (!content) {
    return res.status(400).json({ error: "Missing email content" });
  }

  const MAX_RETRIES = 3;
  let lastError: string | null = null;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const pythonRes = await fetch("http://127.0.0.1:5000/analyze", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ content, attachments }),
      });

      if (!pythonRes.ok) {
        lastError = `Python API returned ${pythonRes.status}`;
        if (attempt < MAX_RETRIES) {
          await new Promise(r => setTimeout(r, 1000));
          continue;
        }
        return res.status(pythonRes.status).json({ error: "LSTM Model analysis failed", details: lastError });
      }

      const analysis = await pythonRes.json();
      return res.json(analysis);

    } catch (err: any) {
      lastError = err?.message;
      if (attempt < MAX_RETRIES) {
        await new Promise(r => setTimeout(r, 1000));
      }
    }
  }

  res.status(500).json({
    error:   "Failed to connect to LSTM Model",
    details: lastError,
    hint:    "Run: python ml/api_server.py",
  });
});

// ── Health: check LSTM Model availability ──────────────────────────────────────
app.get("/api/health", async (req, res) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);

  try {
    const pythonRes = await fetch("http://127.0.0.1:5000/health", {
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (pythonRes.ok) {
      const data = await pythonRes.json();
      return res.json(data);
    }
    res.status(503).json({ status: "unavailable", model_loaded: false });
  } catch (err: any) {
    clearTimeout(timeout);
    res.status(503).json({
      status:  "unavailable",
      model_loaded: false,
      error:   "LSTM Model unavailable",
      hint:    "Run: python ml/api_server.py",
    });
  }
});

// ── Predict: single URL risk assessment ────────────────────────────────────────
app.post("/api/predict", async (req, res) => {
  const { url } = req.body;

  if (!url) {
    return res.status(400).json({ error: "Missing URL" });
  }

  const MAX_RETRIES = 2;
  let lastError: string | null = null;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const pythonRes = await fetch("http://127.0.0.1:5000/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ url }),
      });

      if (!pythonRes.ok) {
        lastError = `Python API returned ${pythonRes.status}`;
        if (attempt < MAX_RETRIES) {
          await new Promise(r => setTimeout(r, 500));
          continue;
        }
        return res.status(pythonRes.status).json({ error: "URL prediction failed" });
      }

      const prediction = await pythonRes.json();
      return res.json(prediction);

    } catch (err: any) {
      lastError = err?.message;
      if (attempt < MAX_RETRIES) {
        await new Promise(r => setTimeout(r, 500));
      }
    }
  }

  res.status(500).json({
    error:   "Failed to predict URL risk",
    details: lastError,
    hint:    "Run: python ml/api_server.py",
  });
});

// ── Vite / static serving ──────────────────────────────────────────────────────
async function startServer() {
  try {
    if (process.env.NODE_ENV !== "production") {
      const vite = await createViteServer({
        server:  { middlewareMode: true },
        appType: "spa",
      });
      app.use(vite.middlewares);
    } else {
      app.use(express.static(path.join(__dirname, "dist")));
      app.get("*", (_req, res) =>
        res.sendFile(path.join(__dirname, "dist", "index.html"))
      );
    }

    app.listen(PORT, "0.0.0.0", () => {
      console.log(`✅ Server running → http://localhost:${PORT}`);
    });
  } catch (err) {
    console.error("Failed to start server:", err);
    process.exit(1);
  }
}

startServer().catch(err => {
  console.error("Startup error:", err);
  process.exit(1);
});
