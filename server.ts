import express from "express";
import { createServer as createViteServer } from "vite";
import { google } from "googleapis";
import cookieSession from "cookie-session";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

dotenv.config();

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
    secure: true,
    sameSite: "none",
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
  if (!req.session?.tokens) return res.status(401).json({ error: "Unauthorized" });

  oauth2Client.setCredentials(req.session.tokens);
  const gmail = google.gmail({ version: "v1", auth: oauth2Client });

  try {
    const response = await gmail.users.messages.list({
      userId: "me",
      maxResults: 20,
    });

    const messages = await Promise.all(
      (response.data.messages || []).map(async (msg) => {
        const detail = await gmail.users.messages.get({
          userId: "me",
          id: msg.id!,
        });
        return detail.data;
      })
    );

    res.json(messages);
  } catch (error) {
    console.error("Error fetching messages", error);
    res.status(500).json({ error: "Failed to fetch messages" });
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

// Vite middleware for development
async function startServer() {
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
}

startServer();
