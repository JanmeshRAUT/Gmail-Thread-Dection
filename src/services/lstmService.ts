/**
 * LSTM Phishing Detection Service
 * ─────────────────────────────────
 * Single responsibility: call /api/analyze and /api/predict.
 * Retry logic removed from here — retries live in server.ts (backend only).
 */

export interface PhishingURL {
  url:        string;
  risk_level: string;
  confidence: number;
}

export interface ThreatAnalysis {
  riskLevel:      'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  summary:        string;
  threats:        string[];
  recommendation: string;
  isSpam?:        boolean;
  spamScore?:     number;
  spamReason?:    string;
  urls_analyzed?: number;
  phishing_urls?: PhishingURL[];
  model?:         string;
}

export interface AnalysisAttachment {
  filename: string;
  mimeType: string;
  data?:    string;
}

/**
 * Analyse email content for phishing threats via the LSTM model.
 * Throws on error — let the caller handle it with UI feedback.
 */
export async function analyzeEmail(
  content:     string,
  attachments: AnalysisAttachment[] = []
): Promise<ThreatAnalysis> {
  const response = await fetch('/api/analyze', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ content, attachments }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({})) as { error?: string };
    throw new Error(err.error ?? `API error: ${response.status}`);
  }

  return response.json() as Promise<ThreatAnalysis>;
}

/**
 * Health check — verify the LSTM Python API is reachable.
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch('/api/health');
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Predict risk for a single URL.
 */
export async function predictURL(url: string): Promise<unknown> {
  const response = await fetch('/api/predict', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ url }),
  });

  if (!response.ok) {
    throw new Error(`Prediction API error: ${response.status}`);
  }

  return response.json();
}
