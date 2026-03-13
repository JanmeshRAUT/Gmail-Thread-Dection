import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Shield, Mail, AlertTriangle, CheckCircle,
  Search, LogOut, ExternalLink, Loader2,
  ChevronRight, AlertCircle, X, Info,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { ThreatAnalysis, analyzeEmail } from './services/lstmService';
import {
  cn, GmailEmail,
  getEmailBody, getHeader, getAttachments,
} from './utils';

// ─── Toast Notification ────────────────────────────────────────────────────────
type ToastType = 'error' | 'warning' | 'info';

interface Toast {
  id:      string;
  type:    ToastType;
  message: string;
}

function ToastContainer({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: string) => void }) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none">
      <AnimatePresence>
        {toasts.map(t => (
          <motion.div
            key={t.id}
            initial={{ opacity: 0, x: 60 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 60 }}
            className={cn(
              'flex items-start gap-3 p-4 rounded-xl border text-sm shadow-xl pointer-events-auto',
              t.type === 'error'   && 'bg-red-950 border-red-800 text-red-300',
              t.type === 'warning' && 'bg-amber-950 border-amber-800 text-amber-300',
              t.type === 'info'    && 'bg-zinc-900 border-zinc-700 text-zinc-300',
            )}
          >
            {t.type === 'error'   && <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-red-400" />}
            {t.type === 'warning' && <AlertCircle  className="w-4 h-4 shrink-0 mt-0.5 text-amber-400" />}
            {t.type === 'info'    && <Info         className="w-4 h-4 shrink-0 mt-0.5 text-zinc-400" />}
            <p className="flex-1 leading-relaxed">{t.message}</p>
            <button
              onClick={() => onDismiss(t.id)}
              className="shrink-0 opacity-60 hover:opacity-100"
            >
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

// ─── Hook: toast manager ───────────────────────────────────────────────────────
function useToast() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const show = useCallback((message: string, type: ToastType = 'info') => {
    const id = crypto.randomUUID();
    setToasts(prev => [...prev, { id, type, message }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000);
  }, []);

  const dismiss = useCallback((id: string) =>
    setToasts(prev => prev.filter(t => t.id !== id)), []);

  return { toasts, show, dismiss };
}

// ─── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const [emails,          setEmails]          = useState<GmailEmail[]>([]);
  const [selectedEmail,   setSelectedEmail]   = useState<GmailEmail | null>(null);
  const [analysis,        setAnalysis]        = useState<ThreatAnalysis | null>(null);
  const [isAnalyzing,     setIsAnalyzing]     = useState(false);
  const [isLoadingEmails, setIsLoadingEmails] = useState(false);

  const { toasts, show: showToast, dismiss: dismissToast } = useToast();

  // ── Auth ─────────────────────────────────────────────────────────────────────
  const checkAuth = useCallback(async () => {
    try {
      const res  = await fetch('/api/auth/status');
      const data = await res.json() as { isAuthenticated: boolean };
      setIsAuthenticated(data.isAuthenticated);
    } catch {
      setIsAuthenticated(false);
    }
  }, []);

  const handleLogin = async () => {
    try {
      const res = await fetch('/api/auth/url');
      if (!res.ok) {
        showToast('Failed to get login URL — check server logs', 'error');
        return;
      }
      const { url } = await res.json() as { url: string };
      window.open(url, 'google_oauth', 'width=600,height=700');
    } catch {
      showToast('Login failed — server may be down', 'error');
    }
  };

  const handleLogout = async () => {
    await fetch('/api/auth/logout', { method: 'POST' });
    setIsAuthenticated(false);
    setEmails([]);
    setSelectedEmail(null);
    setAnalysis(null);
  };

  // ── Emails ───────────────────────────────────────────────────────────────────
  const fetchEmails = useCallback(async () => {
    setIsLoadingEmails(true);
    try {
      const res = await fetch('/api/gmail/messages');
      if (!res.ok) {
        showToast('Failed to load emails', 'error');
        return;
      }
      const data = await res.json() as GmailEmail[];
      setEmails(data);

      // FIX: Reconcile selectedEmail with fresh data to avoid stale reference
      setSelectedEmail(prev =>
        prev ? (data.find(e => e.id === prev.id) ?? prev) : null
      );
    } catch {
      showToast('Network error fetching emails', 'error');
    } finally {
      setIsLoadingEmails(false);
    }
  }, [showToast]);

  // ── Effects ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    checkAuth();

    const handleMessage = (event: MessageEvent) => {
      // Only accept messages from trusted origin
      if (event.origin !== window.location.origin) return;
      if ((event.data as { type?: string })?.type === 'OAUTH_AUTH_SUCCESS') {
        setIsAuthenticated(true);
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [checkAuth]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchEmails();
      const interval = setInterval(fetchEmails, 30_000);
      return () => clearInterval(interval);
    }
  }, [isAuthenticated, fetchEmails]);

  // ── Analysis ──────────────────────────────────────────────────────────────────
  const handleAnalyze = async (email: GmailEmail) => {
    setIsAnalyzing(true);
    setAnalysis(null);
    try {
      const body        = getEmailBody(email.payload);
      const attachments = getAttachments(email.payload).map(a => ({
        filename: a.filename,
        mimeType: a.mimeType,
      }));

      const result = await analyzeEmail(body, attachments);

      if (!result.riskLevel) {
        showToast('Analysis returned an invalid response — please retry', 'warning');
        return;
      }

      setAnalysis(result);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      showToast(`Analysis failed: ${msg}`, 'error');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ── Loading screen ────────────────────────────────────────────────────────────
  if (isAuthenticated === null) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-emerald-500 animate-spin" />
      </div>
    );
  }

  // ── Login screen ──────────────────────────────────────────────────────────────
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-zinc-950 flex flex-col items-center justify-center p-6 text-center">
        <ToastContainer toasts={toasts} onDismiss={dismissToast} />
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md"
        >
          <div className="w-20 h-20 bg-emerald-500/10 rounded-3xl flex items-center justify-center mb-8 mx-auto border border-emerald-500/20">
            <Shield className="w-10 h-10 text-emerald-500" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-4 tracking-tight">
            Gmail Threat Guard
          </h1>
          <p className="text-zinc-400 mb-8 text-lg">
            Protect your inbox with AI-powered threat detection. Analyse content and
            attachments for phishing, malware, and social engineering.
          </p>
          <button
            id="btn-login-google"
            onClick={handleLogin}
            className="w-full bg-white text-black font-semibold py-4 px-8 rounded-2xl hover:bg-zinc-200 transition-colors flex items-center justify-center gap-3 text-lg"
          >
            <Mail className="w-5 h-5" />
            Connect with Gmail
          </button>
          <p className="mt-6 text-xs text-zinc-500">
            Read-only access only. Your data is processed locally and never stored.
          </p>
        </motion.div>
      </div>
    );
  }

  // ── Main App ──────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      {/* Header */}
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-xl sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-emerald-500/10 rounded-xl flex items-center justify-center border border-emerald-500/20">
              <Shield className="w-6 h-6 text-emerald-500" />
            </div>
            <span className="font-bold text-xl tracking-tight">Threat Guard</span>
          </div>

          <div className="flex items-center gap-4">
            <button
              id="btn-refresh-emails"
              onClick={fetchEmails}
              disabled={isLoadingEmails}
              className="p-2 hover:bg-zinc-800 rounded-lg transition-colors disabled:opacity-50"
              title="Refresh emails"
            >
              <Loader2 className={cn('w-5 h-5', isLoadingEmails && 'animate-spin')} />
            </button>
            <button
              id="btn-logout"
              onClick={handleLogout}
              className="flex items-center gap-2 px-4 py-2 hover:bg-zinc-800 rounded-lg transition-colors text-zinc-400 hover:text-white"
            >
              <LogOut className="w-4 h-4" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* Email List */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500">
              Recent Emails
            </h2>
            <span className="text-xs bg-zinc-800 px-2 py-1 rounded-md text-zinc-400">
              {emails.length}
            </span>
          </div>

          <div className="flex flex-col gap-2 overflow-y-auto max-h-[calc(100vh-200px)] pr-2 custom-scrollbar">
            {emails.map(email => {
              const subject    = getHeader(email.payload.headers, 'Subject');
              const from       = getHeader(email.payload.headers, 'From');
              const isSelected = selectedEmail?.id === email.id;

              return (
                <button
                  key={email.id}
                  id={`email-item-${email.id}`}
                  onClick={() => { setSelectedEmail(email); setAnalysis(null); }}
                  className={cn(
                    'text-left p-4 rounded-2xl transition-all border group',
                    isSelected
                      ? 'bg-emerald-500/10 border-emerald-500/30 ring-1 ring-emerald-500/20'
                      : 'bg-zinc-900 border-zinc-800 hover:border-zinc-700',
                  )}
                >
                  <div className="flex justify-between items-start mb-1">
                    <span className={cn(
                      'text-xs font-medium truncate max-w-[180px]',
                      isSelected ? 'text-emerald-400' : 'text-zinc-500',
                    )}>
                      {from}
                    </span>
                    <ChevronRight className={cn(
                      'w-4 h-4 transition-transform',
                      isSelected
                        ? 'text-emerald-500 translate-x-1'
                        : 'text-zinc-700 group-hover:translate-x-0.5',
                    )} />
                  </div>
                  <h3 className={cn(
                    'font-semibold truncate mb-1',
                    isSelected ? 'text-white' : 'text-zinc-200',
                  )}>
                    {subject || '(No Subject)'}
                  </h3>
                  <p className="text-xs text-zinc-500 line-clamp-2 leading-relaxed">
                    {email.snippet}
                  </p>
                </button>
              );
            })}

            {emails.length === 0 && !isLoadingEmails && (
              <div className="text-center py-12 bg-zinc-900/50 rounded-3xl border border-dashed border-zinc-800">
                <Mail className="w-8 h-8 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">No emails found</p>
              </div>
            )}
          </div>
        </div>

        {/* Analysis Panel */}
        <div className="lg:col-span-8">
          <AnimatePresence mode="wait">
            {selectedEmail ? (
              <motion.div
                key={selectedEmail.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="bg-zinc-900 rounded-3xl border border-zinc-800 overflow-hidden flex flex-col h-full min-h-[600px]"
              >
                {/* Email header */}
                <div className="p-8 border-b border-zinc-800 bg-zinc-900/50">
                  <div className="flex justify-between items-start gap-4 mb-6">
                    <div>
                      <h2 className="text-2xl font-bold text-white mb-2">
                        {getHeader(selectedEmail.payload.headers, 'Subject') || '(No Subject)'}
                      </h2>
                      <div className="flex flex-wrap gap-4 text-sm text-zinc-400">
                        <div className="flex items-center gap-2">
                          <span className="text-zinc-500">From:</span>
                          <span className="text-zinc-300">
                            {getHeader(selectedEmail.payload.headers, 'From')}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-zinc-500">Date:</span>
                          <span className="text-zinc-300">
                            {getHeader(selectedEmail.payload.headers, 'Date')}
                          </span>
                        </div>
                      </div>
                    </div>

                    <button
                      id="btn-analyze-threat"
                      onClick={() => handleAnalyze(selectedEmail)}
                      disabled={isAnalyzing}
                      className="bg-emerald-500 hover:bg-emerald-400 disabled:bg-emerald-900 disabled:text-emerald-700 text-black font-bold py-3 px-6 rounded-xl transition-all flex items-center gap-2 cursor-pointer shadow-lg shadow-emerald-500/20"
                    >
                      {isAnalyzing
                        ? <Loader2 className="w-5 h-5 animate-spin" />
                        : <Shield className="w-5 h-5" />
                      }
                      {isAnalyzing ? 'Analyzing…' : 'Analyze Threat'}
                    </button>
                  </div>

                  {/* Attachments */}
                  {getAttachments(selectedEmail.payload).length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {getAttachments(selectedEmail.payload).map(att => (
                        <div
                          key={att.id}
                          className="flex items-center gap-2 bg-zinc-800 px-3 py-1.5 rounded-lg border border-zinc-700 text-xs text-zinc-300"
                        >
                          <AlertCircle className="w-3 h-3 text-zinc-500" />
                          <span>{att.filename}</span>
                          <span className="text-zinc-600">
                            ({Math.round(att.size / 1024)} KB)
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Content area */}
                <div className="flex-1 overflow-y-auto p-8 custom-scrollbar">
                  {analysis ? (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-8"
                    >
                      {/* Risk banner */}
                      <div className={cn(
                        'p-6 rounded-2xl border flex items-center gap-6',
                        analysis.riskLevel === 'CRITICAL' && 'bg-red-500/10    border-red-500/30    text-red-400',
                        analysis.riskLevel === 'HIGH'     && 'bg-orange-500/10 border-orange-500/30 text-orange-400',
                        analysis.riskLevel === 'MEDIUM'   && 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400',
                        analysis.riskLevel === 'LOW'      && 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
                      )}>
                        <div className={cn(
                          'w-16 h-16 rounded-2xl flex items-center justify-center shrink-0',
                          analysis.riskLevel === 'CRITICAL' && 'bg-red-500/20',
                          analysis.riskLevel === 'HIGH'     && 'bg-orange-500/20',
                          analysis.riskLevel === 'MEDIUM'   && 'bg-yellow-500/20',
                          analysis.riskLevel === 'LOW'      && 'bg-emerald-500/20',
                        )}>
                          {analysis.riskLevel === 'LOW'
                            ? <CheckCircle  className="w-8 h-8" />
                            : <AlertTriangle className="w-8 h-8" />
                          }
                        </div>
                        <div>
                          <div className="text-xs font-bold uppercase tracking-widest mb-1 opacity-70">
                            Risk Assessment
                          </div>
                          <div className="text-3xl font-black">{analysis.riskLevel}</div>
                        </div>
                      </div>

                      {/* Summary + Threats */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="space-y-4">
                          <h4 className="text-sm font-bold text-zinc-400 uppercase tracking-wider">Summary</h4>
                          <p className="text-zinc-300 leading-relaxed">{analysis.summary}</p>
                        </div>
                        <div className="space-y-4">
                          <h4 className="text-sm font-bold text-zinc-400 uppercase tracking-wider">Detected Threats</h4>
                          <ul className="space-y-2">
                            {analysis.threats.map((threat, i) => (
                              <li key={i} className="flex items-start gap-3 text-zinc-300">
                                <div className="w-1.5 h-1.5 rounded-full bg-zinc-600 mt-2 shrink-0" />
                                <span>{threat}</span>
                              </li>
                            ))}
                            {analysis.threats.length === 0 && (
                              <li className="text-zinc-500 italic">No specific threats detected</li>
                            )}
                          </ul>
                        </div>
                      </div>

                      {/* Recommendation */}
                      <div className="p-6 bg-zinc-800/50 rounded-2xl border border-zinc-700/50">
                        <h4 className="text-sm font-bold text-zinc-400 uppercase tracking-wider mb-4">
                          Recommendation
                        </h4>
                        <p className="text-zinc-200 font-medium">{analysis.recommendation}</p>
                      </div>

                      {/* Spam card */}
                      {analysis.isSpam !== undefined && (
                        <div className={cn(
                          'p-6 rounded-2xl border flex items-start gap-6',
                          analysis.isSpam
                            ? 'bg-red-500/10 border-red-500/30'
                            : 'bg-emerald-500/10 border-emerald-500/30',
                        )}>
                          <div className={cn(
                            'w-16 h-16 rounded-2xl flex items-center justify-center shrink-0',
                            analysis.isSpam ? 'bg-red-500/20' : 'bg-emerald-500/20',
                          )}>
                            {analysis.isSpam
                              ? <AlertTriangle className="w-8 h-8 text-red-400" />
                              : <CheckCircle  className="w-8 h-8 text-emerald-400" />
                            }
                          </div>

                          <div className="flex-1">
                            <div className={cn(
                              'text-sm font-bold uppercase tracking-widest mb-2',
                              analysis.isSpam ? 'text-red-400' : 'text-emerald-400',
                            )}>
                              {analysis.isSpam ? '⚠ SPAM DETECTED' : '✓ NOT SPAM'}
                            </div>

                            <div className="flex items-center gap-4 mb-3">
                              <div className="flex-1">
                                <div className="text-xs text-zinc-400 mb-1">Spam Score</div>
                                <div className="w-full bg-zinc-700 rounded-full h-2 overflow-hidden">
                                  <div
                                    className={cn(
                                      'h-full transition-all',
                                      (analysis.spamScore ?? 0) < 30 ? 'bg-emerald-500'
                                      : (analysis.spamScore ?? 0) < 60 ? 'bg-yellow-500'
                                      : 'bg-red-500',
                                    )}
                                    style={{ width: `${analysis.spamScore ?? 0}%` }}
                                  />
                                </div>
                                <div className="text-xs text-zinc-400 mt-1">
                                  {analysis.spamScore ?? 0}/100
                                </div>
                              </div>
                            </div>

                            {analysis.spamReason && (
                              <p className="text-sm text-zinc-300">{analysis.spamReason}</p>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Phishing URLs detail */}
                      {(analysis.phishing_urls?.length ?? 0) > 0 && (
                        <div className="p-6 bg-red-950/30 rounded-2xl border border-red-900/50">
                          <h4 className="text-sm font-bold text-red-400 uppercase tracking-wider mb-4">
                            Phishing URLs Detected
                          </h4>
                          <div className="space-y-2">
                            {analysis.phishing_urls!.map((pu, i) => (
                              <div key={i} className="flex items-start gap-3">
                                <ExternalLink className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                                <div className="min-w-0">
                                  <p className="text-xs text-red-300 break-all">{pu.url}</p>
                                  <p className="text-xs text-zinc-500">
                                    {pu.risk_level} — {Math.round(pu.confidence * 100)}% confidence
                                  </p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Original content */}
                      <div className="pt-8 border-t border-zinc-800">
                        <h4 className="text-sm font-bold text-zinc-500 uppercase tracking-wider mb-4">
                          Original Content
                        </h4>
                        <div className="bg-zinc-950 p-6 rounded-2xl border border-zinc-800 text-zinc-400 text-sm whitespace-pre-wrap font-mono">
                          {getEmailBody(selectedEmail.payload)}
                        </div>
                      </div>
                    </motion.div>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-center space-y-6">
                      {isAnalyzing ? (
                        <>
                          <div className="relative">
                            <div className="w-20 h-20 border-4 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin" />
                            <Shield className="w-8 h-8 text-emerald-500 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                          </div>
                          <div>
                            <h3 className="text-xl font-bold text-white mb-2">Analyzing Security…</h3>
                            <p className="text-zinc-500 max-w-xs">
                              LSTM model is scanning URLs and content for phishing patterns.
                            </p>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="w-20 h-20 bg-zinc-800 rounded-3xl flex items-center justify-center border border-zinc-700">
                            <Search className="w-10 h-10 text-zinc-600" />
                          </div>
                          <div>
                            <h3 className="text-xl font-bold text-white mb-2">Ready to Scan</h3>
                            <p className="text-zinc-500 max-w-xs">
                              Click <span className="text-emerald-400 font-semibold">Analyze Threat</span> to
                              run a deep security scan of this email.
                            </p>
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </motion.div>
            ) : (
              <div className="h-full bg-zinc-900/30 rounded-3xl border border-dashed border-zinc-800 flex flex-col items-center justify-center text-center p-12 min-h-[600px]">
                <Mail className="w-16 h-16 text-zinc-800 mb-6" />
                <h3 className="text-2xl font-bold text-zinc-600 mb-2">Select an Email</h3>
                <p className="text-zinc-700 max-w-sm">
                  Choose an email from the list to view its content and perform threat analysis.
                </p>
              </div>
            )}
          </AnimatePresence>
        </div>
      </main>

      <footer className="py-8 border-t border-zinc-900 text-center">
        <p className="text-xs text-zinc-600">
          Powered by Character-Level LSTM &amp; Google Gmail API • Secure Threat Detection
        </p>
      </footer>
    </div>
  );
}
