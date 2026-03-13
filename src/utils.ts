import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// ─── Gmail API type definitions ───────────────────────────────────────────────

export interface GmailHeader {
  name:  string;
  value: string;
}

export interface GmailBody {
  data?:         string;
  attachmentId?: string;
  size?:         number;
}

export interface GmailPart {
  mimeType?: string;
  filename?: string;
  headers?:  GmailHeader[];
  body?:     GmailBody;
  parts?:    GmailPart[];
}

export interface GmailPayload extends GmailPart {
  headers: GmailHeader[];   // always present on top-level payload
}

export interface GmailEmail {
  id:       string;
  threadId: string;
  snippet:  string;
  payload:  GmailPayload;
}

export interface Attachment {
  id:       string;
  filename: string;
  mimeType: string;
  size:     number;
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/** Decode Gmail's URL-safe base64 to a UTF-8 string. */
function decodeBase64(data: string): string {
  try {
    const b64 = data.replace(/-/g, '+').replace(/_/g, '/');
    return decodeURIComponent(
      atob(b64)
        .split('')
        .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
  } catch {
    return data;
  }
}

/** Recursively extract text body from a Gmail payload. */
export function getEmailBody(payload: GmailPart): string {
  if (!payload) return '';

  let body = '';
  if (payload.parts) {
    for (const part of payload.parts) {
      if (part.mimeType === 'text/plain' || part.mimeType === 'text/html') {
        if (part.body?.data) body += decodeBase64(part.body.data);
      } else if (part.parts) {
        body += getEmailBody(part);
      }
    }
  } else if (payload.body?.data) {
    body = decodeBase64(payload.body.data);
  }

  return body;
}

/** Find a header value by name (case-insensitive). */
export function getHeader(headers: GmailHeader[], name: string): string {
  return (
    headers?.find(h => h.name.toLowerCase() === name.toLowerCase())?.value ?? ''
  );
}

/** Recursively collect all attachments from a Gmail payload. */
export function getAttachments(payload: GmailPart): Attachment[] {
  const attachments: Attachment[] = [];

  if (payload.parts) {
    for (const part of payload.parts) {
      if (part.filename && part.body?.attachmentId) {
        attachments.push({
          id:       part.body.attachmentId,
          filename: part.filename,
          mimeType: part.mimeType ?? 'application/octet-stream',
          size:     part.body.size ?? 0,
        });
      } else if (part.parts) {
        attachments.push(...getAttachments(part));
      }
    }
  }

  return attachments;
}
