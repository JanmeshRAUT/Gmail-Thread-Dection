import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getEmailBody(payload: any): string {
  if (!payload) return "";
  
  const decodeBase64 = (data: string): string => {
    try {
      // Handle URL-safe base64 (used by Gmail API)
      const base64 = data.replace(/-/g, '+').replace(/_/g, '/');
      // Use browser's built-in atob for decoding
      return decodeURIComponent(atob(base64).split('').map((c) => {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
      }).join(''));
    } catch (e) {
      console.error('Failed to decode base64:', e);
      return data;
    }
  };
  
  let body = "";
  if (payload.parts) {
    for (const part of payload.parts) {
      if (part.mimeType === "text/plain" || part.mimeType === "text/html") {
        if (part.body && part.body.data) {
          body += decodeBase64(part.body.data);
        }
      } else if (part.parts) {
        body += getEmailBody(part);
      }
    }
  } else if (payload.body && payload.body.data) {
    body = decodeBase64(payload.body.data);
  }
  
  return body;
}

export function getHeader(headers: any[], name: string): string {
  return headers?.find(h => h.name.toLowerCase() === name.toLowerCase())?.value || "";
}

export function getAttachments(payload: any): any[] {
  const attachments: any[] = [];
  if (payload.parts) {
    for (const part of payload.parts) {
      if (part.filename && part.body.attachmentId) {
        attachments.push({
          id: part.body.attachmentId,
          filename: part.filename,
          mimeType: part.mimeType,
          size: part.body.size
        });
      } else if (part.parts) {
        attachments.push(...getAttachments(part));
      }
    }
  }
  return attachments;
}
