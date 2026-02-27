import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getEmailBody(payload: any): string {
  if (!payload) return "";
  
  let body = "";
  if (payload.parts) {
    for (const part of payload.parts) {
      if (part.mimeType === "text/plain") {
        body += Buffer.from(part.body.data, 'base64').toString();
      } else if (part.parts) {
        body += getEmailBody(part);
      }
    }
  } else if (payload.body && payload.body.data) {
    body = Buffer.from(payload.body.data, 'base64').toString();
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
