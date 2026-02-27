import { GoogleGenAI, Type } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

export interface ThreatAnalysis {
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  summary: string;
  threats: string[];
  recommendation: string;
}

export async function analyzeEmail(content: string, attachments: { filename: string, mimeType: string, data?: string }[] = []) {
  const model = "gemini-3.1-pro-preview";
  
  const parts: any[] = [
    { text: `Analyze the following email for security threats (phishing, malware, social engineering, suspicious links, etc.). 
    
    Email Content:
    ${content}
    
    Attachments:
    ${attachments.map(a => `- ${a.filename} (${a.mimeType})`).join('\n')}
    
    Provide a structured analysis.` }
  ];

  // If there are image attachments, we could add them here as inlineData
  // For now, we'll focus on the text and metadata.

  const response = await ai.models.generateContent({
    model,
    contents: { parts },
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          riskLevel: { type: Type.STRING, enum: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] },
          summary: { type: Type.STRING },
          threats: { type: Type.ARRAY, items: { type: Type.STRING } },
          recommendation: { type: Type.STRING },
        },
        required: ['riskLevel', 'summary', 'threats', 'recommendation']
      }
    }
  });

  return JSON.parse(response.text || "{}") as ThreatAnalysis;
}
