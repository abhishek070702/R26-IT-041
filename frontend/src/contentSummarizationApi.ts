import { API_BASE_URL } from "./config";

export type AnalyzeApiResponse = {
  extracted_text: string;
  categories: string[];
  category_counts: Record<string, number>;
  results: unknown[];
  analysis: Record<string, unknown>;
};

export type GenerateOutputResponse = {
  final_output_text: string;
  next_module_payload: string;
};

function parseErrorBody(text: string): string {
  try {
    const j = JSON.parse(text) as { detail?: unknown };
    const d = j.detail;
    if (typeof d === "string") return d;
    if (typeof d === "object" && d !== null && !Array.isArray(d)) {
      const msg = (d as { message?: string }).message;
      if (typeof msg === "string") return msg;
    }
    if (Array.isArray(d))
      return d
        .map((item) =>
          typeof item === "object" && item && "msg" in item
            ? String((item as { msg: string }).msg)
            : JSON.stringify(item),
        )
        .join("; ");
  } catch {
    /* ignore */
  }
  return text;
}

export async function analyzeContentSummarization(
  file: File,
  documentType: string,
): Promise<AnalyzeApiResponse> {
  const body = new FormData();
  body.append("file", file);
  body.append("document_type", documentType);

  const res = await fetch(`${API_BASE_URL}/content/analyze`, {
    method: "POST",
    body,
  });

  const text = await res.text();

  if (!res.ok) {
    throw new Error(parseErrorBody(text) || `Request failed (${res.status})`);
  }

  return JSON.parse(text) as AnalyzeApiResponse;
}

export async function generateContentSummarizationOutput(
  analysis: Record<string, unknown>,
  selectedCategory: string,
  depth: "summary" | "full",
): Promise<GenerateOutputResponse> {
  const res = await fetch(`${API_BASE_URL}/content/generate-output`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      analysis,
      selected_category: selectedCategory,
      depth,
    }),
  });

  const text = await res.text();

  if (!res.ok) {
    throw new Error(parseErrorBody(text) || `Request failed (${res.status})`);
  }

  return JSON.parse(text) as GenerateOutputResponse;
}
