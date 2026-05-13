import { useCallback, useEffect, useState } from "react";
import {
  analyzeContentSummarization,
  generateContentSummarizationOutput,
  type AnalyzeApiResponse,
} from "./contentSummarizationApi";

const DOCUMENT_TYPES = [
  { value: "Newspaper", label: "Newspaper" },
  { value: "Magazine", label: "Magazine" },
  { value: "Novel/Story", label: "Novel / Story" },
  { value: "Textbook", label: "Textbook" },
  { value: "General Document", label: "General Document" },
] as const;

type Depth = "summary" | "full";

export function ContentSummarizationSection() {
  const [documentType, setDocumentType] = useState<string>(DOCUMENT_TYPES[0].value);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeApiResponse | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [depth, setDepth] = useState<Depth>("summary");
  const [finalOutputText, setFinalOutputText] = useState("");
  const [nextModulePayload, setNextModulePayload] = useState("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!analyzeResult) {
      setSelectedCategory(null);
      return;
    }
    const cats = analyzeResult.categories;
    if (cats.length === 1) {
      setSelectedCategory(cats[0]);
    } else {
      setSelectedCategory(null);
    }
  }, [analyzeResult]);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    setFile(f ?? null);
    setError(null);
    setAnalyzeResult(null);
    setFinalOutputText("");
    setNextModulePayload("");
  };

  const onAnalyze = useCallback(async () => {
    if (!file) {
      setError("Choose an image to analyze.");
      return;
    }
    setIsAnalyzing(true);
    setError(null);
    setFinalOutputText("");
    setNextModulePayload("");
    try {
      const data = await analyzeContentSummarization(file, documentType);
      setAnalyzeResult(data);
    } catch (err) {
      setAnalyzeResult(null);
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setIsAnalyzing(false);
    }
  }, [file, documentType]);

  const canGenerate =
    !!analyzeResult?.analysis &&
    analyzeResult.categories.length > 0 &&
    (analyzeResult.categories.length === 1 || !!selectedCategory);

  const onGenerateOutput = useCallback(async () => {
    if (!analyzeResult?.analysis || !canGenerate) return;
    const cat = selectedCategory ?? analyzeResult.categories[0] ?? null;
    if (!cat) return;

    setIsGenerating(true);
    setError(null);
    try {
      const out = await generateContentSummarizationOutput(
        analyzeResult.analysis,
        cat,
        depth,
      );
      setFinalOutputText(out.final_output_text);
      setNextModulePayload(out.next_module_payload);
    } catch (err) {
      setFinalOutputText("");
      setNextModulePayload("");
      setError(err instanceof Error ? err.message : "Generation failed.");
    } finally {
      setIsGenerating(false);
    }
  }, [analyzeResult, canGenerate, depth, selectedCategory]);

  const copyPayload = async () => {
    if (!nextModulePayload) return;
    try {
      await navigator.clipboard.writeText(nextModulePayload);
      setCopied(true);
      setTimeout(() => setCopied(false), 2200);
    } catch {
      setError("Could not copy to clipboard.");
    }
  };

  const cats = analyzeResult?.categories ?? [];
  const counts = analyzeResult?.category_counts ?? {};

  return (
    <div className="summarization-root">
      <p className="summarization-intro">
        Upload a page image, run OCR and category detection, then generate a summary or full text for
        a chosen category. Requires Tesseract, the category model under <code>model/category_model.pkl</code>,
        OpenAI for newspaper or magazine regions, and optional Ollama for reconstruction and summaries.
      </p>

      <div className="grid summarization-grid">
        <section className="card summarization-card">
          <h2>1. Source</h2>
          <label className="file-label summarization-file">
            Choose image
            <input type="file" accept="image/*" onChange={onFileChange} className="hidden-input" />
          </label>
          {file ? (
            <p className="small-text summarization-filename">{file.name}</p>
          ) : null}

          {previewUrl ? (
            <div className="preview-box summarization-preview">
              <img src={previewUrl} alt="Document preview" />
            </div>
          ) : null}

          <label className="summarization-label" htmlFor="summ-doc-type">
            Document type
          </label>
          <select
            id="summ-doc-type"
            className="summarization-select"
            value={documentType}
            onChange={(e) => setDocumentType(e.target.value)}
          >
            {DOCUMENT_TYPES.map((d) => (
              <option key={d.value} value={d.value}>
                {d.label}
              </option>
            ))}
          </select>

          <button
            type="button"
            className="predict-btn summarization-primary"
            onClick={onAnalyze}
            disabled={isAnalyzing || !file}
          >
            {isAnalyzing ? "Analyzing…" : "Run analysis"}
          </button>
        </section>

        <section className="card summarization-card">
          <h2>2. Categories & depth</h2>
          {isAnalyzing ? (
            <p className="small-text">Running pipeline…</p>
          ) : analyzeResult && cats.length > 0 ? (
            <>
              <ul className="summarization-category-list">
                {cats.map((c) => (
                  <li key={c}>
                    <span className="summarization-cat-name">{c}</span>
                    {counts[c] != null ? (
                      <span className="summarization-cat-count">×{counts[c]}</span>
                    ) : null}
                  </li>
                ))}
              </ul>
              {cats.length > 1 ? (
                <div className="summarization-cat-pick">
                  <p className="small-text">Select category for output</p>
                  <div className="summarization-cat-buttons">
                    {cats.map((c) => (
                      <button
                        key={c}
                        type="button"
                        className={
                          selectedCategory === c
                            ? "summarization-cat-btn active"
                            : "summarization-cat-btn"
                        }
                        onClick={() => setSelectedCategory(c)}
                      >
                        {c}
                        {counts[c] != null ? ` (${counts[c]})` : ""}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="small-text">Single category — selected automatically.</p>
              )}

              <fieldset className="summarization-depth">
                <legend className="summarization-label">Output depth</legend>
                <label className="summarization-radio">
                  <input
                    type="radio"
                    name="summ-depth"
                    checked={depth === "summary"}
                    onChange={() => setDepth("summary")}
                  />
                  Summary
                </label>
                <label className="summarization-radio">
                  <input
                    type="radio"
                    name="summ-depth"
                    checked={depth === "full"}
                    onChange={() => setDepth("full")}
                  />
                  Full text
                </label>
              </fieldset>

              <button
                type="button"
                className="describe-btn summarization-secondary"
                onClick={onGenerateOutput}
                disabled={!canGenerate || isGenerating || isAnalyzing}
              >
                {isGenerating ? "Generating…" : "Generate output"}
              </button>
              {cats.length > 1 && !selectedCategory ? (
                <p className="small-text summarization-hint">Pick a category above to enable generation.</p>
              ) : null}
            </>
          ) : (
            <p className="empty-text">Run analysis to see categories.</p>
          )}
        </section>

        <section className="card summarization-card summarization-wide">
          <h2>3. Extracted text</h2>
          {analyzeResult?.extracted_text ? (
            <details className="summarization-details">
              <summary>View extracted text ({analyzeResult.extracted_text.length.toLocaleString()} chars)</summary>
              <pre className="summarization-pre">{analyzeResult.extracted_text}</pre>
            </details>
          ) : (
            <p className="empty-text">No extraction yet.</p>
          )}
        </section>

        <section className="card summarization-card summarization-wide">
          <h2>4. Final output & integration</h2>
          {finalOutputText ? (
            <div className="summarization-output-block">
              <h3 className="section-title">Meaningful output</h3>
              <pre className="summarization-pre light">{finalOutputText}</pre>
            </div>
          ) : (
            <p className="empty-text">Generate output to see text for the next module.</p>
          )}
          {nextModulePayload ? (
            <div className="summarization-output-block">
              <h3 className="section-title">Next module payload</h3>
              <pre className="summarization-pre light">{nextModulePayload}</pre>
              <button type="button" className="capture-btn" onClick={copyPayload}>
                {copied ? "Copied" : "Copy payload"}
              </button>
            </div>
          ) : null}
        </section>
      </div>

      {error ? (
        <p className="error-text summarization-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
