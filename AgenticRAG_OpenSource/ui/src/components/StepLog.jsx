/**
 * StepLog — the detailed, human-readable trace of how the answer was produced.
 * Each step shows what happened, plus the metadata that makes it credible:
 * the model that ran, time taken, token usage, what each tool does, and
 * retrieval stats (chunks, similarity scores, sources).
 */

const KIND_META = {
  reasoning: { tag: "Reasoning", cls: "k-reason" },
  plan: { tag: "Plan", cls: "k-plan" },
  tool: { tag: "Tool call", cls: "k-tool" },
  corrective: { tag: "Corrective-RAG", cls: "k-corrective" },
  finalize: { tag: "Finalize", cls: "k-finalize" },
};

// What each tool does — shown under the tool name so the trace is self-explaining.
const TOOL_INFO = {
  list_data_dimensions: "Lists the available species, count-items and water-quality metrics.",
  get_catch: "Reads fish catch in kg — by year, total trend, top-N, or a single species' trend.",
  get_largest_change: "Finds the species with the largest relative increase or decrease.",
  estimate_value: "Computes economic value (€) = catch × Luke unit price for a species/year.",
  rank_value_species: "Ranks species by economic value and detects if the leader changed.",
  value_trend_or_compare: "Value trend, two-species value comparison, or value crossover year.",
  compare_species_catch: "Compares two species' catch trends, or their (lagged) correlation.",
  correlate_with_metric: "Correlates a species' catch with water-quality metric(s), with optional year lag.",
  count_item_analysis: "Signal-crayfish value vs the top fish species, including the crossover year.",
  forecast_catch: "Projects a species' future catch from its historical trend.",
  document_search: "Vector search over the knowledge base for definitions, methodology and units.",
};

function fmtMs(ms) {
  if (ms == null) return null;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function fmtArgs(args = {}) {
  const entries = Object.entries(args).filter(([k]) => k !== "reformulated");
  if (entries.length === 0) return "";
  return entries
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join(", ");
}

export default function StepLog({ steps }) {
  if (!steps || steps.length === 0) return null;

  return (
    <ol className="steplog">
      {steps.map((step) => {
        const meta = KIND_META[step.kind] || KIND_META.tool;
        const dur = fmtMs(step.elapsed_ms);
        const args = step.kind === "tool" ? fmtArgs(step.arguments) : "";
        const r = step.retrieval;
        return (
          <li key={step.id} className={`step ${step.status} ${meta.cls}`}>
            <span className="step-rail">
              <span className="step-dot">
                {step.status === "done" ? (
                  <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="3.2">
                    <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : step.status === "running" ? (
                  <span className="mini-spinner" />
                ) : null}
              </span>
            </span>

            <div className="step-body">
              <div className="step-head">
                <span className={`step-tag ${meta.cls}`}>{meta.tag}</span>
                {step.kind === "tool" && (
                  <span className={`cat-badge cat-${step.category}`}>
                    {step.category === "retrieval" ? "Retrieval" : "Analytics"}
                  </span>
                )}
                <span className="step-title">
                  {step.kind === "tool" ? <code>{step.title}</code> : step.title}
                  {args && <code className="step-args">({args})</code>}
                </span>

                <span className="step-meta">
                  {step.iteration != null && <span className="chip">iter {step.iteration}</span>}
                  {dur && <span className="chip chip-time">{dur}</span>}
                </span>
              </div>

              {step.kind === "tool" && TOOL_INFO[step.title] && (
                <div className="step-detail">{TOOL_INFO[step.title]}</div>
              )}

              {step.kind === "reasoning" && (step.model || step.tokens) && (
                <div className="step-sub">
                  {step.model && <span className="chip chip-model">⚙ {step.model}</span>}
                  {step.tokens?.in != null && (
                    <span className="chip">{step.tokens.in.toLocaleString()} tok in</span>
                  )}
                  {step.tokens?.out != null && (
                    <span className="chip">{step.tokens.out.toLocaleString()} tok out</span>
                  )}
                </div>
              )}

              {step.detail && <div className="step-detail">{step.detail}</div>}

              {r && (
                <div className="retrieval-box">
                  {r.k != null && <span className="chip">{r.k} chunks</span>}
                  {r.scores?.length > 0 && (
                    <span className="chip">top score {Math.max(...r.scores).toFixed(3)}</span>
                  )}
                  {r.sources?.length > 0 && (
                    <span className="retrieval-sources">sources: {r.sources.join(", ")}</span>
                  )}
                </div>
              )}

              {step.kind === "tool" && step.dataSources?.length > 0 && (
                <div className="source-chips">
                  <span className="source-chips-label">data:</span>
                  {step.dataSources.map((s) => (
                    <span key={s} className={`src-chip ${s.endsWith(".csv") ? "is-dataset" : "is-doc"}`}>
                      {s.endsWith(".csv") ? "▦" : "¶"} {s}
                    </span>
                  ))}
                </div>
              )}

              {step.guardNote && <div className="step-detail guard-note">⚠ {step.guardNote}</div>}

              {step.preview && <pre className="step-preview">{step.preview}</pre>}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
