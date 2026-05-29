import { useEffect, useMemo, useRef, useState } from "react";
import { streamChatMessage } from "../api";
import AgentFlow from "./AgentFlow";
import StepLog from "./StepLog";
import ChartView from "./ChartView";
import DataLineage from "./DataLineage";

const STARTER_PROMPTS = [
  "What was the total catch in 2024?",
  "Which species increased the most?",
  "Most economically important species in 2024?",
  "Vendace catch vs previous year's temperature?",
  "What does crayfish refer to here?",
];

const EMPTY_FLOW = { agent: "idle", tools: "idle", finalize: "idle" };

function makeAssistantDraft() {
  return {
    role: "assistant",
    content: "",
    streaming: true,
    flow: { ...EMPTY_FLOW },
    steps: [],
    toolCalls: [],
    chartData: null,
    loops: 0,
    corrective: false,
    collapsed: false,
    run: null,
    summary: null,
    lineage: [],
    linkSet: new Set(),
  };
}

function lastRunning(steps, kind) {
  for (let i = steps.length - 1; i >= 0; i--) {
    if (steps[i].kind === kind && steps[i].status === "running") return steps[i];
  }
  return null;
}

function fmtMs(ms) {
  if (ms == null) return "";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function shortHost(url) {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/** Compact "run configuration" bar: which models and RAG settings produced this answer. */
function RunConfig({ run }) {
  const { models = {}, config = {} } = run;
  return (
    <div className="runconfig">
      <span className="rc-item rc-llm" title="Agent LLM (decides tools, writes the answer)">
        <span className="rc-key">LLM</span>
        {models.agent} · {models.runtime}
      </span>
      <span className="rc-item" title="Automatic fallback model on error">
        <span className="rc-key">fallback</span>
        {models.fallback}
      </span>
      <span className="rc-item rc-emb" title="Embedding model for vector retrieval">
        <span className="rc-key">embeddings</span>
        {models.embeddings}
      </span>
      <span className="rc-item" title="Local OpenAI-compatible endpoint">
        <span className="rc-key">endpoint</span>
        {shortHost(models.endpoint)}
      </span>
      <span className="rc-item" title="RAG retrieval settings · max agent loop iterations">
        <span className="rc-key">RAG</span>
        top-k {config.rag_top_k} · floor {config.rag_score_floor} · max {config.max_tool_iters} iters
      </span>
    </div>
  );
}

function markRunningDone(steps) {
  steps.forEach((s) => {
    if (s.status === "running") s.status = "done";
  });
}

export default function ChatPanel() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Welcome. Ask anything about Lake Pyhäjärvi fishery, value, or water-quality data. " +
        "Watch the panel light up to see exactly which agent step and tool produced your answer.",
      steps: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const liveRef = useRef(null);
  const stepCounter = useRef(0);
  const scrollRef = useRef(null);

  const nextStepId = () => `s${stepCounter.current++}`;

  const conversationCount = useMemo(
    () => messages.filter((m) => m.role === "user").length,
    [messages]
  );

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  function commit() {
    const m = liveRef.current;
    if (!m) return;
    setMessages((prev) => {
      const next = prev.slice();
      next[next.length - 1] = {
        ...m,
        flow: { ...m.flow },
        steps: m.steps.map((s) => ({ ...s })),
        toolCalls: m.toolCalls.slice(),
        lineage: m.lineage.slice(),
      };
      return next;
    });
  }

  function applyEvent(ev) {
    const m = liveRef.current;
    if (!m) return;

    switch (ev.type) {
      case "start": {
        m.run = { models: ev.models, config: ev.config };
        break;
      }

      case "node": {
        Object.keys(m.flow).forEach((k) => {
          if (m.flow[k] === "active") m.flow[k] = "done";
        });
        m.flow[ev.node] = "active";

        if (ev.node === "agent") {
          m.loops += 1;
          markRunningDone(m.steps);
          m.steps.push({
            id: nextStepId(),
            kind: "reasoning",
            title:
              m.loops === 1
                ? "Reading & analysing the question"
                : "Re-evaluating with the new tool results",
            iteration: ev.iteration,
            status: "running",
          });
        } else if (ev.node === "tools") {
          markRunningDone(m.steps);
        } else if (ev.node === "finalize") {
          markRunningDone(m.steps);
          m.steps.push({
            id: nextStepId(),
            kind: "finalize",
            title: "Composing the grounded answer",
            status: "running",
          });
        }
        break;
      }

      case "agent_step": {
        const r = lastRunning(m.steps, "reasoning");
        if (r) {
          r.status = "done";
          r.model = ev.model;
          r.elapsed_ms = ev.elapsed_ms;
          r.tokens = ev.tokens;
        }
        if (ev.action === "call_tools" && (ev.tools || []).length) {
          m.steps.push({
            id: nextStepId(),
            kind: "plan",
            title: "Selected tools",
            detail: (ev.tools || []).join("  ·  "),
            status: "done",
          });
        }
        break;
      }

      case "tool_result": {
        const dataSources = ev.data_sources || [];
        m.steps.push({
          id: nextStepId(),
          kind: "tool",
          title: ev.tool_name,
          category: ev.category,
          arguments: ev.arguments || {},
          preview: ev.preview || "",
          retrieval: ev.retrieval || null,
          dataSources,
          elapsed_ms: ev.elapsed_ms,
          status: "done",
        });
        m.toolCalls.push({
          tool_name: ev.tool_name,
          arguments: ev.arguments || {},
          result: ev.preview || "",
        });
        // accumulate the data-lineage links (deduped tool→source pairs)
        dataSources.forEach((src) => {
          const key = `${ev.tool_name}__${src}`;
          if (!m.linkSet.has(key)) {
            m.linkSet.add(key);
            m.lineage.push({ tool: ev.tool_name, source: src });
          }
        });
        break;
      }

      case "corrective_rag": {
        m.corrective = true;
        m.steps.push({
          id: nextStepId(),
          kind: "corrective",
          title: "Weak retrieval — reformulating the query",
          detail: ev.query,
          status: "done",
        });
        break;
      }

      case "finalize_step": {
        const f = lastRunning(m.steps, "finalize");
        if (f) {
          f.status = "done";
          f.elapsed_ms = ev.elapsed_ms;
          if (ev.language_guard) {
            f.guardNote =
              "Answer drifted out of English — automatically translated back, numbers & citations preserved.";
          }
        }
        break;
      }

      case "final": {
        markRunningDone(m.steps);
        Object.keys(m.flow).forEach((k) => {
          if (m.flow[k] === "active") m.flow[k] = "done";
        });
        m.flow.finalize = "done";
        m.content = ev.answer || "";
        m.chartData = ev.chart_data || null;
        if (ev.tool_calls && ev.tool_calls.length) m.toolCalls = ev.tool_calls;
        m.summary = {
          total_ms: ev.total_ms,
          iterations: ev.iterations,
          tool_count: ev.tool_count,
          model: m.run?.models?.agent,
        };
        m.streaming = false;
        break;
      }

      case "error": {
        markRunningDone(m.steps);
        m.content = `Something went wrong: ${ev.message}`;
        m.streaming = false;
        break;
      }

      default:
        return;
    }
    commit();
  }

  async function handleSend(customText = null) {
    const text = (customText ?? input).trim();
    if (!text || loading) return;

    const history = messages
      .filter((msg) => msg.role === "user" || (msg.role === "assistant" && msg.content))
      .map((msg) => ({ role: msg.role, content: msg.content }));

    const draft = makeAssistantDraft();
    liveRef.current = draft;

    setMessages((prev) => [...prev, { role: "user", content: text }, draft]);
    setInput("");
    setLoading(true);

    try {
      await streamChatMessage(text, history, applyEvent);
    } catch (error) {
      applyEvent({ type: "error", message: error.message });
    } finally {
      if (liveRef.current && liveRef.current.streaming) {
        liveRef.current.streaming = false;
        commit();
      }
      liveRef.current = null;
      setLoading(false);
    }
  }

  function toggleCollapse(index) {
    setMessages((prev) =>
      prev.map((m, i) => (i === index ? { ...m, collapsed: !m.collapsed } : m))
    );
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="app-shell">
      <aside className="left-panel">
        <div className="brand-card">
          <div className="brand-badge">
            <img
              src="/gptlab-logo.png"
              alt=""
              className="brand-badge-logo"
              onError={(e) => { e.currentTarget.style.display = "none"; }}
            />
            GPT-Lab
          </div>
          <h1>Lake Intelligence</h1>
          <p>
            An open-source <strong>Agentic RAG</strong> assistant that answers questions on Lake
            Pyhäjärvi's fishery, economic, and water-quality data. Runs entirely on your machine.
          </p>
        </div>

        <div className="panel-card">
          <div className="panel-header">
            <h2>Try a question</h2>
          </div>
          <div className="prompt-list">
            {STARTER_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                className="prompt-button"
                onClick={() => handleSend(prompt)}
                disabled={loading}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>

        <div className="panel-card">
          <div className="panel-header">
            <h2>How it works</h2>
          </div>
          <ul className="legend">
            <li><span className="legend-dot d-agent" /> <strong>Agent</strong> reasons &amp; picks tools</li>
            <li><span className="legend-dot d-tools" /> <strong>Tools</strong> retrieve &amp; compute</li>
            <li><span className="legend-dot d-final" /> <strong>Answer</strong> grounded reply</li>
            <li><span className="legend-dot d-loop" /> loops back until it has enough</li>
          </ul>
          <div className="meta-row">
            <span>Questions asked</span>
            <strong>{conversationCount}</strong>
          </div>
          <div className="meta-row">
            <span>Status</span>
            <strong>{loading ? "Generating…" : "Ready"}</strong>
          </div>
        </div>
      </aside>

      <main className="chat-panel">
        <div className="chat-header">
          <div>
            <h2>Assistant</h2>
            <p>Every answer shows the agent loop that produced it — live.</p>
          </div>
          <div className="status-pill">
            <span className={`status-dot ${loading ? "active" : ""}`} />
            {loading ? "Thinking" : "Online"}
          </div>
        </div>

        <div className="messages-container" ref={scrollRef}>
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`message-row ${
                msg.role === "user" ? "message-row-user" : "message-row-assistant"
              }`}
            >
              <div className={`avatar ${msg.role}`}>{msg.role === "user" ? "You" : "AI"}</div>

              <div className={`message-bubble ${msg.role === "user" ? "user-bubble" : "assistant-bubble"}`}>
                {msg.role === "assistant" && msg.steps && (msg.steps.length > 0 || msg.streaming) && (
                  <div className="process-card">
                    <button
                      className="process-head"
                      onClick={() => toggleCollapse(index)}
                      disabled={msg.streaming}
                    >
                      <span className="process-title">
                        {msg.streaming ? (
                          <>
                            <span className="mini-spinner" /> Generating answer…
                          </>
                        ) : (
                          <>
                            <span className="process-check">✓</span> How this answer was generated
                          </>
                        )}
                      </span>
                      <span className="process-right">
                        {msg.summary && (
                          <span className="summary-chips">
                            <span className="chip chip-time">{fmtMs(msg.summary.total_ms)}</span>
                            <span className="chip">{msg.summary.iterations} iters</span>
                            <span className="chip">{msg.summary.tool_count} tools</span>
                          </span>
                        )}
                        {!msg.streaming && (
                          <span className={`chev ${msg.collapsed ? "" : "open"}`}>⌄</span>
                        )}
                      </span>
                    </button>

                    {!msg.collapsed && (
                      <div className="process-body">
                        {msg.run && <RunConfig run={msg.run} />}
                        <AgentFlow
                          flow={msg.flow || EMPTY_FLOW}
                          loops={msg.loops || 0}
                          corrective={msg.corrective}
                        />
                        <StepLog steps={msg.steps} />
                        {msg.lineage && msg.lineage.length > 0 && (
                          <div className="lineage-section">
                            <div className="lineage-section-head">Data lineage — where the answer comes from</div>
                            <DataLineage links={msg.lineage} />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {msg.role === "user"
                  ? msg.content
                  : msg.content && (
                      <div className={`message-content ${msg.streaming ? "" : "revealed"}`}>
                        {msg.content}
                      </div>
                    )}

                {msg.role === "assistant" && msg.chartData && (
                  <ChartView chartData={msg.chartData} />
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="input-wrapper">
          <div className="input-card">
            <textarea
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about catch trends, rankings, value, correlations, or definitions…"
            />
            <div className="input-actions">
              <div className="input-hint">Enter to send · Shift+Enter for a new line</div>
              <button className="send-button" onClick={() => handleSend()} disabled={loading}>
                {loading ? "Generating…" : "Send"}
              </button>
            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <img
          src="/gptlab-logo.png"
          alt="GPT-Lab"
          className="footer-logo"
          onError={(e) => { e.currentTarget.style.display = "none"; }}
        />
        <span>© GPT-Lab 2026</span>
      </footer>
    </div>
  );
}
