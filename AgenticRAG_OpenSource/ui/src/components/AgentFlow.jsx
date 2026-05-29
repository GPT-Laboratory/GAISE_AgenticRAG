/**
 * AgentFlow — the live node graph of the LangGraph machine.
 *
 *      ╭───────────── loop ×N ─────────────╮
 *      ▼                                    │
 *   ┌───────┐      ┌───────┐      ┌──────────┐
 *   │ Agent │ ───▶ │ Tools │ ───▶ │  Answer  │
 *   └───────┘      └───────┘      └──────────┘
 *
 * `flow`       — { agent, tools, finalize } each "idle" | "active" | "done"
 * `loops`      — how many times we've looped back to the agent (>1 lights the arc)
 * `corrective` — corrective-RAG fired (badge on Tools)
 */

const NODES = [
  { key: "agent", label: "Agent", sub: "reasons & routes", icon: BrainIcon },
  { key: "tools", label: "Tools", sub: "retrieve & compute", icon: ToolIcon },
  { key: "finalize", label: "Answer", sub: "grounded reply", icon: CheckCircleIcon },
];

export default function AgentFlow({ flow, loops = 0, corrective = false }) {
  const looped = loops > 1;

  return (
    <div className="flow">
      {/* loop-back arc from Tools → Agent */}
      <div className={`flow-arc ${looped ? "is-active" : ""}`}>
        <svg viewBox="0 0 100 38" preserveAspectRatio="none" aria-hidden="true">
          <path d="M 50 34 C 50 4, 16.5 4, 16.5 34" />
        </svg>
        <span className="flow-arc-head" />
        {looped && <span className="flow-arc-label">loop ×{loops - 1}</span>}
      </div>

      <div className="flow-row">
        {NODES.map((node, i) => {
          const Icon = node.icon;
          const status = flow[node.key] || "idle";
          return (
            <div className="flow-cell" key={node.key}>
              <div className={`flow-node ${status}`}>
                <div className="flow-node-icon">
                  <Icon />
                  <span className="flow-node-state">
                    {status === "done" ? <TickIcon /> : status === "active" ? <Spinner /> : null}
                  </span>
                </div>
                <div className="flow-node-text">
                  <strong>{node.label}</strong>
                  <span>{node.sub}</span>
                </div>
                {node.key === "tools" && corrective && (
                  <span className="flow-badge">↻ corrective-RAG</span>
                )}
              </div>

              {i < NODES.length - 1 && (
                <div
                  className={`flow-edge ${
                    flow[NODES[i + 1].key] === "active" ? "flowing" : ""
                  } ${
                    flow[node.key] === "done" && flow[NODES[i + 1].key] !== "idle"
                      ? "lit"
                      : ""
                  }`}
                >
                  <span className="flow-edge-dot" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ----------------------------- inline icons ----------------------------- */

function Spinner() {
  return <span className="mini-spinner" aria-label="running" />;
}

function TickIcon() {
  return (
    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="3">
      <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M9 4.5A2.5 2.5 0 0 0 6.5 7 2.5 2.5 0 0 0 5 9.3 2.5 2.5 0 0 0 6 14v2.2A2.8 2.8 0 0 0 9 19" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M15 4.5A2.5 2.5 0 0 1 17.5 7 2.5 2.5 0 0 1 19 9.3 2.5 2.5 0 0 1 18 14v2.2A2.8 2.8 0 0 1 15 19" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 4v15" strokeLinecap="round" />
    </svg>
  );
}

function ToolIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M14.7 6.3a4 4 0 0 0-5.4 5.2l-5 5a1.5 1.5 0 0 0 2.1 2.1l5-5a4 4 0 0 0 5.2-5.4l-2.5 2.5-2.1-2.1 2.7-2.3z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function CheckCircleIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.7">
      <circle cx="12" cy="12" r="9" />
      <path d="M8.5 12.5l2.5 2.5 4.5-5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
