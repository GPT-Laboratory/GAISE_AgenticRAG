/**
 * DataLineage — a bipartite provenance graph showing how the answer was
 * assembled: which tools were called (left) and which data sources each one
 * hit (right), with curved SVG connectors drawn between them. This makes the
 * agentic-RAG reality explicit — a single answer fuses multiple data sources.
 *
 * Connectors are measured from the live DOM (container-relative node centres)
 * and redrawn on resize, so they always line up with the rendered cards.
 */
import { useLayoutEffect, useRef, useState } from "react";

const SOURCE_INFO = {
  // Processed CSV datasets (deterministic analytics tools)
  "catch_clean.csv": { label: "Fish catch", desc: "Catch in kg by year & species, 2010–2024", type: "dataset" },
  "count_catch_clean.csv": { label: "Count catch", desc: "Count-based catch (signal crayfish)", type: "dataset" },
  "water_quality_clean.csv": { label: "Water quality", desc: "Chlorophyll, nitrogen, phosphorus, temperature", type: "dataset" },
  "luke_clean.csv": { label: "Luke economics", desc: "Commercial quantity, value & unit price", type: "dataset" },
  // Knowledge-base documents (vector retrieval via document_search). Keys are the
  // exact filenames stored as chunk `source` metadata in the vector index.
  "methodology.md": { label: "Methodology", desc: "Definitions, assumptions & methods", type: "document" },
  "Saalistilasto-kala-ja-rapu-2010-2027.pdf": { label: "Catch statistics (PDF)", desc: "Official fish & crayfish catch statistics, 2010–2027", type: "document" },
  "TAUlle BioAItyohon luke 0600_kausis_20260326-134254.xlsx": { label: "Luke source data", desc: "Commercial quantity, value & price source workbook", type: "document" },
  "TAUlle BioAItyohon pyhaj syvanteen vedenlaatutietoa.xlsx": { label: "Water-quality source", desc: "Pyhäjärvi deep-point water-quality measurements", type: "document" },
  "TAUlle BioAItyohon pyhajarven saalistilasto 2010_2025.xlsx": { label: "Catch source data", desc: "Pyhäjärvi catch statistics, 2010–2025", type: "document" },
};

function sourceInfo(name) {
  return (
    SOURCE_INFO[name] || {
      label: name,
      desc: "",
      type: name.toLowerCase().endsWith(".csv") ? "dataset" : "document",
    }
  );
}

const uniq = (arr) => [...new Set(arr)];

export default function DataLineage({ links }) {
  const containerRef = useRef(null);
  const toolRefs = useRef({});
  const sourceRefs = useRef({});
  const [paths, setPaths] = useState([]);
  const [hover, setHover] = useState(null); // { kind, name }

  const tools = uniq(links.map((l) => l.tool));
  const sources = uniq(links.map((l) => l.source));
  const datasetCount = sources.filter((s) => sourceInfo(s).type === "dataset").length;
  const docCount = sources.length - datasetCount;

  useLayoutEffect(() => {
    function measure() {
      const c = containerRef.current;
      if (!c) return;
      const cb = c.getBoundingClientRect();
      const next = links
        .map((l) => {
          const t = toolRefs.current[l.tool];
          const s = sourceRefs.current[l.source];
          if (!t || !s) return null;
          const tb = t.getBoundingClientRect();
          const sb = s.getBoundingClientRect();
          const x1 = tb.right - cb.left;
          const y1 = tb.top + tb.height / 2 - cb.top;
          const x2 = sb.left - cb.left;
          const y2 = sb.top + sb.height / 2 - cb.top;
          const mx = (x1 + x2) / 2;
          return {
            key: `${l.tool}__${l.source}`,
            tool: l.tool,
            source: l.source,
            type: sourceInfo(l.source).type,
            d: `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`,
          };
        })
        .filter(Boolean);
      setPaths(next);
    }
    measure();
    const ro = new ResizeObserver(measure);
    if (containerRef.current) ro.observe(containerRef.current);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [links]);

  const isLit = (l) => {
    if (!hover) return true;
    return hover.kind === "tool" ? l.tool === hover.name : l.source === hover.name;
  };

  return (
    <div className="lineage">
      <div className="lineage-intro">
        This answer fuses{" "}
        <strong>
          {datasetCount} dataset{datasetCount !== 1 ? "s" : ""}
        </strong>
        {docCount > 0 && (
          <>
            {" "}and{" "}
            <strong>
              {docCount} document{docCount !== 1 ? "s" : ""}
            </strong>
          </>
        )}{" "}
        via <strong>{tools.length}</strong> tool{tools.length !== 1 ? "s" : ""}.
      </div>

      <div className="lineage-graph" ref={containerRef}>
        <svg className="lineage-svg" aria-hidden="true">
          {paths.map((p) => (
            <path
              key={p.key}
              d={p.d}
              className={`lineage-link type-${p.type} ${isLit(p) ? "lit" : "dim"}`}
            />
          ))}
        </svg>

        <div className="lineage-col lineage-tools">
          <div className="lineage-head">Tools</div>
          {tools.map((t) => (
            <div
              key={t}
              ref={(el) => (toolRefs.current[t] = el)}
              className={`lineage-node tool-node ${hover?.name === t ? "hover" : ""}`}
              onMouseEnter={() => setHover({ kind: "tool", name: t })}
              onMouseLeave={() => setHover(null)}
            >
              <span className="node-dot" />
              <code>{t}</code>
            </div>
          ))}
        </div>

        <div className="lineage-col lineage-sources">
          <div className="lineage-head">Data sources</div>
          {sources.map((s) => {
            const info = sourceInfo(s);
            const hits = links.filter((l) => l.source === s).length;
            return (
              <div
                key={s}
                ref={(el) => (sourceRefs.current[s] = el)}
                className={`lineage-node source-node type-${info.type} ${hover?.name === s ? "hover" : ""}`}
                onMouseEnter={() => setHover({ kind: "source", name: s })}
                onMouseLeave={() => setHover(null)}
                title={info.desc}
              >
                <span className={`src-icon ${info.type}`}>{info.type === "dataset" ? "▦" : "¶"}</span>
                <span className="src-text">
                  <strong>{info.label}</strong>
                  <code>{s}</code>
                </span>
                <span className="src-hits" title={`${hits} hit${hits !== 1 ? "s" : ""}`}>{hits}×</span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="lineage-legend">
        <span><span className="lg-swatch dataset" /> CSV dataset (deterministic analytics)</span>
        <span><span className="lg-swatch document" /> Document (vector retrieval)</span>
      </div>
    </div>
  );
}
