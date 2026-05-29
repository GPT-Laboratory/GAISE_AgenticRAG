/**
 * ChartView — renders the deterministic `chart_data` payloads built in
 * app/charts.py as real Recharts visuals, with everything humanized:
 * friendly titles, axis labels with units, readable tick/legend names, and
 * thousands-separated numbers. Unknown shapes fall back to a labelled table.
 */
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

const PALETTE = ["#4f46e5", "#06b6d4", "#f59e0b", "#ec4899", "#10b981", "#8b5cf6"];
const AXIS = { stroke: "#94a3b8", fontSize: 12 };
const GRID = "#e2e8f0";
const LABEL_STYLE = { fill: "#475569", fontSize: 12, fontWeight: 600 };

/* ------------------------------ humanizers ------------------------------ */

const METRIC = {
  temp_1m_c: { label: "Surface temperature", unit: "°C" },
  temp_bottom_c: { label: "Bottom temperature", unit: "°C" },
  chlorophyll_a_surface_ug_l: { label: "Chlorophyll-a", unit: "µg/L" },
  total_n_surface_ug_l: { label: "Total nitrogen (surface)", unit: "µg/L" },
  total_n_mid_ug_l: { label: "Total nitrogen (mid-depth)", unit: "µg/L" },
  total_n_bottom_ug_l: { label: "Total nitrogen (bottom)", unit: "µg/L" },
  total_p_surface_ug_l: { label: "Total phosphorus (surface)", unit: "µg/L" },
  total_p_mid_ug_l: { label: "Total phosphorus (mid-depth)", unit: "µg/L" },
  total_p_bottom_ug_l: { label: "Total phosphorus (bottom)", unit: "µg/L" },
};

const SPECIES = {
  perch: "Perch", pike: "Pike", bream: "Bream", burbot: "Burbot",
  vendace: "Vendace", ruffe: "Ruffe", whitefish: "Whitefish", smelt: "Smelt",
  trout: "Trout", roach: "Roach", bleak: "Bleak", tench: "Tench",
  signal_crayfish: "Signal crayfish",
};

const FIELD = {
  catch_kg: "Catch (kg)",
  estimated_value_eur: "Estimated value (€)",
  year: "Year",
  species_key: "Species",
  value: "Value",
  left_value: "Value",
  right_value: "Value",
};

function titleCase(s) {
  return String(s)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function humanizeSpecies(key) {
  return SPECIES[key] || titleCase(key);
}

function metricLabel(key, lag = 0) {
  const m = METRIC[key];
  let base = m ? `${m.label} (${m.unit})` : titleCase(key);
  if (lag === 1) base += " — previous year";
  else if (lag === -1) base += " — following year";
  return base;
}

function fieldLabel(key) {
  return FIELD[key] || titleCase(key);
}

// "vendace vs temp_bottom_c (lag 1)" → friendly title.
function humanizeTitle(title) {
  if (!title) return "";
  let t = title;
  Object.keys(METRIC).forEach((k) => { t = t.replaceAll(k, METRIC[k].label); });
  Object.keys(SPECIES).forEach((k) => {
    t = t.replace(new RegExp(`\\b${k}\\b`, "g"), SPECIES[k]);
  });
  t = t
    .replace(/\(lag\s*1\)/i, "(previous year)")
    .replace(/\(lag\s*-1\)/i, "(following year)")
    .replace(/\(lag\s*0\)/i, "")
    .trim();
  return t;
}

// Extract the two variables around "vs" from a title, with optional lag.
function parseTwo(title) {
  let t = title || "";
  let lag = 0;
  const lm = t.match(/\(lag\s*(-?\d+)\)/i);
  if (lm) lag = parseInt(lm[1], 10);
  t = t
    .replace(/\(lag\s*-?\d+\)/i, "")
    .replace(/catches?/i, "")
    .replace(/^estimated value crossover:/i, "")
    .replace(/value over time/i, "")
    .trim();
  const parts = t.split(/\s+vs\s+/i);
  if (parts.length !== 2) return null;
  return { left: parts[0].trim(), right: parts[1].trim(), lag };
}

function describeToken(tok, lag = 0) {
  if (METRIC[tok]) return metricLabel(tok, lag);
  return `${humanizeSpecies(tok)} catch (kg)`;
}

const fmtNum = (v) =>
  typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : v;

function tooltipStyle() {
  return {
    contentStyle: {
      borderRadius: 10,
      border: "1px solid #e2e8f0",
      boxShadow: "0 8px 24px rgba(15,23,42,0.12)",
      fontSize: 12,
    },
    formatter: (value, name) => [fmtNum(value), name],
  };
}

/* -------------------------------- view ---------------------------------- */

export default function ChartView({ chartData }) {
  if (!chartData) return null;
  return (
    <figure className="chart-card">
      <figcaption className="chart-title">{humanizeTitle(chartData.title)}</figcaption>
      <div className="chart-area">{renderChart(chartData)}</div>
    </figure>
  );
}

function renderChart(cd) {
  switch (cd.chart_type) {
    case "line": return <SingleLine cd={cd} />;
    case "bar": return <SingleBar cd={cd} />;
    case "scatter": return <ScatterView cd={cd} />;
    case "multi_line": return <MultiLine cd={cd} />;
    case "dual_line": return <DualLine cd={cd} />;
    default: return <FallbackTable cd={cd} />;
  }
}

function SingleLine({ cd }) {
  const x = cd.x_key || "year";
  const y = cd.y_key || "value";
  return (
    <ResponsiveContainer width="100%" height={290}>
      <LineChart data={cd.data} margin={{ top: 8, right: 20, bottom: 28, left: 28 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey={x} tick={AXIS} label={{ value: fieldLabel(x), position: "insideBottom", offset: -14, style: LABEL_STYLE }} />
        <YAxis tick={AXIS} width={72} tickFormatter={fmtNum} label={{ value: fieldLabel(y), angle: -90, position: "insideLeft", offset: -10, style: LABEL_STYLE }} />
        <Tooltip {...tooltipStyle()} />
        <Line type="monotone" dataKey={y} name={fieldLabel(y)} stroke={PALETTE[0]} strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function SingleBar({ cd }) {
  const isSpecies = cd.x_key === "species_key";
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={cd.data} margin={{ top: 8, right: 20, bottom: 40, left: 28 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey={cd.x_key}
          tick={AXIS}
          interval={0}
          angle={-18}
          textAnchor="end"
          height={64}
          tickFormatter={isSpecies ? humanizeSpecies : undefined}
          label={{ value: fieldLabel(cd.x_key), position: "insideBottom", offset: -2, style: LABEL_STYLE }}
        />
        <YAxis tick={AXIS} width={72} tickFormatter={fmtNum} label={{ value: fieldLabel(cd.y_key), angle: -90, position: "insideLeft", offset: -10, style: LABEL_STYLE }} />
        <Tooltip {...tooltipStyle()} cursor={{ fill: "rgba(79,70,229,0.06)" }} labelFormatter={isSpecies ? humanizeSpecies : undefined} />
        <Bar dataKey={cd.y_key} name={fieldLabel(cd.y_key)} fill={PALETTE[0]} radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function ScatterView({ cd }) {
  const x = cd.x_key || "right_value";
  const y = cd.y_key || "left_value";
  const parsed = parseTwo(cd.title);
  const yLabel = parsed ? describeToken(parsed.left, 0) : fieldLabel(y);
  const xLabel = parsed ? describeToken(parsed.right, parsed.lag) : fieldLabel(x);
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 8, right: 20, bottom: 36, left: 28 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis type="number" dataKey={x} name={xLabel} tick={AXIS} tickFormatter={fmtNum} label={{ value: xLabel, position: "insideBottom", offset: -16, style: LABEL_STYLE }} />
        <YAxis type="number" dataKey={y} name={yLabel} tick={AXIS} width={72} tickFormatter={fmtNum} label={{ value: yLabel, angle: -90, position: "insideLeft", offset: -10, style: LABEL_STYLE }} />
        <ZAxis range={[80, 80]} />
        <Tooltip {...tooltipStyle()} cursor={{ strokeDasharray: "3 3" }} />
        <Scatter data={cd.data} fill={PALETTE[0]} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function DualLine({ cd }) {
  const x = cd.x_key || "year";
  const lk = cd.left_y_key || "left_value";
  const rk = cd.right_y_key || "right_value";
  const parsed = parseTwo(cd.title);
  const leftName = parsed ? humanizeSpecies(parsed.left) : "Series 1";
  const rightName = parsed ? humanizeSpecies(parsed.right) : "Series 2";
  return (
    <ResponsiveContainer width="100%" height={290}>
      <LineChart data={cd.data} margin={{ top: 8, right: 20, bottom: 28, left: 28 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey={x} tick={AXIS} label={{ value: fieldLabel(x), position: "insideBottom", offset: -14, style: LABEL_STYLE }} />
        <YAxis tick={AXIS} width={72} tickFormatter={fmtNum} />
        <Tooltip {...tooltipStyle()} />
        <Legend />
        <Line type="monotone" dataKey={lk} name={leftName} stroke={PALETTE[0]} strokeWidth={2.5} dot={{ r: 3 }} />
        <Line type="monotone" dataKey={rk} name={rightName} stroke={PALETTE[1]} strokeWidth={2.5} dot={{ r: 3 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function MultiLine({ cd }) {
  const series = cd.series || {};
  const rawNames = Object.keys(series);
  const byX = new Map();
  rawNames.forEach((raw) => {
    const name = humanizeSpecies(raw);
    (series[raw] || []).forEach((pt) => {
      const xv = pt.year ?? pt.x ?? pt.key;
      const row = byX.get(xv) || { x: xv };
      row[name] = pt.value ?? pt.y;
      byX.set(xv, row);
    });
  });
  const data = [...byX.values()].sort((a, b) => (a.x > b.x ? 1 : -1));
  const names = rawNames.map(humanizeSpecies);

  return (
    <ResponsiveContainer width="100%" height={290}>
      <LineChart data={data} margin={{ top: 8, right: 20, bottom: 24, left: 28 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="x" tick={AXIS} label={{ value: "Year", position: "insideBottom", offset: -12, style: LABEL_STYLE }} />
        <YAxis tick={AXIS} width={72} tickFormatter={fmtNum} />
        <Tooltip {...tooltipStyle()} />
        <Legend />
        {names.map((name, i) => (
          <Line key={name} type="monotone" dataKey={name} stroke={PALETTE[i % PALETTE.length]} strokeWidth={2.5} dot={{ r: 2.5 }} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function FallbackTable({ cd }) {
  const rows = Array.isArray(cd.data) ? cd.data : [];
  if (rows.length === 0) {
    return <pre className="chart-json">{JSON.stringify(cd, null, 2)}</pre>;
  }
  const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  return (
    <div className="chart-table-wrap">
      <table className="chart-table">
        <thead>
          <tr>{cols.map((c) => <th key={c}>{fieldLabel(c)}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c}>{typeof r[c] === "number" ? fmtNum(r[c]) : String(r[c] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
