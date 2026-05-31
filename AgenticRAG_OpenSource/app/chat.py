"""Agentic RAG over the local Qwen 2.5 model (via Ollama), orchestrated with LangGraph.

The flow is an explicit state machine:

    START → agent → (tool_calls?) ─yes→ tools → agent   (loop, capped)
                       │                  │
                       │                  └─ corrective-RAG happens here:
                       │                     weak document_search → reformulate → re-retrieve
                       └─no→ finalize → END

`agent`    – the LLM decides which tools to call (or writes the final answer).
`tools`    – executes the chosen tools via the existing DISPATCH layer, builds
             chart_data, and applies the corrective-RAG step on weak retrieval.
`finalize` – language guard (force English) + citation safety-net footer.

Response contract preserved for the React UI: {answer, tool_calls, chart_data}.
"""
from __future__ import annotations

import json
import operator
import re
import time
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent_tools import DISPATCH, TOOL_SCHEMAS
from app.config import settings
from analytics.document_search import is_weak


AGENT_SYSTEM_PROMPT = """You are a grounded analytical assistant for fishery, economic, and water-quality data from Lake Pyhäjärvi.

CRITICAL: You MUST write every reply in English only, using the Latin alphabet. NEVER output any Chinese, Japanese, Korean, Thai, or Cyrillic characters. Do NOT add parenthetical translations or foreign-language names. Finnish species names in Latin letters (e.g. muikku) are acceptable when needed, but prefer the English name.

You answer ONLY from tools. You have two kinds of tools:
- Analytics tools (get_catch, estimate_value, rank_value_species, correlate_with_metric, count_item_analysis, etc.) for any NUMERIC question: catch in kg, economic value in euros, rankings, trends, correlations, crossovers.
- document_search for DEFINITIONS, methodology, assumptions, units, FAQ phrasings, or any qualitative/explanatory question that is not a number from the datasets.

The datasets cover roughly 2010 through 2024. Every year in that range is queryable — never claim a year is unavailable without first calling the tool for it.

Parameter rules:
- When the user names a specific year (e.g. "in 2024"), pass that year to the tool's `year` parameter. Do not silently fall back to the whole period.
- When the user says "previous year", "last year's", or "the year before", pass `lag_years=1`. "Following/next year" means `lag_years=-1`. Otherwise `lag_years=0`.

Rules:
- For mixed questions, call BOTH kinds of tools.
- Base every number, species, year, price, and source STRICTLY on the tool results returned for THIS question, and state numeric values EXACTLY as they appear. Do not compute new percentages or derived figures unless the tool returned them. Never invent or recall values from memory.
- If a tool result contains an "error" or "message" field (e.g. data missing or a parameter is needed), explain that limitation plainly to the user. NEVER substitute a guessed or remembered value when a tool did not return one.
- Do NOT mention dataset names, file names, or write "(source: …)" citations in your answer. The application displays the data sources separately in a provenance panel, so just present the findings in clean prose.
- Species may be given in Finnish or English (e.g. muikku=vendace, ahven=perch, täplärapu=signal crayfish); pass them straight to the tools, which normalize them.
- Once you have enough information from tools, STOP calling tools and write a concise, direct final answer.

Reminder: the final answer must be written entirely in English.
"""

_MAX_TOOL_RESULT_CHARS = 4000


# --------------------------------------------------------------------------- #
# LLM clients — local Qwen via Ollama's OpenAI-compatible endpoint.
# Primary model with the smaller model as an automatic fallback on error.
# --------------------------------------------------------------------------- #
def _make_llm(with_tools: bool) -> Any:
    primary = ChatOpenAI(
        base_url=settings.OLLAMA_BASE_URL,
        api_key="ollama",
        model=settings.OLLAMA_MODEL,
        temperature=0,
    )
    fallback = ChatOpenAI(
        base_url=settings.OLLAMA_BASE_URL,
        api_key="ollama",
        model=settings.OLLAMA_MODEL_FALLBACK,
        temperature=0,
    )
    if with_tools:
        primary = primary.bind_tools(TOOL_SCHEMAS)
        fallback = fallback.bind_tools(TOOL_SCHEMAS)
    return primary.with_fallbacks([fallback])


LLM_TOOLS = _make_llm(with_tools=True)
LLM_PLAIN = _make_llm(with_tools=False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _render_result(result: dict[str, Any]) -> str:
    """Compact JSON rendering of a tool result for the model, size-bounded."""
    try:
        text = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(result)
    if len(text) > _MAX_TOOL_RESULT_CHARS:
        text = text[:_MAX_TOOL_RESULT_CHARS] + " ...[truncated]"
    return text


def _llm_plain_text(system: str, user: str) -> str:
    """One no-tools LLM call returning plain text (empty string on failure)."""
    try:
        resp = LLM_PLAIN.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return (resp.content or "").strip()
    except Exception:
        return ""


# Thai, CJK (Chinese/Japanese), Korean, and Cyrillic ranges. Finnish (Latin +
# äöå) is fine; only these scripts indicate the model drifted out of English.
_NON_ENGLISH_RE = re.compile(r"[Ѐ-ӿ฀-๿぀-ヿ㐀-鿿가-힯]")


def _looks_non_english(text: str) -> bool:
    return bool(_NON_ENGLISH_RE.search(text))


# A parenthetical aside (ASCII or full-width parens) that contains a non-English
# script — the most common drift pattern, e.g. "smelt （芬兰鱼名称：…）".
_NON_ENGLISH_PAREN_RE = re.compile(
    r"\s*[（(][^（()）]*[Ѐ-ӿ฀-๿぀-ヿ㐀-鿿가-힯][^（()）]*[)）]"
)

_ENGLISH_FALLBACK = (
    "I found the answer but could not render it reliably in English this time. "
    "Please ask the question again."
)


def _force_english(text: str) -> str:
    """Translate a drifted answer back to clean English.

    The local Qwen model is bilingual and sometimes replies (partly) in Chinese;
    a single translation pass can itself drift, so we retry a few times, then
    strip any stray non-English aside as a last resort. We never return text that
    still contains a non-English script.
    """
    current = text
    for _ in range(3):
        out = _llm_plain_text(
            "Rewrite the following text ENTIRELY in clear, natural English. Completely REMOVE "
            "any Chinese, Japanese, Korean, Thai or Cyrillic characters — do NOT keep them and do "
            "NOT add foreign-language names or parenthetical glosses. Preserve every number, year, "
            "euro value, unit and Latin-script (English/Finnish) name EXACTLY. Output only the "
            "English text and nothing else.",
            current,
        )
        if not out or out == current:
            break
        if not _looks_non_english(out):
            return out
        current = out  # partially improved — translate again

    # Last resort: drop non-English parenthetical asides and re-check.
    cleaned = _NON_ENGLISH_PAREN_RE.sub("", current).strip()
    if cleaned and not _looks_non_english(cleaned):
        return cleaned
    return _ENGLISH_FALLBACK


# Inline source citations are redundant now that the UI shows a provenance
# panel — strip them deterministically in case the model adds them anyway.
_CITATION_RE = re.compile(
    r"\s*\((?:[^)]*\bsources?\b[^)]*|[^)]*\.(?:csv|md|txt|pdf)[^)]*)\)",
    re.IGNORECASE,
)
_FOOTER_RE = re.compile(r"\n+\s*sources?\s*:.*$", re.IGNORECASE | re.DOTALL)


def _strip_source_citations(text: str) -> str:
    text = _FOOTER_RE.sub("", text)
    text = _CITATION_RE.sub("", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)   # tidy space before punctuation
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _reformulate(message: str) -> str:
    """One LLM call to rewrite a weak document query (corrective-RAG)."""
    out = _llm_plain_text(
        "Rewrite the user's question as a short, keyword-rich search query for retrieving "
        "relevant passages from fishery/water-quality documents. Reply with ONLY the "
        "rewritten query, no quotes, no explanation.",
        message,
    )
    return out or message


def _build_messages(message: str, history: list[dict[str, Any]]) -> list[AnyMessage]:
    messages: list[AnyMessage] = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    for item in (history or [])[-6:]:
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
    messages.append(HumanMessage(content=message))
    return messages


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_calls_log: Annotated[list[dict[str, Any]], operator.add]
    doc_sources: Annotated[list[str], operator.add]
    chart_data: Optional[dict[str, Any]]
    did_corrective: bool
    steps: int
    answer: str
    user_message: str


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def agent_node(state: AgentState) -> dict[str, Any]:
    """Let the model route to tools or write the final answer."""
    resp = LLM_TOOLS.invoke(state["messages"])
    update: dict[str, Any] = {"messages": [resp], "steps": state["steps"] + 1}
    if not resp.tool_calls:
        update["answer"] = (resp.content or "").strip()
    return update


def tools_node(state: AgentState) -> dict[str, Any]:
    """Execute the model's tool calls; build chart_data; apply corrective-RAG."""
    last = state["messages"][-1]
    seen = {
        (e["tool_name"], json.dumps(e["arguments"], sort_keys=True, default=str))
        for e in state["tool_calls_log"]
    }

    new_messages: list[AnyMessage] = []
    log_adds: list[dict[str, Any]] = []
    src_adds: list[str] = []
    chart_data = state.get("chart_data")
    did_corrective = state["did_corrective"]

    for tc in last.tool_calls:
        name = tc["name"]
        args = tc.get("args") or {}
        tc_id = tc["id"]

        if name not in DISPATCH:
            new_messages.append(
                ToolMessage(
                    content=f"Unknown tool '{name}'. Available tools: {', '.join(DISPATCH)}.",
                    tool_call_id=tc_id,
                )
            )
            continue

        dedupe_key = (name, json.dumps(args, sort_keys=True, default=str))
        if dedupe_key in seen:
            new_messages.append(
                ToolMessage(
                    content="This exact call was already made. Use the previous result and write the final answer.",
                    tool_call_id=tc_id,
                )
            )
            continue
        seen.add(dedupe_key)

        try:
            out = DISPATCH[name](**args)
            result = out.get("result", {})
            chart = out.get("chart")
        except Exception as exc:  # defensive: never let one bad call 500 the request
            result = {"error": str(exc)}
            chart = None

        # Corrective-RAG: grade document retrieval and re-retrieve once if weak.
        if name == "document_search" and not did_corrective and is_weak(result):
            did_corrective = True
            rq = _reformulate(state["user_message"])
            if rq and rq != args.get("query"):
                retry = DISPATCH["document_search"](query=rq)
                retry_result = retry.get("result", {})
                log_adds.append(
                    {
                        "tool_name": "document_search",
                        "arguments": {"query": rq, "reformulated": True},
                        "result": retry_result,
                    }
                )
                if not is_weak(retry_result):
                    result = retry_result

        log_adds.append({"tool_name": name, "arguments": args, "result": result})

        if chart is not None:
            chart_data = chart
        if name == "document_search":
            for s in result.get("source", []) or []:
                src_adds.append(s)

        new_messages.append(ToolMessage(content=_render_result(result), tool_call_id=tc_id))

    return {
        "messages": new_messages,
        "tool_calls_log": log_adds,
        "doc_sources": src_adds,
        "chart_data": chart_data,
        "did_corrective": did_corrective,
    }


def force_final_node(state: AgentState) -> dict[str, Any]:
    """Tool-call budget exhausted: answer the dangling calls with a note and
    force one no-tools synthesis so the message history stays valid."""
    last = state["messages"][-1]
    notes = [
        ToolMessage(
            content="Tool call budget reached. Write the final answer now from the information already gathered.",
            tool_call_id=tc["id"],
        )
        for tc in last.tool_calls
    ]
    try:
        resp = LLM_PLAIN.invoke(state["messages"] + notes)
        answer = (resp.content or "").strip()
    except Exception:
        answer = ""
    return {"messages": notes, "answer": answer}


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Language guard + citation safety-net footer."""
    answer = (state.get("answer") or "").strip()
    if not answer:
        answer = (
            "I could not produce a grounded answer from the available tools and data. "
            "Please try rephrasing your question."
        )

    # The local model occasionally drifts out of English; force it back.
    if _looks_non_english(answer):
        answer = _force_english(answer)

    # Provenance is shown in the UI's data-source panel, not inline — strip any
    # "(source: …)" citations or trailing "Sources:" footer the model emitted.
    answer = _strip_source_citations(answer)

    return {"answer": answer}


def _route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools" if state["steps"] < settings.MAX_TOOL_ITERS else "force_final"
    return "finalize"


# --------------------------------------------------------------------------- #
# Compile the graph once at import time.
# --------------------------------------------------------------------------- #
def _build_graph() -> Any:
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("force_final", force_final_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "force_final": "force_final", "finalize": "finalize"},
    )
    g.add_edge("tools", "agent")
    g.add_edge("force_final", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


GRAPH = _build_graph()


def _initial_state(message: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "messages": _build_messages(message, history),
        "tool_calls_log": [],
        "doc_sources": [],
        "chart_data": None,
        "did_corrective": False,
        "steps": 0,
        "answer": "",
        "user_message": message,
    }


def generate_chat_response(message: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    final = GRAPH.invoke(_initial_state(message, history))
    return {
        "answer": final["answer"],
        "tool_calls": final["tool_calls_log"],
        "chart_data": final["chart_data"],
    }


# --------------------------------------------------------------------------- #
# Streaming: surface every graph step live so the UI can visualise the agent
# loop as it runs. We translate LangGraph's per-node "updates" into a small,
# UI-friendly event vocabulary. The contract of the *final* event matches
# generate_chat_response exactly (answer, tool_calls, chart_data).
# --------------------------------------------------------------------------- #
_PREVIEW_CHARS = 600


def _result_preview(result: Any) -> str:
    try:
        text = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(result)
    if len(text) > _PREVIEW_CHARS:
        text = text[:_PREVIEW_CHARS] + " …"
    return text


def _doc_search_detail(result: Any) -> Optional[dict[str, Any]]:
    """Pull human-readable retrieval stats out of a document_search result."""
    if not isinstance(result, dict):
        return None
    hits = result.get("results") or result.get("matches") or result.get("chunks")
    scores = []
    if isinstance(hits, list):
        for h in hits:
            if isinstance(h, dict):
                s = h.get("score") or h.get("similarity") or h.get("distance")
                if isinstance(s, (int, float)):
                    scores.append(round(float(s), 3))
    sources = result.get("source") or result.get("sources") or []
    if isinstance(sources, str):
        sources = [sources]
    detail = {
        "k": len(hits) if isinstance(hits, list) else None,
        "scores": scores or None,
        "sources": list(sources) or None,
    }
    return detail if any(v for v in detail.values()) else None


# Canonical provenance for the deterministic CSV-backed tools. Each tool always
# draws on the same data file(s) regardless of which branch it returns, so we
# fall back to this when a tool result omits an explicit `source` (e.g. its
# "no data found for that year/species" branch). This guarantees data lineage
# is shown for every analytics answer, not just the happy-path returns.
# `document_search` is intentionally absent: its sources are the documents
# actually retrieved, so an empty result legitimately means "nothing retrieved".
TOOL_DATA_SOURCES: dict[str, list[str]] = {
    "get_catch": ["catch_clean.csv"],
    "get_largest_change": ["catch_clean.csv"],
    "compare_species_catch": ["catch_clean.csv"],
    "forecast_catch": ["catch_clean.csv"],
    "estimate_value": ["catch_clean.csv", "luke_clean.csv"],
    "rank_value_species": ["catch_clean.csv", "luke_clean.csv"],
    "value_trend_or_compare": ["catch_clean.csv", "luke_clean.csv"],
    "correlate_with_metric": ["catch_clean.csv", "water_quality_clean.csv"],
    "count_item_analysis": ["count_catch_clean.csv", "catch_clean.csv", "luke_clean.csv"],
}

# list_data_dimensions draws on a different file depending on which dimension
# was requested; map the argument to its source for the lineage fallback.
_DIMENSION_SOURCES: dict[str, str] = {
    "species": "catch_clean.csv",
    "count_items": "count_catch_clean.csv",
    "metrics": "water_quality_clean.csv",
}


def _extract_sources(result: Any, tool_name: str | None = None, arguments: Any = None) -> list[str]:
    """The data file(s) a tool result drew on (CSV datasets or KB documents).

    Prefers the explicit `source`/`sources` key on the result. When that is
    absent (an edge/no-data branch that forgot to set it), falls back to the
    canonical per-tool mapping so lineage stays visible for every answer.
    """
    if isinstance(result, dict):
        raw = result.get("source")
        if raw is None:
            raw = result.get("sources")
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            explicit = [s for s in raw if isinstance(s, str) and s]
            if explicit:
                return explicit

        # A missing/invalid-parameter early return (see app.agent_tools._missing)
        # short-circuits before any data file is read, so it has no provenance —
        # don't fabricate one via the fallback below.
        msg = result.get("message")
        if isinstance(msg, str) and msg.startswith("Missing required parameter"):
            return []

    if tool_name in TOOL_DATA_SOURCES:
        return list(TOOL_DATA_SOURCES[tool_name])
    if tool_name == "list_data_dimensions" and isinstance(arguments, dict):
        kind = str(arguments.get("kind") or "").strip().lower()
        if kind in _DIMENSION_SOURCES:
            return [_DIMENSION_SOURCES[kind]]
    return []


def _source_type(name: str) -> str:
    return "dataset" if name.lower().endswith(".csv") else "document"


def _msg_model(resp: Any) -> Optional[str]:
    meta = getattr(resp, "response_metadata", None) or {}
    return meta.get("model_name") or meta.get("model")


def _msg_tokens(resp: Any) -> dict[str, Any]:
    usage = getattr(resp, "usage_metadata", None) or {}
    return {"in": usage.get("input_tokens"), "out": usage.get("output_tokens")}


def stream_chat_response(message: str, history: list[dict[str, Any]]):
    """Yield rich UI events as the graph executes, so the front end can show
    exactly how the answer is produced: which model ran, how long each step
    took, token usage, the tools called and what they returned.

    Event shapes (all dicts with a "type"):
      {"type": "start", "models": {...}, "config": {...}}
      {"type": "node", "node": "agent|tools|finalize", "status": "active", "iteration": N}
      {"type": "agent_step", "iteration", "model", "elapsed_ms", "tokens", "action", "tools"}
      {"type": "tool_result", "tool_name", "category", "arguments", "preview",
                              "retrieval"?, "elapsed_ms"}
      {"type": "corrective_rag", "query"}
      {"type": "finalize_step", "elapsed_ms", "language_guard", "sources"}
      {"type": "final", "answer", "tool_calls", "chart_data", "total_ms",
                        "iterations", "tool_count"}
      {"type": "error", "message"}
    """
    full_log: list[dict[str, Any]] = []
    chart_data: Optional[dict[str, Any]] = None
    answer = ""
    iteration = 0
    all_sources: list[str] = []
    # data-lineage accumulators: which tools hit which data files
    source_stats: dict[str, dict[str, Any]] = {}
    lineage_links: list[dict[str, str]] = []
    seen_links: set[tuple[str, str]] = set()
    t_start = time.perf_counter()
    last_t = t_start

    yield {
        "type": "start",
        "models": {
            "agent": settings.OLLAMA_MODEL,
            "fallback": settings.OLLAMA_MODEL_FALLBACK,
            "embeddings": settings.EMBEDDING_MODEL,
            "runtime": "Ollama",
            "endpoint": settings.OLLAMA_BASE_URL,
        },
        "config": {
            "rag_top_k": settings.RAG_TOP_K,
            "rag_score_floor": settings.RAG_SCORE_FLOOR,
            "max_tool_iters": settings.MAX_TOOL_ITERS,
        },
    }
    # The graph always enters `agent` first — light it up before the (slow)
    # first LLM call so the wait itself reads as "the agent is thinking".
    yield {"type": "node", "node": "agent", "status": "active", "iteration": 1}

    try:
        for update in GRAPH.stream(_initial_state(message, history), stream_mode="updates"):
            now = time.perf_counter()
            elapsed_ms = int((now - last_t) * 1000)
            last_t = now

            for node, data in update.items():
                if node == "agent":
                    iteration += 1
                    msgs = data.get("messages") or []
                    resp = msgs[-1] if msgs else None
                    tool_calls = list(getattr(resp, "tool_calls", None) or [])
                    yield {
                        "type": "agent_step",
                        "iteration": iteration,
                        "model": _msg_model(resp) or settings.OLLAMA_MODEL,
                        "elapsed_ms": elapsed_ms,
                        "tokens": _msg_tokens(resp),
                        "action": "call_tools" if tool_calls else "final",
                        "tools": [tc.get("name") for tc in tool_calls],
                    }
                    if tool_calls:
                        yield {"type": "node", "node": "tools", "status": "active"}
                    else:
                        if data.get("answer"):
                            answer = data["answer"]
                        yield {"type": "node", "node": "finalize", "status": "active"}

                elif node == "tools":
                    entries = data.get("tool_calls_log") or []
                    per_tool_ms = int(elapsed_ms / max(len(entries), 1))
                    for entry in entries:
                        full_log.append(entry)
                        name = entry["tool_name"]
                        if entry.get("arguments", {}).get("reformulated"):
                            yield {
                                "type": "corrective_rag",
                                "query": entry["arguments"].get("query", ""),
                            }
                        retrieval = _doc_search_detail(entry["result"]) if name == "document_search" else None
                        if retrieval and retrieval.get("sources"):
                            all_sources.extend(retrieval["sources"])

                        # Provenance: which data file(s) this tool call hit.
                        data_sources = _extract_sources(entry["result"], name, entry.get("arguments"))
                        for src in data_sources:
                            stat = source_stats.setdefault(
                                src, {"name": src, "type": _source_type(src), "hits": 0, "tools": []}
                            )
                            stat["hits"] += 1
                            if name not in stat["tools"]:
                                stat["tools"].append(name)
                            if (name, src) not in seen_links:
                                seen_links.add((name, src))
                                lineage_links.append({"tool": name, "source": src, "type": _source_type(src)})

                        yield {
                            "type": "tool_result",
                            "tool_name": name,
                            "category": "retrieval" if name == "document_search" else "analytics",
                            "arguments": entry["arguments"],
                            "preview": _result_preview(entry["result"]),
                            "retrieval": retrieval,
                            "data_sources": data_sources,
                            "elapsed_ms": per_tool_ms,
                        }
                    if data.get("chart_data") is not None:
                        chart_data = data["chart_data"]
                    # Loop back to the agent for the next decision.
                    yield {"type": "node", "node": "agent", "status": "active", "iteration": iteration + 1}

                elif node == "force_final":
                    if data.get("answer"):
                        answer = data["answer"]
                    yield {"type": "node", "node": "finalize", "status": "active"}

                elif node == "finalize":
                    prev = answer
                    answer = data.get("answer") or answer
                    yield {
                        "type": "finalize_step",
                        "elapsed_ms": elapsed_ms,
                        "language_guard": _looks_non_english(prev) if prev else False,
                        "sources": list(dict.fromkeys(all_sources)) or None,
                    }

        yield {
            "type": "final",
            "answer": answer,
            "tool_calls": full_log,
            "chart_data": chart_data,
            "total_ms": int((time.perf_counter() - t_start) * 1000),
            "iterations": iteration,
            "tool_count": len(full_log),
            "data_sources": sorted(source_stats.values(), key=lambda s: (-s["hits"], s["name"])),
            "lineage": lineage_links,
        }
    except Exception as exc:  # never break the SSE stream with a raw traceback
        yield {"type": "error", "message": str(exc)}
