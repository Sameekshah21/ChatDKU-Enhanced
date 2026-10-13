"""ChatDKU Improvement 1 v6: automatic coverage-first RAG with fast citation diagnostics.

Integrated enhanced ChatDKU implementation. The CLI and Streamlit use this module.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from urllib.request import Request, urlopen

from chatdku.config import get_settings
from chatdku.tools.vector_retriever import VectorRetriever
from chatdku.tools.keyword_retriever import KeywordRetriever


def source_name(doc):
    return str(doc.metadata.get("source_path") or doc.metadata.get("file_name") or "Unknown source")


def query_plan(question):
    """Conservative decomposition: preserve original question, add focused searches."""
    q = question.strip()
    queries = [q]
    # These are SEARCH HINTS, not asserted facts or graduation requirements.
    topics = [
        (r"graduat|degree requirement|complete.*degree", [
            "graduation degree requirements minimum credits Duke DKU Chinese international",
            "general education common core distribution graduation requirements",
            "major requirements degree graduation declaration courses",
            "academic standing GPA graduation satisfactory progress minimum",
            "graduation exceptions transfer credits AP credit no credit residency",
        ]),
        (r"major|declar", [
            "major declaration requirements deadlines advising",
            "major coursework prerequisites degree requirements",
        ]),
        (r"advis|advisor", [
            "academic advisor responsibilities students curriculum educational planning",
            "advising appointments registration major declaration",
        ]),
        (r"credit|transfer|AP\b", [
            "minimum graduation credits transfer credit prematriculation AP",
            "credit no credit grading restrictions degree requirements",
        ]),
        (r"academic polic|academic stand|GPA|probation", [
            "academic standing GPA probation satisfactory academic progress",
            "academic policies withdrawal repeat courses exceptions",
        ]),
    ]
    for pattern, hints in topics:
        if re.search(pattern, q, re.I):
            queries.extend(hints)
    # Also search individual clauses of multi-part questions.
    parts = re.split(r"[,;?]|\b(?:including|as well as)\b", q, flags=re.I)
    queries.extend(part.strip() for part in parts if len(part.strip().split()) >= 3)
    unique = []
    for item in queries:
        if item.casefold() not in {x.casefold() for x in unique}:
            unique.append(item)
    return unique[:12]


def retrieve(queries, per_query, max_passages):
    vector = VectorRetriever()
    keyword = KeywordRetriever()
    fused = {}
    durations = defaultdict(float)
    for qi, query in enumerate(queries):
        for channel, retriever in (("vector", vector), ("keyword", keyword)):
            started = time.perf_counter()
            docs = retriever.query(query, top_k=per_query)
            durations[channel] += time.perf_counter() - started
            for rank, doc in enumerate(docs, 1):
                entry = fused.setdefault(doc.node_id, {"doc": doc, "score": 0.0, "topics": set(), "channels": set()})
                # Reciprocal rank fusion; boost nothing based on incomparable raw scores.
                entry["score"] += 1 / (60 + rank)
                entry["topics"].add(qi)
                entry["channels"].add(channel)
    # Greedy coverage-aware selection, rewarding uncovered subqueries.
    remaining = list(fused.values())
    selected = []
    covered = set()
    while remaining and len(selected) < max_passages:
        best = max(remaining, key=lambda x: (
            x["score"] + 0.035 * len(x["topics"] - covered),
            len(x["topics"]), x["doc"].node_id
        ))
        selected.append(best)
        covered.update(best["topics"])
        remaining.remove(best)
    return selected, len(fused), durations, covered


def build_evidence(selected, max_chars):
    """Preserve citation identity and allocate evidence fairly across passages."""
    if not selected:
        return ""
    headers = []
    for i, item in enumerate(selected, 1):
        d = item["doc"]
        headers.append(
            f"[{i}] Source: {source_name(d)}\n"
            f"URL: {d.metadata.get('url') or 'not provided'}\n"
            f"Node ID: {d.node_id}\nExcerpt:\n"
        )
    separator = "\n\n---\n\n"
    remaining = max_chars - sum(map(len, headers)) - (len(selected)-1)*len(separator)
    if remaining < 250 * len(selected):
        raise ValueError("Evidence budget too small for selected passage count. Increase --max-chars or reduce --passages.")
    texts = [x["doc"].text.strip() for x in selected]
    allowance = [min(len(t), remaining // len(texts)) for t in texts]
    spare = remaining - sum(allowance)
    while spare:
        needy = [i for i, t in enumerate(texts) if len(t) > allowance[i]]
        if not needy:
            break
        for i in needy:
            delta = min(len(texts[i])-allowance[i], max(1, spare // len(needy)))
            allowance[i] += delta
            spare -= delta
            if spare <= 0:
                break
    return separator.join(
        h + t[:n] + ("\n[EXCERPT TRUNCATED]" if n < len(t) else "")
        for h, t, n in zip(headers, texts, allowance)
    )


def ollama(messages, tokens, timeout):
    settings = get_settings()
    base = settings.llm_base_url.rstrip("/")
    if not base.endswith("/v1"):
        raise ValueError("Expected CHATDKU_LLM_BASE_URL ending in /v1 (e.g. http://localhost:11434/v1)")
    endpoint = base[:-3] + "/api/chat"
    payload = {
        "model": settings.llm_model.removeprefix("openai/"),
        "stream": False,
        "think": False,
        "format": {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]},
        "messages": messages,
        "options": {"temperature": 0, "num_predict": tokens, "num_ctx": min(12288, max(4096, 2 ** ((max(1, sum(len(m["content"]) for m in messages) // 4 + tokens + 256) - 1).bit_length())))},
    }
    request = Request(endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result, time.perf_counter()-started


def final_content(message):
    content = str(message.get("content") or "")
    content = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", content, flags=re.I | re.S)
    endings = list(re.finditer(r"</think\s*>", content, flags=re.I))
    if endings:
        content = content[endings[-1].end():]
    opening = re.search(r"<think\b[^>]*>", content, flags=re.I)
    if opening:
        content = content[:opening.start()]
    content = content.strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
            return parsed["answer"].strip()
    except (ValueError, TypeError):
        pass
    return content


SYSTEM = (
    "You are ChatDKU, a helpful DKU student information assistant. "
    "Return only a JSON object with one key: answer. The answer must be the FINAL student-facing response, "
    "not your reasoning, notes, planning, self-review, or source analysis. "
    "Start answering immediately. Use only the provided source excerpts for DKU-specific facts. "
    "Cite supporting excerpt numbers [1], [2] after policy claims. "
    "Cover every part of the student's question, including important conditions and exceptions supported by sources. "
    "Do not invent policies, deadlines, or requirements. If evidence is missing, say so briefly. "
    "For simple questions, use one short paragraph or 3-5 compact bullets. "
    "For multi-part questions, use organized sections with necessary detail. "
    "Do not copy source excerpts verbatim at length. Never output chain-of-thought. /no_think"
)

AUDIT_SYSTEM = (
    "You are a strict evidence auditor. Compare ANSWER with SOURCE EXCERPTS. "
    "For each substantive DKU-specific claim, check whether its cited excerpts actually support it. "
    "Also check each requested topic for omissions where excerpts provide an answer. "
    "Return ONLY valid JSON with keys: unsupported (list of strings describing claims), "
    "missing (list of strings describing supported omitted details), "
    "citation_problems (list of strings), and notes (list of strings). "
    "Do not invent missing DKU facts. An empty list means no issue found. "
    "This is an advisory audit, not proof of correctness."
)


def automatic_budget(question, queries):
    """Adaptive evidence and generation budgets, with no user configuration needed."""
    detailed = bool(re.search(r"\b(all|every|comprehensive|in detail|exceptions|requirements|policies|compare|differences)\b", question, re.I))
    multipart = len(re.findall(r"[?;]", question)) > 1 or len(question.split()) > 35
    if detailed or multipart:
        return {"per_query": 5, "passages": 10, "max_chars": 18000, "output_tokens": 1900}
    return {"per_query": 4, "passages": 5, "max_chars": 8500, "output_tokens": 850}



def answer_question(question: str, *, timeout: int = 900) -> dict:
    """Run integrated evidence-grounded v6 path; return structured diagnostics.

    Ollama /api/chat is required for this implementation. The legacy agent
    remains available via the standard `agent` command.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty")
    queries = query_plan(question)
    if len(question.split()) <= 25 and not re.search(
        r"\b(all|every|comprehensive|in detail|exceptions|requirements|policies|compare)\b", question, re.I
    ):
        queries = queries[:2]
    budget = automatic_budget(question, queries)
    selected, unique, timings, covered = retrieve(queries, budget["per_query"], budget["passages"])
    if not selected:
        return {"answer": "I couldn't find indexed DKU evidence for this question.",
                "sources": [], "finish": "no_evidence", "complete": False,
                "candidate_count": unique}
    evidence = build_evidence(selected, budget["max_chars"])
    prompt = (f"SOURCE EXCERPTS:\n{evidence}\n\nORIGINAL USER QUESTION:\n{question}\n\n"
              "Answer directly and completely with evidence-linked citations.")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    result, generation = ollama(messages, budget["output_tokens"], timeout)
    answer = final_content(result.get("message") or {})
    retry_used = False
    if result.get("done_reason") == "length" or not answer or answer.startswith("{"):
        retry_used = True
        second, seconds = ollama(
            [{"role": "system", "content": SYSTEM + " Output the answer immediately."},
             {"role": "user", "content": prompt + "\nReturn the final answer now, without deliberation."}],
            min(3200, max(budget["output_tokens"] * 2, 1400)), timeout)
        generation += seconds
        retry_answer = final_content(second.get("message") or {})
        if retry_answer and not retry_answer.startswith("{"):
            result, answer = second, retry_answer
    complete = bool(answer and not answer.startswith("{") and result.get("done_reason") != "length")
    cited = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    bad_citations = sorted(n for n in cited if n < 1 or n > len(selected))
    sources = [{"number": i, "name": source_name(item["doc"]),
                "node_id": str(item["doc"].node_id)}
               for i, item in enumerate(selected, 1)]
    return {"answer": answer if complete else (answer or "[No final answer generated]"),
            "complete": complete, "finish": result.get("done_reason", "unknown"),
            "retry_used": retry_used, "sources": sources, "bad_citations": bad_citations,
            "candidate_count": unique, "covered_queries": len(covered),
            "queries": len(queries), "generation_seconds": generation,
            "retrieval_seconds": dict(timings)}


def main():
    p = argparse.ArgumentParser(description="ChatDKU multi-query, quality-first RAG")
    p.add_argument("question", nargs="+")
    p.add_argument("--per-query", type=int, default=None, help="Developer override; normally automatic")
    p.add_argument("--passages", type=int, default=None, help="Developer override; normally automatic")
    p.add_argument("--max-chars", type=int, default=None, help="Developer override; normally automatic")
    p.add_argument("--output-tokens", type=int, default=None, help="Developer override; normally automatic")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--audit", action="store_true", help="Optional slow experimental AI audit (not recommended on CPU)")
    args = p.parse_args()
    question = " ".join(args.question).strip()
    t0 = time.perf_counter()
    queries = query_plan(question)
    # For straightforward questions, avoid repetitive retrieval passes.
    if len(question.split()) <= 25 and not re.search(r"\b(all|every|comprehensive|in detail|exceptions|requirements|policies|compare)\b", question, re.I):
        queries = queries[:2]
    budget = automatic_budget(question, queries)
    for field in ("per_query", "passages", "max_chars", "output_tokens"):
        if getattr(args, field) is None:
            setattr(args, field, budget[field])
    if min(args.per_query, args.passages, args.max_chars, args.output_tokens, args.timeout) < 1:
        p.error("All numeric arguments must be positive")
    print(f"Automatically selected retrieval/generation budgets: {args.passages} passages, {args.output_tokens} output tokens. AI audit: {'enabled' if args.audit else 'off'}.", flush=True)
    print(f"Searching {len(queries)} focused queries:", flush=True)
    for q in queries:
        print(" - " + q, flush=True)
    selected, unique, timings, covered = retrieve(queries, args.per_query, args.passages)
    if not selected:
        print("No indexed evidence found. Cannot provide a grounded answer.")
        return
    evidence = build_evidence(selected, args.max_chars)
    print(f"Retrieved {unique} unique candidates; selected {len(selected)} passages ({len(evidence)} characters).", flush=True)
    print(f"Query coverage: {len(covered)}/{len(queries)}; vector {timings['vector']:.1f}s; keyword {timings['keyword']:.1f}s", flush=True)
    prompt = (
        f"SOURCE EXCERPTS:\n{evidence}\n\n"
        f"ORIGINAL USER QUESTION:\n{question}\n\n"
        "Answer directly and completely with evidence-linked citations. "
        "Do not include unsupported claims."
    )
    result, generation = ollama(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        args.output_tokens, args.timeout
    )
    answer = final_content(result.get("message") or {})
    # A length stop can mean truncated JSON or incomplete response. Retry once with
    # a larger budget and stronger final-answer-only instruction, never silently
    # label a truncated answer complete.
    if result.get("done_reason") == "length" or not answer or answer.startswith("{"):
        print("First generation incomplete; retrying once with a larger final-answer budget...", flush=True)
        retry_messages = [
            {"role": "system", "content": SYSTEM + " IMPORTANT: Output the final answer immediately, no preamble."},
            {"role": "user", "content": prompt + "\nProvide the final response now. Do not deliberate."},
        ]
        retry, retry_time = ollama(retry_messages, min(3200, max(args.output_tokens * 2, 1400)), args.timeout)
        retry_answer = final_content(retry.get("message") or {})
        generation += retry_time
        if retry_answer and not retry_answer.startswith("{"):
            result, answer = retry, retry_answer
    print("\nANSWER:\n" + (answer or "[NO FINAL ANSWER GENERATED]"))
    print("\nSOURCES:")
    for i, item in enumerate(selected, 1):
        d = item["doc"]
        print(f"[{i}] {source_name(d)} | node={d.node_id} | URL={d.metadata.get('url') or 'not provided'}")
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    bad = sorted(n for n in cited if n < 1 or n > len(selected))
    if bad:
        print(f"\nCITATION WARNING: nonexistent citation numbers: {bad}")
    if not cited:
        print("\nCITATION WARNING: no citations in answer")
    # Fast deterministic checks. These check citation form, not factual entailment.
    chunks = re.split(r"(?<=[.!?])\s+|\n+", answer)
    uncited = [c.strip()[:170] for c in chunks if len(c.strip()) >= 75 and not re.search(r"\[\d+\]", c) and not c.lstrip().startswith(("#", "-", "*"))]
    if uncited:
        print(f"\nCITATION REVIEW: {len(uncited)} substantial text segments lack an inline numeric citation (heuristic):")
        for segment in uncited[:6]:
            print(" - " + segment)
    print("CITATION LIMITATION: References are not verified for factual entailment.")
    print(f"\nGeneration: {generation:.1f}s | finish: {result.get('done_reason', 'unknown')} | output tokens: {result.get('eval_count', 'unknown')}")
    if result.get("done_reason") == "length" or not answer:
        print("WARNING: Incomplete or missing final answer. Do not rely on this result.")
    if args.audit and answer:
        print("\nChecking factual support and coverage against retrieved evidence...", flush=True)
        audit_prompt = f"SOURCE EXCERPTS:\n{evidence}\n\nQUESTION:\n{question}\n\nANSWER TO AUDIT:\n{answer}"
        audit_result, audit_time = ollama(
            [{"role": "system", "content": AUDIT_SYSTEM},
             {"role": "user", "content": audit_prompt}],
            min(1400, args.output_tokens), args.timeout
        )
        audit_text = final_content(audit_result.get("message") or {})
        try:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", audit_text.strip(), flags=re.I)
            audit = json.loads(cleaned)
            print("AUDIT (model-generated, not independent verification):")
            print(json.dumps(audit, indent=2, ensure_ascii=False))
        except (ValueError, TypeError):
            print("AUDIT could not be parsed; do not treat it as passed.")
            print(audit_text[:2500])
        print(f"Audit time: {audit_time:.1f}s | audit finish: {audit_result.get('done_reason', 'unknown')}")
    print(f"\nTOTAL TIME: {time.perf_counter()-t0:.1f}s")
    print("NOTE: Citation checks do not establish factual accuracy; verify consequential policies in the DKU bulletin.")


if __name__ == "__main__":
    main()
