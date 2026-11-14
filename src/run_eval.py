"""Run retrieval benchmarks and real Agentic RAG evaluations on dev/test splits."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agentic import (
    Bm25Index,
    HybridRrfRetriever,
    LlmRerankRetriever,
    OpenRouterController,
    OpenRouterEmbedder,
    PgVectorDenseRetriever,
    TfidfRetriever,
    run_agent,
)
from baseline import hit, pack_context


def build_retriever(method: str, chunks: list[dict[str, Any]], embedder: OpenRouterEmbedder | None = None):
    if method == 'tfidf':
        return TfidfRetriever(chunks), None
    if method == 'bm25':
        return Bm25Index(chunks), None
    emb = embedder or OpenRouterEmbedder()
    dense = PgVectorDenseRetriever(embedder=emb)
    if method == 'dense':
        return dense, None
    bm25 = Bm25Index(chunks)
    hybrid = HybridRrfRetriever(dense=dense, lexical=bm25, rrf_k=60, pool_k=15)
    if method == 'hybrid_rrf':
        return hybrid, None
    if method == 'hybrid_rerank':
        reranker = LlmRerankRetriever(base=hybrid, model='openai/gpt-4o-mini', pool_k=10)
        return reranker, reranker
    raise ValueError(f'Unknown retrieval method: {method}')


def eval_retrieval(method: str, chunks: list[dict[str, Any]], questions: list[dict[str, Any]],
                   split: str, budget: int, out_dir: Path) -> dict[str, Any]:
    embedder = OpenRouterEmbedder() if method in ('dense', 'hybrid_rrf', 'hybrid_rerank') else None
    retriever, reranker = build_retriever(method, chunks, embedder=embedder)
    selected = [q for q in questions if q.get('split') == split]
    traces = []
    for q in selected:
        t0 = time.perf_counter()
        results = retriever.search(q['question'], filters=q.get('filter'), top_k=3)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]
        traces.append({
            'query_id': q['id'],
            'split': split,
            'question': q['question'],
            'filter': q.get('filter'),
            'method': method,
            'budget_chars': budget,
            'results': marked,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': round(elapsed_ms, 2),
            'hit1': bool(marked and marked[0]['hit']),
            'hit3': any(c['hit'] for c in marked),
            'context_hit': any(c['hit'] and c['id'] in ids for c in marked),
        })
    n = len(traces)
    emb_cost = embedder.total_cost_usd if embedder else 0.0
    rerank_cost = reranker.total_cost_usd if reranker else 0.0
    summary = {
        'method': method,
        'split': split,
        'budget_chars': budget,
        'n': n,
        'hit1': sum(t['hit1'] for t in traces),
        'hit1_rate': round(sum(t['hit1'] for t in traces) / n, 4) if n else 0.0,
        'hit3': sum(t['hit3'] for t in traces),
        'hit3_rate': round(sum(t['hit3'] for t in traces) / n, 4) if n else 0.0,
        'context_hit': sum(t['context_hit'] for t in traces),
        'context_hit_rate': round(sum(t['context_hit'] for t in traces) / n, 4) if n else 0.0,
        'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces), 1) if n else 0.0,
        'median_ms': round(statistics.median(t['elapsed_ms'] for t in traces), 2) if n else 0.0,
        'embed_tokens': embedder.total_tokens if embedder else 0,
        'rerank_tokens': reranker.total_tokens if reranker else 0,
        'total_cost_usd': round(emb_cost + rerank_cost, 6),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def eval_agent(method: str, chunks: list[dict[str, Any]], cases: list[dict[str, Any]],
               questions: list[dict[str, Any]], split: str, budget: int, max_searches: int,
               top_k: int, model: str, out_dir: Path) -> dict[str, Any]:
    embedder = OpenRouterEmbedder() if method in ('dense', 'hybrid_rrf', 'hybrid_rerank') else None
    retriever, reranker = build_retriever(method, chunks, embedder=embedder)
    q_by_id = {q['id']: q for q in questions}
    selected = [c for c in cases if c.get('split') == split]
    traces = []
    total_prompt = 0
    total_completion = 0
    total_llm_cost = 0.0
    for case in selected:
        controller = OpenRouterController(model=model)
        flt = q_by_id.get(case['id'], {}).get('filter')
        t0 = time.perf_counter()
        try:
            res = run_agent(
                case['question'],
                controller,
                retriever,
                filters=flt,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget,
            )
            err = None
        except Exception as exc:
            res = {'status': 'error', 'answer': '', 'citations': [], 'events': []}
            err = f'{type(exc).__name__}: {exc}'
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_prompt += controller.total_prompt_tokens
        total_completion += controller.total_completion_tokens
        total_llm_cost += controller.total_cost_usd
        search_calls = sum(1 for e in res.get('events', []) if e.get('event') == 'search')
        traces.append({
            'query_id': case['id'],
            'split': split,
            'answerable': case['answerable'],
            'question': case['question'],
            'filter': flt,
            'status': res['status'],
            'answer': res['answer'],
            'citations': res['citations'],
            'search_calls': search_calls,
            'events': res['events'],
            'elapsed_ms': round(elapsed_ms, 2),
            'prompt_tokens': controller.total_prompt_tokens,
            'completion_tokens': controller.total_completion_tokens,
            'llm_cost_usd': round(controller.total_cost_usd, 6),
            'error': err,
        })
    emb_cost = embedder.total_cost_usd if embedder else 0.0
    rerank_cost = reranker.total_cost_usd if reranker else 0.0
    summary = {
        'mode': 'agent',
        'method': method,
        'model': model,
        'split': split,
        'budget_chars': budget,
        'max_searches': max_searches,
        'top_k': top_k,
        'n': len(traces),
        'answered': sum(1 for t in traces if t['status'] == 'answered'),
        'abstain': sum(1 for t in traces if t['status'] == 'abstain'),
        'errors': sum(1 for t in traces if t['status'] == 'error'),
        'avg_searches': round(statistics.mean(t['search_calls'] for t in traces), 2) if traces else 0.0,
        'median_ms': round(statistics.median(t['elapsed_ms'] for t in traces), 2) if traces else 0.0,
        'total_prompt_tokens': total_prompt,
        'total_completion_tokens': total_completion,
        'total_cost_usd': round(total_llm_cost + emb_cost + rerank_cost, 6),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description='Run Day 7 retrieval & agent experiments')
    parser.add_argument('--mode', required=True, choices=('retrieval', 'agent'))
    parser.add_argument('--method', required=True, choices=('tfidf', 'bm25', 'dense', 'hybrid_rrf', 'hybrid_rerank'))
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--chunks', type=Path, default=Path('data/day6_converted/chunks.json'))
    parser.add_argument('--questions', type=Path, default=Path('data/day6_converted/questions.json'))
    parser.add_argument('--cases', type=Path, default=Path('data/day6_converted/agent_cases.json'))
    parser.add_argument('--budget', type=int, default=800)
    parser.add_argument('--max-searches', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--model', default='openai/gpt-4o-mini')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    if args.mode == 'retrieval':
        eval_retrieval(args.method, chunks, questions, args.split, args.budget, args.output)
    else:
        cases = json.loads(args.cases.read_text(encoding='utf-8'))
        eval_agent(args.method, chunks, cases, questions, args.split, args.budget, args.max_searches, args.top_k, args.model, args.output)


if __name__ == '__main__':
    main()

