"""Multimodal Agentic RAG pipeline for the 100-page GPT-4 Technical Report (ADVANCED.md)."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import pymupdf
import requests
from dotenv import load_dotenv

from agentic import Bm25Index, OpenRouterEmbedder


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
    """Split page text into overlapping chunks."""
    clean = re.sub(r'\n{3,}', '\n\n', text).strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        if end < len(clean):
            split_at = clean.rfind('\n', start + max_chars // 2, end)
            if split_at == -1:
                split_at = clean.rfind(' ', start + max_chars // 2, end)
            if split_at > start:
                end = split_at
        piece = clean[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


def openrouter_vision(image_path: Path, prompt: str, api_key: str, model: str = 'openai/gpt-4o-mini') -> tuple[str, int, float]:
    """Call OpenRouter Vision model on a rendered PDF page PNG."""
    b64 = base64.b64encode(image_path.read_bytes()).decode('ascii')
    resp = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/day07-retrieval-context',
            'X-Title': 'Day07-Advanced-Multimodal',
        },
        json={
            'model': model,
            'temperature': 0.0,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                    ],
                }
            ],
        },
        timeout=90,
    )
    resp.raise_for_status()
    payload = resp.json()
    usage = payload.get('usage') or {}
    tokens = int(usage.get('total_tokens') or 0)
    cost = float(usage['cost']) if usage.get('cost') is not None else 0.0
    return payload['choices'][0]['message']['content'].strip(), tokens, cost


def embed_batch(texts: list[str], api_key: str, model: str = 'openai/text-embedding-3-small',
                batch_size: int = 64) -> tuple[list[list[float]], int, float]:
    """Embed texts in batches via OpenRouter."""
    vectors: list[list[float]] = []
    total_tokens = 0
    total_cost = 0.0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = requests.post(
            'https://openrouter.ai/api/v1/embeddings',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://github.com/day07-retrieval-context',
                'X-Title': 'Day07-Advanced-Multimodal',
            },
            json={'model': model, 'input': batch},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        usage = payload.get('usage') or {}
        tokens = int(usage.get('total_tokens') or 0)
        total_tokens += tokens
        if usage.get('cost') is not None:
            total_cost += float(usage['cost'])
        else:
            total_cost += tokens * 0.02 / 1_000_000
        data_sorted = sorted(payload['data'], key=lambda d: d['index'])
        vectors.extend(d['embedding'] for d in data_sorted)
    return vectors, total_tokens, total_cost


def build_index(pdf_path: Path, out_dir: Path, model: str = 'openai/gpt-4o-mini') -> dict[str, Any]:
    """Index all 100 pages (text + rendered visual pages with Vision summaries + embeddings)."""
    api_key = os.getenv('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('OPENROUTER_API_KEY is required')
    t0 = time.perf_counter()
    pages_dir = out_dir / 'pages'
    pages_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    chunks: list[dict[str, Any]] = []
    visual_pages: list[int] = []
    empty_pages: list[int] = []
    vision_tokens = 0
    vision_cost = 0.0

    for page_num in range(1, len(doc) + 1):
        page = doc[page_num - 1]
        raw_text = page.get_text('text') or ''
        has_img = bool(page.get_images())
        has_fig_tbl = bool(re.search(r'(?i)\b(Figure|Table)\s+\d+', raw_text))
        is_visual = has_img or has_fig_tbl

        img_rel = None
        if is_visual:
            visual_pages.append(page_num)
            img_path = pages_dir / f'page_{page_num:03d}.png'
            if not img_path.exists():
                pix = page.get_pixmap(dpi=150)
                pix.save(str(img_path))
            img_rel = str(img_path.as_posix())

            vis_prompt = (
                f"You are indexing page {page_num} of the GPT-4 Technical Report.\n"
                "Describe EVERY table, chart/plot, diagram, or image on this page in structured detail.\n"
                "- For tables: list table title, columns, rows, exact numbers, and units.\n"
                "- For charts/figures: list figure number, title, x/y axes, legend labels, units, trends, and key values.\n"
                "- For images/screenshots: describe all visual elements, text inside the image, and what it illustrates.\n"
                "If this page has no actual table/chart/image (only text mentioning Figure/Table), reply 'NO_VISUAL_ELEMENT'."
            )
            summary_txt, tok, cst = openrouter_vision(img_path, vis_prompt, api_key, model=model)
            vision_tokens += tok
            vision_cost += cst
            if summary_txt and 'NO_VISUAL_ELEMENT' not in summary_txt:
                chunks.append({
                    'id': f'gpt4-p{page_num:03d}-v00',
                    'source_id': 'gpt4-tech-report',
                    'source': pdf_path.name,
                    'page': page_num,
                    'text': f'[Visual Summary | Page {page_num}]\n{summary_txt}',
                    'metadata': {'modality': 'visual', 'image_path': img_rel},
                })

        text_pieces = chunk_text(raw_text, max_chars=900, overlap=150)
        if not text_pieces and not is_visual:
            empty_pages.append(page_num)
        for idx, piece in enumerate(text_pieces):
            chunks.append({
                'id': f'gpt4-p{page_num:03d}-t{idx:02d}',
                'source_id': 'gpt4-tech-report',
                'source': pdf_path.name,
                'page': page_num,
                'text': piece,
                'metadata': {'modality': 'text', 'image_path': img_rel},
            })

    texts = [c['text'] for c in chunks]
    vectors, emb_tokens, emb_cost = embed_batch(texts, api_key)

    (out_dir / 'index_chunks.json').write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'index_embeddings.json').write_text(json.dumps(vectors), encoding='utf-8')

    stats = {
        'pdf_path': str(pdf_path.as_posix()),
        'total_pages': len(doc),
        'total_chunks': len(chunks),
        'text_chunks': sum(1 for c in chunks if c['metadata']['modality'] == 'text'),
        'visual_chunks': sum(1 for c in chunks if c['metadata']['modality'] == 'visual'),
        'visual_pages_scanned': visual_pages,
        'empty_pages': empty_pages,
        'embedding_model': 'openai/text-embedding-3-small',
        'embedding_dimensions': len(vectors[0]) if vectors else 0,
        'vision_model': model,
        'embed_tokens': emb_tokens,
        'vision_tokens': vision_tokens,
        'total_cost_usd': round(emb_cost + vision_cost, 6),
        'elapsed_seconds': round(time.perf_counter() - t0, 2),
    }
    (out_dir / 'index_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


class LocalHybridRetriever:
    """Hybrid RRF (Dense cosine over indexed embeddings + BM25) on 100-page PDF."""

    def __init__(self, chunks: list[dict[str, Any]], vectors: list[list[float]], embedder: OpenRouterEmbedder) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.norms = [math.sqrt(sum(x * x for x in v)) or 1.0 for v in vectors]
        self.bm25 = Bm25Index(chunks)
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5, pool_k: int = 20, rrf_k: int = 60) -> list[dict[str, Any]]:
        q_vec = self.embedder.embed_query(query)
        q_norm = math.sqrt(sum(x * x for x in q_vec)) or 1.0
        dense_scored = []
        for chunk, vec, norm in zip(self.chunks, self.vectors, self.norms):
            sim = sum(a * b for a, b in zip(q_vec, vec)) / (q_norm * norm)
            dense_scored.append((sim, chunk))
        dense_scored.sort(key=lambda x: (-x[0], x[1]['id']))
        dense_hits = [item[1] for item in dense_scored[:pool_k]]
        lex_hits = self.bm25.search(query, top_k=pool_k)

        rrf_scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for rank, c in enumerate(dense_hits, start=1):
            by_id[c['id']] = c
            rrf_scores[c['id']] = rrf_scores.get(c['id'], 0.0) + 1.0 / (rrf_k + rank)
        for rank, c in enumerate(lex_hits, start=1):
            by_id.setdefault(c['id'], c)
            rrf_scores[c['id']] = rrf_scores.get(c['id'], 0.0) + 1.0 / (rrf_k + rank)
        ranked = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [{**by_id[cid], 'score': round(sc, 6)} for cid, sc in ranked]


def run_multimodal_agent(question: str, retriever: LocalHybridRetriever, pdf_path: Path, pages_dir: Path,
                         api_key: str, model: str = 'openai/gpt-4o-mini',
                         max_steps: int = 6, top_k: int = 6, budget_chars: int = 8000) -> dict[str, Any]:
    """Run multimodal agent with search + inspect_page(page) tools."""
    system_prompt = (
        "You are an exact, evidence-grounded Multimodal RAG Agent for the 100-page GPT-4 Technical Report.\n"
        "Respond ONLY with a valid JSON object matching one of three actions:\n"
        '1) {"action": "search", "query": "<search query>"}\n'
        '2) {"action": "inspect_page", "page": <int 1..100>, "focus": "<specific table/chart/image detail to read from the page image>"}\n'
        '3) {"action": "answer", "answer": "<exact answer with units/conditions and separate [id] citations like [id1][id2]>", "citations": ["id1", ...]}\n'
        "Rules:\n"
        "- Always search first if context is empty. Never repeat a query you already searched.\n"
        "- MANDATORY VISUAL CHECK: Whenever the question asks about a Table, Chart, Figure, Diagram, Image, or specific benchmark/exam numbers, you MUST call `inspect_page(page)` on the candidate page(s) BEFORE answering, so you read exact rows, columns, axes, legends, and values directly from the page image.\n"
        "- For multi-hop questions, gather and inspect all required pages before answering.\n"
        "- Every citation in `citations` MUST appear in `answer` in separate brackets like `[id1][id2]` (never `[id1, id2]`) and match the exact ID in `Valid IDs` that supports the stated fact.\n"
        "- If the first search is incomplete, try one more search with different keywords before concluding insufficient evidence.\n"
        "- If the document does NOT contain enough evidence after searching, return:\n"
        '  {"action": "answer", "answer": "Insufficient evidence in the GPT-4 Technical Report.", "citations": []}\n'
        "- Never guess or fabricate numbers."
    )
    context_items: list[tuple[str, str]] = []
    valid_ids: set[str] = set()
    searched_queries: list[str] = []
    inspected_pages: set[int] = set()
    events: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    llm_cost = 0.0
    doc = pymupdf.open(pdf_path)

    def build_context_str() -> str:
        blocks = []
        total = 0
        for cid, txt in context_items:
            if total + len(txt) + 2 > budget_chars and blocks:
                continue
            blocks.append(txt)
            total += len(txt) + 2
        return '\n\n'.join(blocks)

    for _ in range(max_steps + 1):
        ctx_str = build_context_str()
        user_msg = (
            f"Question: {question}\n\n"
            f"Already searched queries (do NOT repeat): {searched_queries}\n"
            f"Already inspected pages (do NOT repeat): {sorted(inspected_pages)}\n"
            f"Valid IDs in context: {sorted(valid_ids)}\n\n"
            f"Context:\n{ctx_str if ctx_str else '(empty — call search first)'}"
        )
        resp = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://github.com/day07-retrieval-context',
                'X-Title': 'Day07-Advanced-Multimodal',
            },
            json={
                'model': model,
                'temperature': 0.0,
                'response_format': {'type': 'json_object'},
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_msg},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        usage = payload.get('usage') or {}
        prompt_tokens += int(usage.get('prompt_tokens') or 0)
        completion_tokens += int(usage.get('completion_tokens') or 0)
        if usage.get('cost') is not None:
            llm_cost += float(usage['cost'])

        action = json.loads(payload['choices'][0]['message']['content'])
        act = action.get('action')

        if act == 'search':
            if len([e for e in events if e['event'] in ('search', 'inspect_page')]) >= max_steps:
                return {'status': 'abstain', 'answer': 'Insufficient evidence (step limit).', 'citations': [],
                        'events': events + [{'event': 'step_limit'}],
                        'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'llm_cost_usd': llm_cost}
            query = str(action.get('query') or '').strip()
            searched_queries.append(query)
            hits = retriever.search(query, top_k=top_k)
            for h in hits:
                if h['id'] not in valid_ids:
                    valid_ids.add(h['id'])
                    context_items.append((h['id'], f"[{h['id']} | page {h['page']}]\n{h['text']}"))
            events.append({'event': 'search', 'query': query, 'results': [{'id': h['id'], 'page': h['page'], 'score': h['score']} for h in hits]})

        elif act == 'inspect_page':
            if len([e for e in events if e['event'] in ('search', 'inspect_page')]) >= max_steps:
                return {'status': 'abstain', 'answer': 'Insufficient evidence (step limit).', 'citations': [],
                        'events': events + [{'event': 'step_limit'}],
                        'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'llm_cost_usd': llm_cost}
            page_num = int(action.get('page', 1))
            focus = str(action.get('focus') or question)
            if page_num in inspected_pages:
                events.append({'event': 'inspect_page', 'page': page_num, 'skipped': 'already_inspected'})
                continue
            inspected_pages.add(page_num)
            if not (1 <= page_num <= len(doc)):
                events.append({'event': 'inspect_page', 'page': page_num, 'error': 'page out of range'})
                continue
            img_path = pages_dir / f'page_{page_num:03d}.png'
            if not img_path.exists():
                pages_dir.mkdir(parents=True, exist_ok=True)
                doc[page_num - 1].get_pixmap(dpi=180).save(str(img_path))
            vis_prompt = (
                f"Inspect page {page_num} of the GPT-4 Technical Report carefully.\n"
                f"Focus: {focus}\n"
                "Extract the exact numbers, labels, units, table cells, chart values, or image details relevant to the focus. "
                "If the requested information is NOT on this page, state clearly 'NOT_FOUND_ON_PAGE'."
            )
            vis_ans, v_tok, v_cst = openrouter_vision(img_path, vis_prompt, api_key, model=model)
            prompt_tokens += v_tok
            llm_cost += v_cst
            pid = f'page-{page_num:03d}'
            if pid not in valid_ids:
                valid_ids.add(pid)
                context_items.insert(0, (pid, f"[{pid} | Visual Inspection]\n{vis_ans}"))
            events.append({'event': 'inspect_page', 'page': page_num, 'focus': focus, 'inspection': vis_ans})

        elif act == 'answer':
            ans = str(action.get('answer') or '').strip()
            cites = action.get('citations') or []
            if ans and isinstance(cites, list) and cites and all(c in valid_ids for c in cites):
                missing_inline = [f'[{c}]' for c in cites if c not in ans]
                if missing_inline:
                    ans = ans + ' ' + ''.join(missing_inline)
                return {'status': 'answered', 'answer': ans, 'citations': cites, 'events': events + [{'event': 'answer'}],
                        'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'llm_cost_usd': llm_cost}
            return {'status': 'abstain', 'answer': 'Insufficient evidence in the GPT-4 Technical Report.', 'citations': [],
                    'events': events + [{'event': 'abstain'}],
                    'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'llm_cost_usd': llm_cost}
        else:
            raise ValueError(f'Unknown action: {act}')

    return {'status': 'abstain', 'answer': 'Insufficient evidence.', 'citations': [], 'events': events,
            'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'llm_cost_usd': llm_cost}


def run_benchmark(index_dir: Path, pdf_path: Path, benchmark_path: Path, split: str, out_dir: Path,
                  model: str = 'openai/gpt-4o-mini') -> None:
    api_key = os.getenv('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('OPENROUTER_API_KEY is required')
    chunks = json.loads((index_dir / 'index_chunks.json').read_text(encoding='utf-8'))
    vectors = json.loads((index_dir / 'index_embeddings.json').read_text(encoding='utf-8'))
    embedder = OpenRouterEmbedder()
    retriever = LocalHybridRetriever(chunks, vectors, embedder)
    cases = [c for c in json.loads(benchmark_path.read_text(encoding='utf-8')) if c['split'] == split]

    traces = []
    total_cost = 0.0
    for c in cases:
        t0 = time.perf_counter()
        try:
            res = run_multimodal_agent(c['question'], retriever, pdf_path, index_dir / 'pages', api_key, model=model)
            err = None
        except Exception as exc:
            res = {'status': 'error', 'answer': '', 'citations': [], 'events': [], 'prompt_tokens': 0, 'completion_tokens': 0, 'llm_cost_usd': 0.0}
            err = f'{type(exc).__name__}: {exc}'
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        total_cost += res['llm_cost_usd']
        traces.append({
            'query_id': c['id'],
            'split': split,
            'category': c['category'],
            'answerable': c['answerable'],
            'question': c['question'],
            'status': res['status'],
            'answer': res['answer'],
            'citations': res['citations'],
            'events': res['events'],
            'elapsed_ms': elapsed_ms,
            'prompt_tokens': res['prompt_tokens'],
            'completion_tokens': res['completion_tokens'],
            'llm_cost_usd': round(res['llm_cost_usd'], 6),
            'error': err,
        })
    summary = {
        'split': split,
        'model': model,
        'n': len(traces),
        'answered': sum(1 for t in traces if t['status'] == 'answered'),
        'abstain': sum(1 for t in traces if t['status'] == 'abstain'),
        'errors': sum(1 for t in traces if t['status'] == 'error'),
        'median_ms': round(statistics.median(t['elapsed_ms'] for t in traces), 2) if traces else 0.0,
        'total_cost_usd': round(total_cost + embedder.total_cost_usd, 6),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description='Advanced Multimodal RAG on 100-page PDF')
    parser.add_argument('--mode', required=True, choices=('index', 'run'))
    parser.add_argument('--pdf', type=Path, default=Path('data/advanced/gpt4_technical_report.pdf'))
    parser.add_argument('--index-dir', type=Path, default=Path('data/advanced'))
    parser.add_argument('--benchmark', type=Path, default=Path('data/advanced/benchmark.json'))
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--model', default='openai/gpt-4o-mini')
    parser.add_argument('--output', type=Path, default=Path('results/advanced-dev'))
    args = parser.parse_args()
    if args.mode == 'index':
        build_index(args.pdf, args.index_dir, model=args.model)
    else:
        run_benchmark(args.index_dir, args.pdf, args.benchmark, args.split, args.output, model=args.model)


if __name__ == '__main__':
    main()

