"""GPT-5 + tool-using agent baseline for contribution generation.

For each target paper, runs an OpenAI Chat Completions agent loop with four
leakage-safe corpus tools (search_papers, get_paper, get_author_recent_papers,
get_references_of). The agent emits a final Reasoning / Title / Abstract block;
the harness parses and saves it in the same format as `baseline_gpt_parallel.py`.

Run:
    python -m task_followup_prediction.generate.baseline_agent_gpt \
        --split test \
        --model gpt-5-2025-08-07 \
        --max_query_papers 20 \
        --num_workers 4 \
        --output_dir data/task_followup_prediction/test/generations
"""

import os
import json
import argparse
import concurrent.futures as cf
from typing import Dict, List, Optional

import numpy as np
import openai
from tqdm import tqdm

import utils
from task_followup_prediction.dataset import get_query_papers
from task_followup_prediction.generate import agent_tools
from task_followup_prediction.generate.agent_tools import (
    TOOL_SCHEMAS, ToolContext, CorpusView, GritLMRuntime, dispatch_tool_call,
)


MAX_ROUNDS_DEFAULT = 10


# ---------------------------------------------------------------------------
# Per-target helpers
# ---------------------------------------------------------------------------

def _build_user_message(rec: Dict, papers_dict: Dict) -> str:
    """User message: authors + inlined references with corpus_ids + cutoff_date."""
    authors = rec.get("authors") or []
    author_lines = []
    for a in authors:
        if not isinstance(a, dict):
            continue
        author_lines.append(f"- author_id={a.get('author_id')}, name={a.get('name')}")

    ref_blocks = []
    for r in rec.get("key_references") or []:
        rid = r.get("corpus_id") if isinstance(r, dict) else None
        if rid is None:
            continue
        ref_paper = papers_dict.get(rid)
        if ref_paper is None:
            continue
        ref_blocks.append(
            f"--- Reference (corpus_id={rid}) ---\n"
            f"Title: {ref_paper.get('title','').strip()}\n"
            f"Abstract: {ref_paper.get('abstract','').strip()}"
        )

    return (
        f"Target paper authors:\n" + ("\n".join(author_lines) or "(none provided)") + "\n\n"
        f"Target paper cutoff_date: {rec['date']}\n\n"
        f"Influential references of the target paper (use these as the primary grounding):\n\n"
        + "\n\n".join(ref_blocks)
        + "\n\nPredict the target paper's title and abstract."
    )


def _parse_final_answer(text: str) -> Dict[str, str]:
    """Parse 'Reasoning: ... Title: ... Abstract: ...' from the final assistant message."""
    out = {"reasoning": "", "title": "", "abstract": ""}
    if "Reasoning:" in text and "Title:" in text and "Abstract:" in text:
        try:
            out["reasoning"] = text.split("Reasoning:", 1)[1].split("Title:", 1)[0].strip()
            out["title"] = text.split("Title:", 1)[1].split("Abstract:", 1)[0].strip()
            out["abstract"] = text.split("Abstract:", 1)[1].strip()
        except Exception:
            pass
    elif "Title:" in text and "Abstract:" in text:
        try:
            out["title"] = text.split("Title:", 1)[1].split("Abstract:", 1)[0].strip()
            out["abstract"] = text.split("Abstract:", 1)[1].strip()
        except Exception:
            pass
    return out


def run_agent(
    rec: Dict,
    papers_dict: Dict,
    corpus: CorpusView,
    gritlm: GritLMRuntime,
    client: openai.OpenAI,
    model: str,
    system_prompt: str,
    max_rounds: int,
) -> Dict:
    """Run the agent loop for one target paper. Mutates `rec` with title/abstract/reasoning/trace."""
    cutoff_date = rec["date"]
    target_corpus_id = rec["corpus_id"]
    target_paper = papers_dict.get(target_corpus_id, {})
    ctx = ToolContext(
        corpus=corpus,
        gritlm_runtime=gritlm,
        cutoff_date=cutoff_date,
        target_corpus_id=target_corpus_id,
        target_title=target_paper.get("title", ""),
        target_abstract=target_paper.get("abstract", ""),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_message(rec, papers_dict)},
    ]
    trace = []  # list of {"round", "tool_calls"} for debugging

    final_text = ""
    for round_idx in range(max_rounds + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=TOOL_SCHEMAS,
            )
        except Exception as e:
            utils.log(f"[{target_corpus_id}] OpenAI call failed at round {round_idx}: {e}")
            break

        msg = response.choices[0].message
        # Any tool calls?
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            final_text = (msg.content or "").strip()
            break

        # Append assistant tool-call message, then execute each tool, then append tool results.
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })
        round_trace = {"round": round_idx, "tool_calls": []}
        for tc in tool_calls:
            tool_name = tc.function.name
            args_json = tc.function.arguments or "{}"
            result_json = dispatch_tool_call(tool_name, args_json, ctx)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_json})
            # Trim the trace entry so the saved JSON is reasonable in size
            round_trace["tool_calls"].append({
                "name": tool_name,
                "arguments_preview": args_json[:200],
                "result_preview": result_json[:300],
            })
        trace.append(round_trace)

        if round_idx == max_rounds:
            # Used up all rounds without a final answer; ask for one without tools.
            messages.append({"role": "user", "content": "Round budget exhausted. Emit your final Reasoning/Title/Abstract now."})
            try:
                response = client.chat.completions.create(model=model, messages=messages)
                final_text = (response.choices[0].message.content or "").strip()
            except Exception as e:
                utils.log(f"[{target_corpus_id}] OpenAI final-call failed: {e}")
            break

    parsed = _parse_final_answer(final_text)
    rec["title"] = parsed["title"]
    rec["abstract"] = parsed["abstract"]
    rec["reasoning"] = parsed["reasoning"]
    rec["agent_rounds"] = len(trace)
    rec["agent_trace"] = trace
    rec["agent_final_text_len"] = len(final_text)
    return rec


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _build_corpus_view(all_papers, all_embeddings, sd2publications, distance_metric: str = "cosine") -> CorpusView:
    """Build a single FAISS index over all corpus papers' GRIT embeddings."""
    utils.log("Building FAISS index over corpus GRIT embeddings...")
    papers_dict = {p["corpus_id"]: p for p in all_papers}
    embeddings_dict = {}
    for cid in papers_dict:
        if cid in all_embeddings:
            emb = all_embeddings[cid]
            vec = emb["key"] if isinstance(emb, dict) else emb[0]
            embeddings_dict[cid] = vec.reshape(-1)
    # Sorted faiss_id_map -> corpus_id mapping
    faiss_id_map = list(embeddings_dict.keys())
    matrix = np.stack([embeddings_dict[c] for c in faiss_id_map], axis=0).astype(np.float32)
    if distance_metric == "cosine":
        # Normalize for inner-product = cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        matrix = matrix / norms
    import faiss  # local import; relies on env having faiss-cpu OR faiss-gpu
    index = faiss.IndexFlatIP(matrix.shape[1]) if distance_metric == "cosine" else faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    utils.log(f"FAISS index: {index.ntotal} vectors, dim={matrix.shape[1]}")
    return CorpusView(papers_dict=papers_dict, sd2publications=sd2publications, faiss_index=index, faiss_id_map=faiss_id_map)


def main():
    parser = argparse.ArgumentParser(description="GPT-5 + tools agent baseline for contribution generation")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--embeddings_dir", type=str, default="data/corpus/test")
    parser.add_argument("--embedding_type", type=str, default="grit", choices=["grit"])  # only GRIT supported here
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/generations")
    parser.add_argument("--model", type=str, default="gpt-5-2025-08-07")
    parser.add_argument("--max_query_papers", type=int, default=20, help="Limit eval to first N target papers (for smoke testing).")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel agent threads. GRIT inference is locked, so high values mostly help OpenAI throughput.")
    parser.add_argument("--max_rounds", type=int, default=MAX_ROUNDS_DEFAULT)
    parser.add_argument("--gritlm_model", type=str, default="GritLM/GritLM-7B")
    args = parser.parse_args()

    utils.log("Loading corpus + GRIT embeddings...")
    all_papers, sd2publications, all_embeddings = utils.load_corpus(
        hf_repo_id=args.hf_repo_id, split=args.split,
        embeddings_dir=args.embeddings_dir, embedding_type=args.embedding_type,
        load_sd2publications=True,
    )
    if all_embeddings is None:
        raise RuntimeError(f"No GRIT embeddings found in {args.embeddings_dir}")

    corpus = _build_corpus_view(all_papers, all_embeddings, sd2publications)
    papers_dict = corpus.papers_dict

    utils.log("Loading GritLM (one-time, GPU)...")
    gritlm = GritLMRuntime(args.gritlm_model)

    query_papers = get_query_papers(all_papers, max_papers=args.max_query_papers)
    utils.log(f"Loaded {len(all_papers)} papers; {len(query_papers)} target instances")

    with open("task_followup_prediction/templates/prediction_system_agent.prompt", "r") as f:
        system_prompt = f.read()

    client = openai.OpenAI()

    def _worker(rec):
        return run_agent(rec, papers_dict, corpus, gritlm, client, args.model, system_prompt, args.max_rounds)

    utils.log(f"Running {len(query_papers)} agents with {args.num_workers} workers, max_rounds={args.max_rounds}")
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        for done in tqdm(ex.map(_worker, query_papers), total=len(query_papers), desc="Agent runs"):
            results.append(done)

    output_path = os.path.join(args.output_dir, f"generations.agent.{args.model}.json")
    utils.save_json(results, output_path, overwrite=True, metadata=utils.update_metadata([], args))
    utils.log(f"Saved {len(results)} generations to {output_path}")


if __name__ == "__main__":
    main()
