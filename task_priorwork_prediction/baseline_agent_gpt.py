"""GPT-5 + tool-using agent baseline for prior work prediction.

For each evaluation instance, runs an OpenAI Chat Completions agent loop with five
leakage-safe corpus tools (search_papers, get_paper, get_author_recent_papers,
get_references_of, get_papers_citing). The agent emits a final
`Reasoning: ... \\nPredictions: [corpus_id, ...]` block; the harness parses and
saves the corpus_id list, padded to k=1000 with random pre-cutoff candidates.

Run:
    python3 -m task_priorwork_prediction.baseline_agent_gpt \\
        --split test \\
        --embeddings_dir data/corpus/test \\
        --model gpt-5-mini \\
        --max_instances 100 \\
        --num_workers 16 \\
        --output_dir data/task_priorwork_prediction/test/predictions
"""

import os
import re
import ast
import json
import random
import argparse
import concurrent.futures as cf
from typing import Dict, List

import numpy as np
import openai
from tqdm import tqdm

import utils
from task_priorwork_prediction.dataset import create_evaluation_instances
from task_priorwork_prediction import agent_tools
from task_priorwork_prediction.agent_tools import (
    TOOL_SCHEMAS, ToolContext, CorpusView, GritLMRuntime, dispatch_tool_call, build_cited_by_index,
)

random.seed(42)

MAX_ROUNDS_DEFAULT = 10
PREDICTIONS_RE = re.compile(r"Predictions:\s*(\[[^\]]*\])", re.DOTALL)


def _build_user_message(instance: Dict, papers_dict: Dict) -> str:
    authors = []
    for aid in instance.get("author_ids") or []:
        author_name = None
        for cid in papers_dict.get(aid, []) if isinstance(papers_dict.get(aid), list) else []:
            pass
        authors.append(f"- author_id={aid}")
    target_corpus_id = instance.get("corpus_id")
    paper = papers_dict.get(target_corpus_id, {}) if target_corpus_id else {}
    real_authors = paper.get("authors") or []
    name_lookup = {a.get("author_id"): a.get("name") for a in real_authors if isinstance(a, dict)}
    author_lines = [f"- author_id={aid}, name={name_lookup.get(aid, '?')}" for aid in instance.get("author_ids") or []]
    return (
        "Target paper authors:\n"
        + ("\n".join(author_lines) or "(none)")
        + f"\n\nTarget paper cutoff_date: {instance['date']}\n\n"
        "Predict the corpus_ids of papers the target authors will cite as KEY (highly influential) references "
        "in the upcoming target paper. Use the tools to explore the authors' recent work, the citation graph, "
        "and topical neighborhoods. Aim for a ranked list of ~100-200 corpus_ids."
    )


def _parse_predictions(text: str, valid_ids: set) -> List:
    """Parse `Predictions: [id1, id2, ...]` from the final assistant message and validate IDs."""
    match = PREDICTIONS_RE.search(text)
    if match is None:
        return [], 0
    raw = match.group(1)
    parsed = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            if isinstance(parsed, list):
                break
        except (ValueError, SyntaxError, json.JSONDecodeError):
            parsed = None
    if not isinstance(parsed, list):
        return [], 0
    parsed = [str(x) for x in parsed]
    kept = [cid for cid in parsed if cid in valid_ids]
    hallucinated = len(parsed) - len(kept)
    return kept, hallucinated


def _pad_predictions(predicted_ids, k, papers_dict, cutoff_date, exclude):
    if len(predicted_ids) >= k:
        return predicted_ids[:k]
    excluded = set(predicted_ids) | set(exclude)
    available = [cid for cid in papers_dict if cid not in excluded and papers_dict[cid].get("date", "") < cutoff_date]
    needed = k - len(predicted_ids)
    extras = random.sample(available, min(needed, len(available)))
    return predicted_ids + extras


def run_agent(instance, papers_dict, corpus, gritlm, client, model, system_prompt, max_rounds):
    cutoff_date = instance["date"]
    target_corpus_id = instance["corpus_id"]
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
        {"role": "user", "content": _build_user_message(instance, papers_dict)},
    ]
    trace = []
    final_text = ""
    for round_idx in range(max_rounds + 1):
        try:
            response = client.chat.completions.create(model=model, messages=messages, tools=TOOL_SCHEMAS)
        except Exception as e:
            utils.log(f"[{target_corpus_id}] OpenAI call failed at round {round_idx}: {e}")
            break

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            final_text = (msg.content or "").strip()
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })
        round_trace = {"round": round_idx, "tool_calls": []}
        for tc in tool_calls:
            tool_name = tc.function.name
            args_json = tc.function.arguments or "{}"
            result_json = dispatch_tool_call(tool_name, args_json, ctx)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_json})
            round_trace["tool_calls"].append({"name": tool_name, "arguments_preview": args_json[:200], "result_preview": result_json[:300]})
        trace.append(round_trace)

        if round_idx == max_rounds:
            messages.append({"role": "user", "content": "Round budget exhausted. Emit your final 'Reasoning:' + 'Predictions:' block now."})
            try:
                response = client.chat.completions.create(model=model, messages=messages)
                final_text = (response.choices[0].message.content or "").strip()
            except Exception as e:
                utils.log(f"[{target_corpus_id}] OpenAI final-call failed: {e}")
            break

    predicted_ids, hallucinated = _parse_predictions(final_text, ctx.candidate_ids)
    return {
        "predicted_ids": predicted_ids,
        "hallucinated": hallucinated,
        "agent_rounds": len(trace),
        "agent_trace": trace,
        "final_text_len": len(final_text),
        "final_text_tail": final_text[-1500:],
    }


def _build_corpus_view(all_papers, all_embeddings, sd2publications, distance_metric: str = "cosine") -> CorpusView:
    utils.log("Building FAISS index over corpus GRIT embeddings...")
    papers_dict = {p["corpus_id"]: p for p in all_papers}
    embeddings_dict = {}
    for cid in papers_dict:
        if cid in all_embeddings:
            emb = all_embeddings[cid]
            vec = emb["key"] if isinstance(emb, dict) else emb[0]
            embeddings_dict[cid] = vec.reshape(-1)
    faiss_id_map = list(embeddings_dict.keys())
    matrix = np.stack([embeddings_dict[c] for c in faiss_id_map], axis=0).astype(np.float32)
    if distance_metric == "cosine":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        matrix = matrix / norms
    import faiss
    index = faiss.IndexFlatIP(matrix.shape[1]) if distance_metric == "cosine" else faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    utils.log(f"FAISS index: {index.ntotal} vectors, dim={matrix.shape[1]}")
    utils.log("Building citation inverted index (cited_by)...")
    cited_by = build_cited_by_index(all_papers)
    utils.log(f"cited_by index: {len(cited_by)} keys")
    return CorpusView(papers_dict=papers_dict, sd2publications=sd2publications, faiss_index=index, faiss_id_map=faiss_id_map, cited_by=cited_by)


def main():
    parser = argparse.ArgumentParser(description="GPT-5 + tools agent baseline for prior work prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--embeddings_dir", type=str, default="data/corpus/test")
    parser.add_argument("--embedding_type", type=str, default="grit", choices=["grit"])
    parser.add_argument("--output_dir", type=str, default="data/task_priorwork_prediction/test/predictions")
    parser.add_argument("--model", type=str, default="gpt-5-2025-08-07")
    parser.add_argument("--k", type=int, default=1000, help="Pad/truncate predictions to this length")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_rounds", type=int, default=MAX_ROUNDS_DEFAULT)
    parser.add_argument("--gritlm_model", type=str, default="GritLM/GritLM-7B")
    parser.add_argument("--prompt_template", type=str, default="task_priorwork_prediction/templates/agent_priorwork.prompt")
    parser.add_argument("--output_suffix", type=str, default="")
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

    utils.log("Creating evaluation instances")
    evaluation_instances = create_evaluation_instances(all_papers, sd2publications, papers_dict)
    utils.log(f"Created {len(evaluation_instances)} evaluation instances")

    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = sorted(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances")
    else:
        selected_indices = list(range(len(evaluation_instances)))

    with open(args.prompt_template, "r") as f:
        system_prompt = f.read()

    client = openai.OpenAI()

    output_path = os.path.join(args.output_dir, f"predictions.agent.{args.model}{args.output_suffix}.json")
    utils.log(f"Output path: {output_path}")

    selected = [(idx, *evaluation_instances[idx]) for idx in selected_indices]

    def _worker(item):
        idx, date, instance = item
        instance = {**instance, "date": date}
        out = run_agent(instance, papers_dict, corpus, gritlm, client, args.model, system_prompt, args.max_rounds)
        predicted_ids = _pad_predictions(out["predicted_ids"], args.k, papers_dict, instance["date"], [instance["corpus_id"]])
        predicted_scores = [1.0 / (i + 1) for i in range(len(predicted_ids))]
        return {
            "corpus_id": instance["corpus_id"],
            "gt_reference_ids": instance["gt_reference_ids"],
            "predicted_reference_ids": predicted_ids,
            "predicted_reference_scores": predicted_scores,
            "num_model_predictions": len(out["predicted_ids"]),
            "num_hallucinated_ids": out["hallucinated"],
            "agent_rounds": out["agent_rounds"],
            "final_text_len": out["final_text_len"],
            "final_text_tail": out["final_text_tail"],
        }

    utils.log(f"Running {len(selected)} agent queries with {args.num_workers} workers, max_rounds={args.max_rounds}, model={args.model}")
    predictions = []
    save_lock = __import__("threading").Lock()
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(_worker, item): item for item in selected}
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Agent priorwork"):
            rec = fut.result()
            predictions.append(rec)
            with save_lock:
                try:
                    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)
                except Exception as e:
                    utils.log(f"Save failed for {rec['corpus_id']}: {type(e).__name__}: {e}")
            utils.log(f"Done {rec['corpus_id']}: rounds={rec['agent_rounds']} model_preds={rec['num_model_predictions']} halluc={rec['num_hallucinated_ids']}")

    total_hallucinated = sum(p["num_hallucinated_ids"] for p in predictions)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}; total hallucinated dropped: {total_hallucinated}")


if __name__ == "__main__":
    main()
