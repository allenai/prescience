"""Compute LACER scores for diversity analysis using pre-computed k-NN neighbors."""
import os
import re
import random
import argparse
import time
import concurrent.futures as cf
from collections import defaultdict

import openai
from tqdm import tqdm

import utils

SYSTEM_PROMPT = "You are an expert reviewer that assigns similarity scores to research paper title-abstract pairs."
SCORE_REGEX = re.compile(r"Score\s*:\s*([-+]?[0-9]+(?:\.\d+)?)", re.IGNORECASE)
MAX_RETRIES = 5
BASE_DELAY = 1.0


def parse_score(response_text):
    """Parse numeric score from model response text."""
    match = SCORE_REGEX.search(response_text)
    if not match:
        raise ValueError(f"Unable to parse score from response: {response_text}")
    return float(match.group(1))


def build_prompt(prompt_template, reference, generation):
    """Build scoring prompt from template and paper data."""
    prompt = prompt_template.replace("{{reference_title}}", reference["title"])
    prompt = prompt.replace("{{reference_abstract}}", reference["abstract"])
    prompt = prompt.replace("{{generation_title}}", generation["title"])
    prompt = prompt.replace("{{generation_abstract}}", generation["abstract"])
    return prompt


def score_pair(job):
    """Score a single paper-neighbor pair using OpenAI with retry logic."""
    client = job["client"]
    prompt_template = job["prompt_template"]
    model = job["model"]
    query_paper = job["query_paper"]
    neighbor_paper = job["neighbor_paper"]
    neighbor_info = job["neighbor_info"]

    result = dict(neighbor_info)
    prompt = build_prompt(prompt_template, query_paper, neighbor_paper)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
            content = response.choices[0].message.content.strip()
            score = parse_score(content)
            result["lacer_score"] = score
            result["lacer_response"] = content
            return result
        except Exception as exc:
            if "429" in str(exc) and attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            result["lacer_error"] = str(exc)
            return result

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute LACER scores for diversity analysis from pre-computed neighbors.")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo (natural corpus)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--synthetic_dir", type=str, default="simulated", help="Synthetic corpus directory under data/multiturn/")
    parser.add_argument("--compute_on", type=str, required=True, choices=["natural", "synthetic"], help="Which corpus to compute LACER scores on")
    parser.add_argument("--neighbors_path", type=str, required=True, help="Path to pre-computed diversity neighbors JSON")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for LACER results JSON")
    parser.add_argument("--prompt_path", type=str, default="task_followup_prediction/evaluate/templates/lacer_scoring_prompt_percentile50.110.txt", help="Path to LACER prompt template")
    parser.add_argument("--n", type=int, default=None, help="Subsample to n papers per bucket (default: use all)")
    parser.add_argument("--k", type=int, default=None, help="Truncate to k neighbors per paper (default: use all)")
    parser.add_argument("--model", type=str, default="gpt-5-2025-08-07", help="OpenAI model for scoring")
    parser.add_argument("--max_workers", type=int, default=10, help="Number of parallel API workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    args = parser.parse_args()

    random.seed(args.seed)

    utils.log(f"Loading neighbors from {args.neighbors_path}")
    neighbors_data, neighbors_metadata = utils.load_json(args.neighbors_path)
    utils.log(f"Loaded {len(neighbors_data)} paper records")

    if args.compute_on == "synthetic":
        utils.log(f"Loading synthetic corpus from data/multiturn/{args.synthetic_dir}")
        synthetic_path = os.path.join("data/multiturn", args.synthetic_dir, "all_papers.json")
        all_papers, _ = utils.load_json(synthetic_path)
        all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    else:
        utils.log(f"Loading natural corpus from HuggingFace (repo={args.hf_repo_id}, split={args.split})")
        all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
        all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers")

    utils.log(f"Loading prompt template from {args.prompt_path}")
    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read().strip()

    # Group by bucket for subsampling
    bucket_records = defaultdict(list)
    for record in neighbors_data:
        bucket_records[record["bucket"]].append(record)

    # Subsample n papers per bucket if specified
    sampled_records = []
    for bucket in sorted(bucket_records.keys()):
        records = bucket_records[bucket]
        if args.n is not None and len(records) > args.n:
            records = random.sample(records, args.n)
        sampled_records.extend(records)
    utils.log(f"After subsampling: {len(sampled_records)} paper records")

    # Prepare scoring jobs
    client = openai.OpenAI()
    jobs = []
    for record in sampled_records:
        query_id = record["corpus_id"]
        if query_id not in all_papers_dict:
            utils.log(f"Warning: query paper {query_id} not found in corpus, skipping")
            continue
        query_paper = all_papers_dict[query_id]

        neighbors = record["neighbors"]
        if args.k is not None:
            neighbors = neighbors[:args.k]

        for neighbor_info in neighbors:
            neighbor_id = neighbor_info["corpus_id"]
            if neighbor_id not in all_papers_dict:
                utils.log(f"Warning: neighbor paper {neighbor_id} not found in corpus, skipping")
                continue
            neighbor_paper = all_papers_dict[neighbor_id]

            jobs.append({
                "client": client,
                "prompt_template": prompt_template,
                "model": args.model,
                "query_id": query_id,
                "bucket": record["bucket"],
                "query_paper": {"title": query_paper["title"], "abstract": query_paper["abstract"]},
                "neighbor_paper": {"title": neighbor_paper["title"], "abstract": neighbor_paper["abstract"]},
                "neighbor_info": neighbor_info,
            })

    utils.log(f"Scoring {len(jobs)} paper-neighbor pairs with {args.max_workers} workers using {args.model}")

    # Score all pairs in parallel
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        scored_neighbors = list(tqdm(executor.map(score_pair, jobs), total=len(jobs), desc="Scoring pairs"))

    # Reconstruct output records
    job_idx = 0
    results = []
    for record in sampled_records:
        query_id = record["corpus_id"]
        if query_id not in all_papers_dict:
            continue

        neighbors = record["neighbors"]
        if args.k is not None:
            neighbors = neighbors[:args.k]

        scored_list = []
        for neighbor_info in neighbors:
            neighbor_id = neighbor_info["corpus_id"]
            if neighbor_id not in all_papers_dict:
                continue
            scored_list.append(scored_neighbors[job_idx])
            job_idx += 1

        results.append({
            "corpus_id": query_id,
            "bucket": record["bucket"],
            "neighbors": scored_list,
        })

    error_count = sum(1 for r in results for n in r["neighbors"] if "lacer_error" in n)
    utils.log(f"Finished scoring; errors on {error_count} pairs")

    scores = [n["lacer_score"] for r in results for n in r["neighbors"] if "lacer_score" in n]
    if scores:
        utils.log(f"Mean LACER score: {sum(scores) / len(scores):.2f} (n={len(scores)})")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    utils.save_json(results, args.output_path, metadata=utils.update_metadata(neighbors_metadata, args))
    utils.log(f"Saved {len(results)} paper results to {args.output_path}")


if __name__ == "__main__":
    main()
