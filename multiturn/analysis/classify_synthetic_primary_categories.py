"""Classify synthetic target papers into primary arXiv categories using OpenAI."""
import argparse
import os
import random
import time
import json
import concurrent.futures as cf
from collections import Counter

from tqdm import tqdm
import openai

import utils

OTHER_LABEL = "other"
RNG = random.Random(42)


def safe_text(text, max_chars):
    if not text:
        return ""
    return text.strip()[:max_chars]


def strip_code_fences(content):
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def parse_primary_category(raw, labels):
    cleaned = strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = {}
    if "primary_category" in data:
        label = str(data["primary_category"]).strip()
    elif "topic" in data:
        label = str(data["topic"]).strip()
    else:
        label = ""
    if label not in labels:
        label = OTHER_LABEL
    confidence = float(data["confidence"]) if "confidence" in data else 0.0
    rationale = str(data["rationale"]).strip() if "rationale" in data else ""
    if not 0.0 <= confidence <= 1.0:
        confidence = 0.0
    return {"primary_category": label, "confidence": confidence, "rationale": rationale}


def build_system_prompt(labels):
    label_list = ", ".join(labels)
    return (
        "You are a careful scientific classifier. Given a paper Title and Abstract, pick ONE primary arXiv category "
        f"from this set:\n{label_list}\n\n"
        "Rules:\n"
        "1) Answer ONLY as compact JSON: {\"primary_category\":\"<label>\", \"confidence\":<0..1>, \"rationale\":\"<1-2 sentences>\"}\n"
        f"2) If unsure or off-domain, choose \"{OTHER_LABEL}\".\n"
        "3) Never invent labels outside the set.\n"
    )


def sample_fewshots(natural_papers, shot_labels, shots_per_label):
    """Sample few-shot examples. Returns (shots, used_corpus_ids)."""
    target_labels = [label for label in shot_labels if label != OTHER_LABEL]
    if not target_labels:
        return [], set()
    buckets = {label: [] for label in target_labels}
    used_ids = set()
    filled = set()
    order = list(range(len(natural_papers)))
    RNG.shuffle(order)
    for idx in order:
        paper = natural_papers[idx]
        if "categories" not in paper or not paper["categories"]:
            continue
        label_list = paper["categories"]
        if not label_list:
            continue
        label = label_list[0]
        if label not in buckets:
            continue
        if len(buckets[label]) >= shots_per_label:
            continue
        title = safe_text(paper["title"], 200) if "title" in paper else ""
        abstract = safe_text(paper["abstract"], 1600) if "abstract" in paper else ""
        if not title or not abstract:
            continue
        buckets[label].append((label, title, abstract))
        used_ids.add(paper["corpus_id"])
        if len(buckets[label]) >= shots_per_label:
            filled.add(label)
        if len(filled) == len(target_labels):
            break
    shots = []
    for label in shot_labels:
        if label == OTHER_LABEL or label not in buckets:
            continue
        shots.extend(buckets[label])
    return shots, used_ids


def fewshot_messages(shots):
    messages = []
    for label, title, abstract in shots:
        user_content = (
            f"Title: {title}\n"
            f"Abstract: {abstract}\n\n"
            "Pick one primary arXiv category from the provided set."
        )
        assistant_content = json.dumps(
            {"primary_category": label, "confidence": 0.9, "rationale": "Label chosen for demonstration."},
            ensure_ascii=False,
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


def build_query_messages(title, abstract, system_prompt, shots):
    messages = [{"role": "system", "content": system_prompt}]
    if shots:
        messages.extend(fewshot_messages(shots))
    user_content = (
        f"Title: {safe_text(title, 200)}\n"
        f"Abstract: {safe_text(abstract, 1600)}\n\n"
        "Pick one primary arXiv category from the provided set."
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def query_once(client, model, messages, labels):
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content.strip()
    return parse_primary_category(content, labels)


def with_retries(fn, max_retries=5, base_delay=1.0):
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                print(f"Classification request failed after retries: {exc}")
                return {"primary_category": OTHER_LABEL, "confidence": 0.0, "rationale": f"error: {exc}"}
            delay = base_delay * (2 ** (attempt - 1)) + RNG.random() * 0.5
            time.sleep(delay)


def classify_jobs(jobs, client, model, labels, max_workers, desc="Classifying"):
    results = {}

    def run_job(job):
        corpus_id, messages = job
        response = with_retries(lambda: query_once(client, model, messages, labels))
        return corpus_id, response["primary_category"]

    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = [executor.submit(run_job, job) for job in jobs]
        for future in tqdm(cf.as_completed(future_to_job), total=len(jobs), desc=desc):
            corpus_id, label = future.result()
            results[corpus_id] = label
    return results


def main():
    parser = argparse.ArgumentParser(description="Classify synthetic target papers into primary arXiv categories using OpenAI.")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo (natural corpus)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--synthetic_dir", type=str, default="simulated", help="Synthetic experiment directory under data/multiturn/")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save corpus_id->primary category JSON mapping")
    parser.add_argument("--model", type=str, default="gpt-5-2025-08-07", help="OpenAI chat model name")
    parser.add_argument("--max_workers", type=int, default=512, help="Number of parallel workers for OpenAI requests")
    parser.add_argument("--shots_per_category", type=int, default=5, help="Few-shot examples per natural category")
    parser.add_argument("--max_fewshot_categories", type=int, default=12, help="Maximum number of natural categories to include in few-shot prompts")
    parser.add_argument("--max_papers", type=int, default=0, help="Optional limit on number of synthetic target papers to classify (0 means all)")
    args = parser.parse_args()

    all_papers_nat, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    all_papers_syn = utils.load_json(f"data/multiturn/{args.synthetic_dir}/all_papers.json")[0]

    natural_targets = [paper for paper in all_papers_nat if "roles" in paper and "target" in paper["roles"]]
    synthetic_targets = [paper for paper in all_papers_syn if "roles" in paper and "synthetic" in paper["roles"]]

    if args.max_papers > 0:
        synthetic_targets = synthetic_targets[:args.max_papers]

    label_set = set()
    category_counts = Counter()
    for paper in natural_targets:
        if "categories" in paper and paper["categories"]:
            label = paper["categories"][0]
            label_set.add(label)
            category_counts[label] += 1
    labels = sorted(label_set)
    if OTHER_LABEL not in labels:
        labels.append(OTHER_LABEL)

    shot_labels = []
    if args.max_fewshot_categories > 0:
        shot_labels = [label for label, _ in category_counts.most_common(args.max_fewshot_categories)]
    shots, _ = sample_fewshots(natural_targets, shot_labels, args.shots_per_category)
    print(f"Using {len(shot_labels)} few-shot categories with {len(shots)} total exemplars.")

    system_prompt = build_system_prompt(labels)

    jobs = []
    for paper in tqdm(synthetic_targets, desc="Preparing classification jobs"):
        title = paper["title"] if "title" in paper else ""
        abstract = paper["abstract"] if "abstract" in paper else ""
        messages = build_query_messages(title, abstract, system_prompt, shots)
        jobs.append((paper["corpus_id"], messages))

    if not jobs:
        raise RuntimeError("No synthetic target papers found for classification.")

    client = openai.OpenAI()
    predictions = classify_jobs(jobs, client, args.model, labels, args.max_workers, desc="Classifying synthetic targets")

    if args.output_path:
        output_path = args.output_path
    else:
        output_path = os.path.join("data/multiturn", args.synthetic_dir, f"synthetic_primary_categories_{args.model}.json")

    metadata = utils.update_metadata([], args)
    utils.save_json(predictions, output_path, metadata=metadata, overwrite=True)

    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
