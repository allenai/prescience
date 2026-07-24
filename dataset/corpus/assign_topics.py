"""Multi-label topic assignment for PreScience target papers using OpenAI.

Assigns each target paper zero or more topics from `dataset/corpus/topics_list.txt` by
calling an OpenAI chat model once per paper. Output is JSONL with one record per paper:
{"corpus_id": "...", "topics": [...]}. The output file is appended to, so this script is
resumable after interruption — already-labeled corpus_ids are skipped on restart.

Stratified sampling across ISO weeks is supported for validation-phase runs.
"""

import argparse
import os
import json
import random
import time
import concurrent.futures as cf
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm
import openai

import utils


RNG = random.Random(42)


def load_topics(path):
    with open(path) as f:
        topics = [line.strip() for line in f if line.strip()]
    return topics


def build_system_prompt(topics):
    """System prompt containing the full topic list. Kept identical across calls so OpenAI's automatic prompt caching applies."""
    bullets = "\n".join(f"- {t}" for t in topics)
    return ("You are a scientific paper topic classifier. Given a paper title and abstract, assign all topics from the list below that clearly apply.\n\nTopics:\n" + bullets + "\n\nRules:\n1) Output ONLY compact JSON of the form {\"topics\": [\"<topic>\", \"<topic>\", ...]}.\n2) Each string in the list MUST exactly match one topic from the list above (same spelling and punctuation).\n3) If no topic clearly applies, output {\"topics\": []}.\n4) Never invent topics outside the list.\n5) List all applicable topics in descending order of relevance. Papers commonly span multiple topics — a method, a task, a domain, and a data concern can each map to different topics. Include every topic the paper directly addresses, not just the primary focus. Exclude topics that are only mentioned in passing or tangentially adjacent.\n")


def strip_code_fences(s):
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def parse_topics(raw, topic_set):
    """Parse model JSON; return a deduplicated list of valid topics (dropping unknowns/noise)."""
    cleaned = strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    raw_topics = data.get("topics", []) if isinstance(data, dict) else []
    if not isinstance(raw_topics, list):
        return []
    seen, out = set(), []
    for t in raw_topics:
        if isinstance(t, str) and t in topic_set and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def safe_text(text, max_chars):
    if not text:
        return ""
    return text.strip()[:max_chars]


def query_once(client, model, system_prompt, title, abstract, topic_set):
    user_content = f"Title: {safe_text(title, 300)}\nAbstract: {safe_text(abstract, 2000)}\n\nList all applicable topics in descending order of relevance."
    response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}])
    content = response.choices[0].message.content.strip()
    return parse_topics(content, topic_set)


def with_retries(fn, max_retries=5, base_delay=1.0):
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                utils.log(f"Request failed after {max_retries} retries: {exc}")
                return None
            delay = base_delay * (2 ** (attempt - 1)) + RNG.random() * 0.5
            time.sleep(delay)


def stratified_sample(papers, samples_per_week):
    """Stratified sample of papers by ISO week of publication date."""
    weekly = defaultdict(list)
    for paper in papers:
        dt = datetime.strptime(paper["date"], "%Y-%m-%d")
        iso_year, iso_week, _ = dt.isocalendar()
        weekly[(iso_year, iso_week)].append(paper)
    sampled = []
    for key in sorted(weekly):
        week_papers = weekly[key]
        n = min(samples_per_week, len(week_papers))
        sampled.extend(RNG.sample(week_papers, n))
    utils.log(f"Stratified sample: {len(sampled)} papers across {len(weekly)} weeks ({samples_per_week}/week).")
    return sampled


def load_existing_corpus_ids(output_path):
    if not os.path.exists(output_path):
        return set()
    labeled = set()
    with open(output_path) as f:
        for line in f:
            try:
                labeled.add(json.loads(line)["corpus_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return labeled


def label_papers(papers, client, model, system_prompt, topic_set, output_path, max_workers):
    already = load_existing_corpus_ids(output_path)
    to_label = [p for p in papers if p["corpus_id"] not in already]
    if not to_label:
        utils.log(f"All {len(papers)} papers already labeled in {output_path}; nothing to do.")
        return
    utils.log(f"Labeling {len(to_label)} papers (skipped {len(already)} already present)")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_f = open(output_path, "a")
    written = 0

    def work(paper):
        topics = with_retries(lambda: query_once(client, model, system_prompt, paper.get("title", ""), paper.get("abstract", ""), topic_set))
        if topics is None:
            return None
        return {"corpus_id": paper["corpus_id"], "topics": topics}

    try:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(work, p) for p in to_label]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"Labeling ({model})"):
                record = future.result()
                if record is None:
                    continue
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                written += 1
    finally:
        out_f.close()
    utils.log(f"Wrote {written} new records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-label topic assignment for PreScience target papers via OpenAI.")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"], choices=["train", "test"], help="Splits to include")
    parser.add_argument("--topics_path", type=str, default="dataset/corpus/topics_list.txt", help="File containing one topic per line")
    parser.add_argument("--model", type=str, required=True, help="OpenAI chat model identifier")
    parser.add_argument("--output_path", type=str, required=True, help="JSONL output path (resumable: appends and skips already-labeled ids)")
    parser.add_argument("--samples_per_week", type=int, default=0, help="Stratified sample per ISO week (0 = label all target papers)")
    parser.add_argument("--max_papers", type=int, default=0, help="Hard cap on papers labeled (0 = no cap)")
    parser.add_argument("--max_workers", type=int, default=128, help="Parallel API workers")
    args = parser.parse_args()

    topics = load_topics(args.topics_path)
    topic_set = set(topics)
    utils.log(f"Loaded {len(topics)} topics from {args.topics_path}")

    all_targets, seen_ids = [], set()
    for split in args.splits:
        utils.log(f"Loading {split} split")
        papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=split, embedding_type=None, load_sd2publications=False)
        for p in papers:
            if p["corpus_id"] in seen_ids:
                continue
            if "roles" not in p or "target" not in p["roles"]:
                continue
            seen_ids.add(p["corpus_id"])
            all_targets.append(p)
    utils.log(f"Total unique target papers across splits {args.splits}: {len(all_targets)}")

    if args.samples_per_week > 0:
        all_targets = stratified_sample(all_targets, args.samples_per_week)
    if args.max_papers > 0:
        all_targets = all_targets[:args.max_papers]

    client = openai.OpenAI()
    system_prompt = build_system_prompt(topics)
    utils.log(f"System prompt length: {len(system_prompt)} chars")
    label_papers(all_targets, client, args.model, system_prompt, topic_set, args.output_path, args.max_workers)


if __name__ == "__main__":
    main()
