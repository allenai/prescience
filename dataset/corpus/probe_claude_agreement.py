"""Cross-model topic-label agreement probe (Claude Opus 4.7 vs GPT-5.4).

Re-runs the v5 topic-labeling prompt on a 500-paper sample using Claude (Anthropic SDK)
so we can compare its labels against the GPT-5.4 v5 labels we already have. Pulls the
sample's corpus_ids directly from a prior validation JSONL for deterministic alignment.

Resumable: appends to --output_path and skips corpus_ids already labeled.
"""

import argparse
import json
import os
import time
import concurrent.futures as cf

from tqdm import tqdm
import anthropic

import utils
from dataset.corpus.assign_topics import load_topics, build_system_prompt, parse_topics, safe_text


def with_retries(fn, max_retries=6, base_delay=2.0):
    """Exponential backoff with jitter; longer than the OpenAI version because Anthropic 429s benefit from cooling down."""
    import random
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                utils.log(f"Request failed after {max_retries} retries: {exc}")
                return None
            delay = base_delay * (2 ** (attempt - 1)) + random.random() * 0.5
            time.sleep(delay)


def query_claude(client, model, system_prompt, title, abstract, topic_set):
    user_content = f"Title: {safe_text(title, 300)}\nAbstract: {safe_text(abstract, 2000)}\n\nList all applicable topics in descending order of relevance."
    response = client.messages.create(model=model, max_tokens=300, system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}], messages=[{"role": "user", "content": user_content}])
    content = response.content[0].text.strip()
    return parse_topics(content, topic_set)


def load_sample_corpus_ids(sample_source_path):
    ids = []
    with open(sample_source_path) as f:
        for line in f:
            ids.append(json.loads(line)["corpus_id"])
    return ids


def load_existing(output_path):
    if not os.path.exists(output_path):
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["corpus_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    parser = argparse.ArgumentParser(description="Run Claude topic labeling for cross-model agreement validation")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience")
    parser.add_argument("--topics_path", type=str, default="dataset/corpus/topics_list_v5.txt")
    parser.add_argument("--sample_source_path", type=str, required=True, help="JSONL file whose corpus_ids define the sample to probe (alignment with prior validation runs)")
    parser.add_argument("--output_path", type=str, required=True, help="JSONL output (resumable, appends and skips already-labeled ids)")
    parser.add_argument("--model", type=str, default="claude-opus-4-7", help="Anthropic model identifier")
    parser.add_argument("--max_workers", type=int, default=8, help="Parallel API workers (Anthropic Opus rate limits are tight)")
    args = parser.parse_args()

    topics = load_topics(args.topics_path)
    topic_set = set(topics)
    utils.log(f"Loaded {len(topics)} topics from {args.topics_path}")

    sample_ids = load_sample_corpus_ids(args.sample_source_path)
    utils.log(f"Loaded {len(sample_ids)} corpus_ids from sample source {args.sample_source_path}")

    sample_id_set = set(sample_ids)
    by_id = {}
    for split in ["train", "test"]:
        utils.log(f"Loading {split} split for title/abstract lookup")
        papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=split, embedding_type=None, load_sd2publications=False)
        for p in papers:
            if p["corpus_id"] in sample_id_set and p["corpus_id"] not in by_id:
                by_id[p["corpus_id"]] = p
    utils.log(f"Resolved {len(by_id)} / {len(sample_ids)} sample papers from corpus")

    already = load_existing(args.output_path)
    to_label = [cid for cid in sample_ids if cid not in already]
    utils.log(f"Already labeled: {len(already)}; to label: {len(to_label)}")
    if not to_label:
        utils.log("Nothing to do.")
        return

    client = anthropic.Anthropic()
    system_prompt = build_system_prompt(topics)
    utils.log(f"System prompt length: {len(system_prompt)} chars")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out_f = open(args.output_path, "a")
    written = 0

    def work(cid):
        p = by_id.get(cid)
        if p is None:
            return {"corpus_id": cid, "topics": []}
        labels = with_retries(lambda: query_claude(client, args.model, system_prompt, p.get("title", ""), p.get("abstract", ""), topic_set))
        if labels is None:
            return None
        return {"corpus_id": cid, "topics": labels}

    try:
        with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(work, cid) for cid in to_label]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"Probing ({args.model})"):
                rec = future.result()
                if rec is None:
                    continue
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                written += 1
    finally:
        out_f.close()
    utils.log(f"Wrote {written} probe records to {args.output_path}")


if __name__ == "__main__":
    main()
