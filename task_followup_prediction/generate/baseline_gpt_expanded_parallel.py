"""GPT baseline for followup work prediction with expanded context (author history + related papers via FAISS)."""

import os
import argparse
import concurrent.futures as cf

import openai
from tqdm import tqdm

import utils
from task_followup_prediction.dataset import get_query_papers
from task_followup_prediction.generate.expanded_context import build_messages_with_faiss_walk


def query_gpt(record, retry_num=0, max_retries=5):
    client, model, messages = record["client"], record["model"], record["messages"]
    try:
        response = client.chat.completions.create(model=model, messages=messages)
        answer = response.choices[0].message.content.strip()
        reasoning = answer.split("Reasoning: ")[1].split("Title: ")[0].strip()
        title = answer.split("Title: ")[1].split("Abstract: ")[0].strip()
        abstract = answer.split("Abstract: ")[1].strip()
        del record["client"], record["messages"], record["model"]
        record.update({"title": title, "abstract": abstract, "reasoning": reasoning})
        return record
    except Exception as e:
        utils.log(f"Error: {e}")
        if retry_num < max_retries:
            utils.log(f"Retrying {retry_num + 1}/{max_retries}...")
            return query_gpt(record, retry_num + 1, max_retries)
        del record["client"], record["messages"], record["model"]
        record.update({"title": "", "abstract": "", "reasoning": ""})
        return record


def load_system_prompt(include_author_history, include_related_papers, include_impact, override_path):
    if override_path:
        path = override_path
    else:
        if include_author_history and include_related_papers:
            mode = "expanded_full"
        elif include_author_history:
            mode = "expanded_authors"
        elif include_related_papers:
            mode = "expanded_related"
        else:
            mode = "vanilla"
        suffix = "_impact" if include_impact else ""
        path = f"task_followup_prediction/templates/prediction_system_{mode}{suffix}.prompt"
    with open(path, "r") as f:
        return f.read(), path


def select_query_papers(all_papers, query_corpus_ids_path, max_query_papers):
    if query_corpus_ids_path:
        ids, _ = utils.load_json(query_corpus_ids_path)
        ids_set = set(ids)
        id2paper = {p["corpus_id"]: p for p in all_papers}
        missing = ids_set - set(id2paper.keys())
        assert not missing, f"{len(missing)} requested corpus_ids missing from split"
        selected = [id2paper[cid] for cid in ids]
        for p in selected:
            assert "target" in p["roles"] and len(p["key_references"]) > 0, f"{p['corpus_id']} is not a valid target"
        return selected
    return get_query_papers(all_papers, max_papers=max_query_papers)


def detect_leakage(messages, target_record):
    user_content = messages[1]["content"]
    cid = target_record["corpus_id"]
    if cid in user_content:
        return "corpus_id"
    if target_record["title"] in user_content:
        return "title"
    if target_record["abstract"] in user_content:
        return "abstract"
    abstract = target_record["abstract"]
    if len(abstract) >= 260 and abstract[200:260] in user_content:
        return "mid-abstract"
    return None


def main():
    parser = argparse.ArgumentParser(description="GPT baseline for followup prediction with expanded context")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/generations")
    parser.add_argument("--model", type=str, default="gpt-5-2025-08-07")
    parser.add_argument("--max_query_papers", type=int, default=1000)
    parser.add_argument("--query_corpus_ids_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=128)
    parser.add_argument("--include_author_history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_related_papers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_author_papers_per_author", type=int, default=3)
    parser.add_argument("--max_total_author_papers", type=int, default=15)
    parser.add_argument("--num_related_papers", type=int, default=10)
    parser.add_argument("--embedding_type", type=str, default=None, choices=[None, "gtr", "grit", "specter2"])
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--prompt_template_path", type=str, default=None)
    parser.add_argument("--assert_no_leakage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_impact", action="store_true", help="Inject oracle target citation impact line at top of user prompt (intentional temporal leakage)")
    parser.add_argument("--dry_run", action="store_true", help="Build messages and assert safety but skip API calls")
    args = parser.parse_args()

    if args.include_related_papers:
        assert args.embedding_type and args.embeddings_dir, "--embedding_type and --embeddings_dir required when --include_related_papers"
    distance_metric = "l2" if args.embedding_type == "specter2" else "cosine"

    utils.log(f"Loading corpus (split={args.split})")
    all_papers, _, all_embeddings = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=args.embeddings_dir, embedding_type=args.embedding_type if args.include_related_papers else None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    query_papers = select_query_papers(all_papers, args.query_corpus_ids_path, args.max_query_papers)
    utils.log(f"Loaded {len(all_papers)} papers, {len(query_papers)} query papers")

    system_prompt, prompt_path = load_system_prompt(args.include_author_history, args.include_related_papers, args.include_impact, args.prompt_template_path)
    utils.log(f"Using system prompt: {prompt_path}")

    query_papers_dict = {p["corpus_id"]: p for p in query_papers}
    all_papers_sorted = sorted(all_papers, key=lambda p: p["date"])

    utils.log("Building messages (sequential date-walk with FAISS)")
    built = build_messages_with_faiss_walk(query_papers_dict=query_papers_dict, all_papers_sorted=all_papers_sorted, all_papers_dict=all_papers_dict, all_embeddings=all_embeddings if args.include_related_papers else {}, distance_metric=distance_metric, system_prompt=system_prompt, num_author_papers_per_author=args.num_author_papers_per_author, max_total_author_papers=args.max_total_author_papers, num_related_papers=args.num_related_papers, include_author_history=args.include_author_history, include_related_papers=args.include_related_papers, include_impact=args.include_impact)
    utils.log(f"Built messages for {len(built)} query targets")

    openai_client = None if args.dry_run else openai.OpenAI()
    jobs = []
    skipped = []
    for cid, messages, n_author, n_related in built:
        target = query_papers_dict[cid]
        if args.assert_no_leakage:
            leak = detect_leakage(messages, target)
            if leak is not None:
                skipped.append({"corpus_id": cid, "leak": leak})
                continue
        rec = dict(target)
        rec["client"] = openai_client
        rec["model"] = args.model
        rec["messages"] = messages
        rec["num_author_papers_used"] = n_author
        rec["num_related_papers_used"] = n_related
        jobs.append(rec)
    if skipped:
        utils.log(f"Skipped {len(skipped)} targets due to verbatim text leakage: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    if args.dry_run:
        utils.log(f"Dry run complete: built and validated {len(jobs)} messages; skipping API calls")
        utils.log(f"First message user prompt:\n{jobs[0]['messages'][1]['content'][:2000]}")
        return

    utils.log(f"Running {len(jobs)} queries with {args.num_workers} workers")
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        results = list(tqdm(ex.map(query_gpt, jobs), total=len(jobs), desc="Generating samples"))

    if args.include_author_history and args.include_related_papers:
        filename_prefix = "generations.expanded_full"
    elif args.include_author_history:
        filename_prefix = "generations.expanded_authors"
    elif args.include_related_papers:
        filename_prefix = "generations.expanded_related"
    else:
        filename_prefix = "generations.vanilla"
    if args.include_impact:
        filename_prefix += "_impact"
    output_path = os.path.join(args.output_dir, f"{filename_prefix}.{args.model}.json")
    utils.save_json(results, output_path, overwrite=True, metadata=utils.update_metadata([], args))
    utils.log(f"Saved {len(results)} generations to {output_path}")


if __name__ == "__main__":
    main()
