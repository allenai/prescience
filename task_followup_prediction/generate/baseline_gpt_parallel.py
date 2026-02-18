"""GPT baseline for followup work prediction. Generates title/abstract predictions using OpenAI API."""

import os
import argparse
import concurrent.futures as cf

import openai
from tqdm import tqdm

import utils
from task_followup_prediction.dataset import get_query_papers, create_messages


def query_gpt(record, retry_num=0, max_retries=5):
    """Query OpenAI API and parse the response into title, abstract, and reasoning."""
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
        else:
            del record["client"], record["messages"], record["model"]
            record.update({"title": "", "abstract": "", "reasoning": ""})
            return record


def main():
    """Generate followup work predictions using OpenAI API."""
    parser = argparse.ArgumentParser(description="GPT baseline for followup work prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/generations")
    parser.add_argument("--model", type=str, default="gpt-5-2025-08-07")
    parser.add_argument("--max_query_papers", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=128, help="Number of parallel requests")
    args = parser.parse_args()

    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    query_papers = get_query_papers(all_papers, max_papers=args.max_query_papers)

    with open("task_followup_prediction/templates/prediction_system.prompt", "r") as f:
        system_prompt = f.read()

    openai_client = openai.OpenAI()
    utils.log(f"Loaded {len(all_papers)} papers, {len(query_papers)} query papers after filtering")

    utils.log("Preparing jobs")
    jobs = []
    for rec in query_papers:
        rec["client"] = openai_client
        rec["model"] = args.model
        rec["messages"] = create_messages(rec, system_prompt, all_papers_dict)
        jobs.append(rec)

    utils.log(f"Running {len(jobs)} queries with {args.num_workers} workers")
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        results_iter = ex.map(query_gpt, jobs)
        for rec in tqdm(results_iter, total=len(jobs), desc="Generating samples"):
            pass

    output_path = os.path.join(args.output_dir, f"generations.{args.model}.json")
    utils.save_json(query_papers, output_path, overwrite=True, metadata=utils.update_metadata([], args))
    utils.log(f"Saved {len(query_papers)} generations to {output_path}")


if __name__ == "__main__":
    main()
