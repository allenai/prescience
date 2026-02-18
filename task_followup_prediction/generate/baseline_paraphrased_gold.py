"""Ground truth paraphrase baseline for followup work prediction. Predicts by paraphrasing the actual paper."""

import os
import random
import argparse
import concurrent.futures as cf

import openai
from tqdm import tqdm

import utils

random.seed(42)

PARAPHRASE_SYSTEM_PROMPT = """You are a helpful assistant that paraphrases scientific paper titles and abstracts.
Given a title and abstract, produce a paraphrased version that conveys the same information but with different wording.
Maintain the scientific accuracy and key concepts while varying the sentence structure and vocabulary.

Respond in the following format:
Title: <paraphrased title>
Abstract: <paraphrased abstract>"""


def query_paraphrase(record, retry_num=0, max_retries=5):
    """Query OpenAI API to paraphrase the ground truth title and abstract."""
    client, model = record["client"], record["model"]
    original_title = record["ground_truth_title"]
    original_abstract = record["ground_truth_abstract"]

    messages = [
        {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Title: {original_title}\n\nAbstract: {original_abstract}"}
    ]

    try:
        response = client.chat.completions.create(model=model, messages=messages)
        answer = response.choices[0].message.content.strip()

        title = answer.split("Title: ")[1].split("Abstract: ")[0].strip()
        abstract = answer.split("Abstract: ")[1].strip()
        del record["client"], record["model"], record["ground_truth_title"], record["ground_truth_abstract"]
        record.update({
            "title": title,
            "abstract": abstract,
            "reasoning": "Paraphrased from ground truth"
        })
        return record

    except Exception as e:
        utils.log(f"Error: {e}")
        if retry_num < max_retries:
            utils.log(f"Retrying {retry_num + 1}/{max_retries}...")
            return query_paraphrase(record, retry_num + 1, max_retries)
        else:
            del record["client"], record["model"], record["ground_truth_title"], record["ground_truth_abstract"]
            record.update({"title": "", "abstract": "", "reasoning": ""})
            return record


def main():
    """Generate followup work predictions by paraphrasing ground truth."""
    parser = argparse.ArgumentParser(description="Ground truth paraphrase baseline for followup work prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/generations")
    parser.add_argument("--model", type=str, default="gpt-5-2025-08-07")
    parser.add_argument("--max_query_papers", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=128, help="Number of parallel requests")
    args = parser.parse_args()

    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    query_papers = utils.filter_by_roles(all_papers, ["target"])

    if len(query_papers) > args.max_query_papers:
        query_papers = random.sample(query_papers, args.max_query_papers)
    utils.log(f"Loaded {len(all_papers)} papers, {len(query_papers)} query papers after filtering")

    openai_client = openai.OpenAI()

    utils.log("Preparing jobs")
    jobs = []
    for rec in query_papers:
        rec["client"] = openai_client
        rec["model"] = args.model
        rec["ground_truth_title"] = rec["title"]
        rec["ground_truth_abstract"] = rec["abstract"]
        jobs.append(rec)

    utils.log(f"Running {len(jobs)} queries with {args.num_workers} workers")
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        results_iter = ex.map(query_paraphrase, jobs)
        for rec in tqdm(results_iter, total=len(jobs), desc="Generating paraphrases"):
            pass

    output_path = os.path.join(args.output_dir, f"generations.paraphrased_gold-{args.model}.json")
    utils.save_json(query_papers, output_path, overwrite=True, metadata=utils.update_metadata([], args))
    utils.log(f"Saved {len(query_papers)} generations to {output_path}")


if __name__ == "__main__":
    main()
