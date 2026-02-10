"""LLM-based failure mode categorization with emergent taxonomy discovery."""
import os
import re
import random
import argparse
import concurrent.futures as cf

import openai
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils


TAXONOMY_DISCOVERY_PROMPT = """Here are {N} examples of failed followup paper predictions with their LACER evaluation reasoning.
Each evaluation explains why the generated paper didn't match the reference followup paper.

{samples}

Based on these examples, propose a taxonomy of 4-6 distinct failure modes that capture the main reasons predictions fail.

For each category, provide:
1. A short name (2-4 words)
2. A one-sentence description

Return ONLY a numbered list in this exact format:
1. [Name]: [Description]
2. [Name]: [Description]
...

Do not include examples or additional text."""

CATEGORIZATION_PROMPT = """Given this LACER evaluation reasoning that compares a generated paper to a reference paper, categorize the primary failure mode.

LACER Evaluation Reasoning:
{lacer_response}

Categories:
{categories}

Respond with ONLY the category number (1-{num_categories}).
"""

TAXONOMY_REGEX = re.compile(r"(\d+)\.\s*\[?([^\]:]+)\]?\s*[:\-]\s*(.+)")
CATEGORY_REGEX = re.compile(r"^\s*(\d+)")


def parse_taxonomy(response_text):
    """Parse taxonomy from model response."""
    categories = {}
    for line in response_text.strip().split("\n"):
        match = TAXONOMY_REGEX.search(line)
        if match:
            num = int(match.group(1))
            name = match.group(2).strip().strip("[]")
            desc = match.group(3).strip()
            categories[num] = {"name": name, "description": desc}
    return categories


def parse_category(response_text, num_categories):
    """Parse category number from model response."""
    match = CATEGORY_REGEX.search(response_text.strip())
    if match:
        cat = int(match.group(1))
        if 1 <= cat <= num_categories:
            return cat
    return num_categories  # Default to last category


def discover_taxonomy(client, model, failures, num_samples=25):
    """Discover failure mode taxonomy from samples."""
    samples = random.sample(failures, min(num_samples, len(failures)))
    samples_text = "\n\n".join([
        f"Example {i+1} (LACER = {s['lacer_score']:.1f}):\n{s['lacer_response'][:1500]}"
        for i, s in enumerate(samples)
    ])

    prompt = TAXONOMY_DISCOVERY_PROMPT.format(N=len(samples), samples=samples_text)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1024,
    )
    content = response.choices[0].message.content.strip()
    return parse_taxonomy(content), content


def categorize_instance(record):
    """Categorize a single failure instance."""
    client = record["client"]
    model = record["model"]
    lacer_response = record["lacer_response"]
    categories = record["categories"]
    num_categories = len(categories)

    categories_text = "\n".join([
        f"{num}. {cat['name']}: {cat['description']}"
        for num, cat in sorted(categories.items())
    ])

    prompt = CATEGORIZATION_PROMPT.format(
        lacer_response=lacer_response[:2000],
        categories=categories_text,
        num_categories=num_categories
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=16,
        )
        content = response.choices[0].message.content.strip()
        category = parse_category(content, num_categories)
        return {
            "corpus_id": record["corpus_id"],
            "lacer_score": record["lacer_score"],
            "category": category,
            "category_name": categories[category]["name"],
            "raw_response": content,
        }
    except Exception as exc:
        return {
            "corpus_id": record["corpus_id"],
            "lacer_score": record["lacer_score"],
            "category": num_categories,
            "category_name": categories[num_categories]["name"],
            "raw_response": f"Error: {str(exc)}",
        }


def main():
    parser = argparse.ArgumentParser(description="Categorize failure modes with emergent taxonomy")
    parser.add_argument("--scored_path", type=str, default="data/task_followup_prediction/test/lacer_scored_opus/generations.gpt-5.2-2025-12-11.lacer_scored.json", help="Path to LACER-scored file")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/analysis", help="Output directory")
    parser.add_argument("--threshold", type=float, default=4.0, help="LACER score threshold for failure")
    parser.add_argument("--model", type=str, default="gpt-5.2-2025-12-11", help="Model for categorization")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--taxonomy_samples", type=int, default=25, help="Number of samples for taxonomy discovery")
    parser.add_argument("--max_instances", type=int, default=None, help="Optional cap on instances")
    args = parser.parse_args()

    random.seed(42)

    utils.log(f"Loading scored data from {args.scored_path}")
    data_file, _ = utils.load_json(args.scored_path)
    data = data_file["data"] if "data" in data_file else data_file
    utils.log(f"Loaded {len(data)} instances")

    # Filter to low-scoring instances
    failures = [d for d in data if d["lacer_score"] < args.threshold]
    utils.log(f"Found {len(failures)} instances with LACER < {args.threshold}")

    if args.max_instances is not None:
        failures = failures[:args.max_instances]
        utils.log(f"Capped to {len(failures)} instances")

    # Initialize OpenAI client
    client = openai.OpenAI()

    # Phase 1: Discover taxonomy
    utils.log(f"Phase 1: Discovering taxonomy from {args.taxonomy_samples} samples using {args.model}")
    categories, raw_taxonomy = discover_taxonomy(client, args.model, failures, args.taxonomy_samples)
    utils.log(f"Discovered {len(categories)} categories:")
    for num, cat in sorted(categories.items()):
        utils.log(f"  {num}. {cat['name']}: {cat['description']}")

    # Save taxonomy
    os.makedirs(args.output_dir, exist_ok=True)
    taxonomy_path = os.path.join(args.output_dir, "failure_taxonomy.json")
    taxonomy_data = {
        "model": args.model,
        "num_samples": args.taxonomy_samples,
        "categories": {str(k): v for k, v in categories.items()},
        "raw_response": raw_taxonomy,
    }
    utils.save_json(taxonomy_data, taxonomy_path, utils.update_metadata([], args))
    utils.log(f"Saved taxonomy to {taxonomy_path}")

    # Phase 2: Categorize all failures
    utils.log(f"Phase 2: Categorizing {len(failures)} failures")
    jobs = []
    for instance in failures:
        job = {
            "corpus_id": instance["corpus_id"],
            "lacer_score": instance["lacer_score"],
            "lacer_response": instance["lacer_response"],
            "client": client,
            "model": args.model,
            "categories": categories,
        }
        jobs.append(job)

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(categorize_instance, job) for job in jobs]
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Categorizing"):
            results.append(future.result())

    # Aggregate results
    category_counts = {num: 0 for num in categories}
    for result in results:
        category_counts[result["category"]] += 1

    total = len(results)
    utils.log("Failure mode distribution:")
    for cat_num in sorted(categories.keys()):
        count = category_counts[cat_num]
        pct = 100.0 * count / total if total > 0 else 0
        utils.log(f"  {cat_num}. {categories[cat_num]['name']}: {count} ({pct:.1f}%)")

    # Save detailed results
    output_json_path = os.path.join(args.output_dir, "failure_modes_gpt5.2.json")
    output_data = {
        "threshold": args.threshold,
        "total_failures": total,
        "taxonomy": {str(k): v for k, v in categories.items()},
        "category_counts": {categories[k]["name"]: v for k, v in category_counts.items()},
        "instances": results,
    }
    utils.save_json(output_data, output_json_path, utils.update_metadata([], args))
    utils.log(f"Saved detailed results to {output_json_path}")

    # Create bar chart
    utils.log("Creating plot")
    plt.figure(figsize=(3.5, 2.8))
    plt.rcParams.update({"font.size": 8})

    labels = [categories[num]["name"] for num in sorted(categories.keys())]
    counts = [category_counts[num] for num in sorted(categories.keys())]
    percentages = [100.0 * c / total if total > 0 else 0 for c in counts]

    x = np.arange(len(labels))
    bars = plt.bar(x, percentages, color="#0072B2", alpha=0.8)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(count), ha="center", va="bottom", fontsize=6)

    plt.xlabel("Failure Mode", fontsize=8)
    plt.ylabel("Percentage (%)", fontsize=8)
    plt.title(f"Failure Mode Distribution (LACER < {args.threshold})", fontsize=9, fontweight="bold")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=6)
    plt.yticks(fontsize=7)
    plt.ylim(0, max(percentages) * 1.2 if percentages else 100)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, axis="y")
    plt.tight_layout(pad=0.5)

    output_plot_path = os.path.join(args.output_dir, "failure_mode_distribution.png")
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_plot_path}")


if __name__ == "__main__":
    main()
