"""Simulation script for extrapolating scientific research trajectories.

Composes co-author prediction, prior work selection, and followup generation
baselines to iteratively generate synthetic papers day by day.
"""

import os
import copy
import random
import argparse
import datetime
import numpy as np
import concurrent.futures as cf
from collections import Counter
from tqdm import tqdm

import openai
import anthropic

import utils

# Import coauthor prediction functions
from task_coauthor_prediction.dataset import get_preexisting_publications_for_author
from task_coauthor_prediction.baseline_frequency import (
    predict_coauthors_one_shot as predict_coauthors_frequency_one_shot,
    predict_coauthors_iterated as predict_coauthors_frequency_iterated,
)
from task_coauthor_prediction.baseline_embedding_fusion import (
    get_embedding as get_paper_embedding,
    create_mean_author_embedding,
    predict_coauthors_one_shot as predict_coauthors_embedding_fusion_one_shot,
    predict_coauthors_iterated as predict_coauthors_embedding_fusion_iterated,
)
from task_coauthor_prediction.baseline_rank_fusion import (
    predict_coauthors_one_shot as predict_coauthors_rank_fusion_one_shot,
    predict_coauthors_iterated as predict_coauthors_rank_fusion_iterated,
)

# Import prior work prediction functions
from task_priorwork_prediction.baseline_frequency import predict_references as predict_references_frequency
from task_priorwork_prediction.baseline_embedding_fusion import predict_references as predict_references_embedding_fusion
from task_priorwork_prediction.baseline_rank_fusion import predict_references as predict_references_rank_fusion

# Import followup generation functions
from task_followup_prediction.generate.baseline_gpt_parallel import create_messages, query_gpt
from task_followup_prediction.generate.baseline_claude_parallel import query_claude

# Constants
SYNTHETIC_CORPUS_ID_PREFIX = "S"
SYNTHETIC_AUTHOR_ID_PREFIX = "SYNTH_"


def compute_daily_paper_statistics(target_papers):
    """Compute mean and std of daily paper counts from target papers."""
    daily_counts = Counter(p["date"] for p in target_papers)
    counts_array = np.array(list(daily_counts.values()), dtype=float)
    mean_count = float(np.mean(counts_array))
    std_count = float(np.std(counts_array, ddof=1)) if len(counts_array) > 1 else 0.0
    return mean_count, std_count


def compute_smoothed_count_distribution(counts):
    """Compute smoothed empirical distribution from count data."""
    frequency = Counter(counts)
    values = np.array(sorted(frequency.keys()), dtype=int)
    smoothed = np.array([frequency[v] + 1 for v in values], dtype=float)
    probabilities = smoothed / smoothed.sum()
    return values, probabilities


def sample_from_distribution(distribution):
    """Sample a value from a (values, probabilities) distribution."""
    values, probabilities = distribution
    return int(np.random.choice(values, p=probabilities))


def compute_first_time_author_probability(target_papers):
    """Compute the fraction of authors who have no prior publications."""
    total_authors = 0
    first_time_authors = 0
    for paper in target_papers:
        for author in paper["authors"]:
            total_authors += 1
            if len(author["publication_history"]) == 0:
                first_time_authors += 1
    if total_authors == 0:
        return 0.0
    return first_time_authors / total_authors


def get_new_synthetic_corpus_id(state):
    """Generate a new synthetic corpus ID."""
    next_int = state["next_synthetic_corpus_id"]
    state["next_synthetic_corpus_id"] = next_int + 1
    return f"{SYNTHETIC_CORPUS_ID_PREFIX}{next_int:08d}"


def get_new_synthetic_author_id(state):
    """Generate a new synthetic author ID and initialize their publication history."""
    next_int = state["next_synthetic_author_id"]
    state["next_synthetic_author_id"] = next_int + 1
    new_author_id = f"{SYNTHETIC_AUTHOR_ID_PREFIX}{next_int:08d}"
    state["sd2publications"][new_author_id] = []
    return new_author_id


def sample_author_candidate(state, allow_new=True):
    """Sample an existing author or create a new one based on first-time author probability."""
    existing_authors = list(state["sd2publications"].keys())
    first_time_prob = state["first_time_author_prob"]

    if allow_new and random.random() < first_time_prob:
        return get_new_synthetic_author_id(state)
    else:
        return random.choice(existing_authors)


def author_has_history(author_id, date, sd2publications, all_papers_dict):
    """Check if author has any publications before the given date."""
    pubs = get_preexisting_publications_for_author(author_id, date, sd2publications, all_papers_dict)
    return len(pubs) > 0


def get_distance_metric(index):
    """Extract distance metric from index tuple."""
    return index[3]


def predict_coauthors(seed_author_id, num_to_predict, k, date, state, args, excluded_ids=None):
    """Predict coauthors using the configured baseline and strategy."""
    sd2publications = state["sd2publications"]
    all_papers_dict = state["all_papers_dict"]
    author_index_embeddings = state["author_index_embeddings"]
    author_index = state["author_index"]
    all_author_ids = set(sd2publications.keys())
    distance_metric = get_distance_metric(author_index)

    if args.coauthor_baseline == "frequency":
        if args.coauthor_strategy == "one_shot":
            pred_ids, _ = predict_coauthors_frequency_one_shot(
                seed_author_id, k, args.num_recent_papers, date, sd2publications, all_papers_dict, excluded_ids
            )
        else:
            pred_ids, _ = predict_coauthors_frequency_iterated(
                seed_author_id, num_to_predict, k, args.num_recent_papers, date, sd2publications, all_papers_dict, excluded_ids
            )
    elif args.coauthor_baseline == "embedding_fusion":
        if args.coauthor_strategy == "one_shot":
            pred_ids, _ = predict_coauthors_embedding_fusion_one_shot(
                seed_author_id, args.num_recent_papers, k, author_index, date,
                sd2publications, all_papers_dict, author_index_embeddings, distance_metric, all_author_ids, excluded_ids
            )
        else:
            pred_ids, _ = predict_coauthors_embedding_fusion_iterated(
                seed_author_id, args.num_recent_papers, num_to_predict, k, author_index, date,
                sd2publications, all_papers_dict, author_index_embeddings, distance_metric, all_author_ids, excluded_ids
            )
    elif args.coauthor_baseline == "rank_fusion":
        if args.coauthor_strategy == "one_shot":
            pred_ids, _ = predict_coauthors_rank_fusion_one_shot(
                seed_author_id, args.num_recent_papers, k, author_index, date,
                sd2publications, all_papers_dict, author_index_embeddings, distance_metric, all_author_ids, excluded_ids
            )
        else:
            pred_ids, _ = predict_coauthors_rank_fusion_iterated(
                seed_author_id, args.num_recent_papers, num_to_predict, k, author_index, date,
                sd2publications, all_papers_dict, author_index_embeddings, distance_metric, all_author_ids, excluded_ids
            )
    else:
        raise ValueError(f"Unknown coauthor baseline: {args.coauthor_baseline}")

    return pred_ids


def predict_prior_work(author_ids, num_to_predict, date, state, args):
    """Predict prior work references using the configured baseline.

    Falls back to random sampling if baseline returns too few results
    (e.g., when all authors are first-time with no publication history).
    """
    sd2publications = state["sd2publications"]
    all_papers_dict = state["all_papers_dict"]
    paper_index_embeddings = state["paper_index_embeddings"]
    paper_index = state["paper_index"]
    all_corpus_ids = list(all_papers_dict.keys())
    distance_metric = get_distance_metric(paper_index)

    if args.priorwork_baseline == "frequency":
        pred_ids, _ = predict_references_frequency(
            author_ids, date, args.num_recent_papers, num_to_predict,
            sd2publications, all_papers_dict, all_corpus_ids
        )
    elif args.priorwork_baseline == "embedding_fusion":
        pred_ids, _ = predict_references_embedding_fusion(
            author_ids, date, args.num_recent_papers, num_to_predict,
            paper_index, sd2publications, all_papers_dict, paper_index_embeddings, distance_metric
        )
    elif args.priorwork_baseline == "rank_fusion":
        pred_ids, _ = predict_references_rank_fusion(
            author_ids, date, args.num_recent_papers, num_to_predict,
            paper_index, sd2publications, all_papers_dict, paper_index_embeddings, distance_metric
        )
    else:
        raise ValueError(f"Unknown priorwork baseline: {args.priorwork_baseline}")

    # Pad with random papers if baseline returned too few (e.g., all first-time authors)
    if len(pred_ids) < num_to_predict:
        excluded = set(pred_ids)
        available_papers = [cid for cid in all_corpus_ids if cid not in excluded and all_papers_dict[cid]["date"] < date]
        num_needed = num_to_predict - len(pred_ids)
        random_papers = random.sample(available_papers, min(num_needed, len(available_papers)))
        pred_ids = list(pred_ids) + random_papers

    return pred_ids

def generate_followups_batch(papers_data, all_papers_dict, system_prompt, client, model, backend, num_workers):
    """Generate followup papers in parallel."""
    jobs = []
    for key_reference_ids in papers_data:
        rec = {
            "key_references": [{"corpus_id": cid} for cid in key_reference_ids],
            "client": client,
            "model": model,
            "messages": create_messages(
                {"key_references": [{"corpus_id": cid} for cid in key_reference_ids]},
                system_prompt,
                all_papers_dict
            ),
        }
        jobs.append(rec)

    query_fn = query_gpt if backend == "openai" else query_claude

    # Use future-to-index mapping to preserve submission order
    results = [None] * len(jobs)
    with cf.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {executor.submit(query_fn, job): idx for idx, job in enumerate(jobs)}
        for future in tqdm(cf.as_completed(future_to_idx), total=len(future_to_idx), desc="Generating followups"):
            idx = future_to_idx[future]
            result = future.result()
            if "title" in result and "abstract" in result and result["title"] and result["abstract"]:
                results[idx] = {"title": result["title"], "abstract": result["abstract"]}

    return results


def generate_author_team(num_authors, date, state, args):
    """Generate a team of authors for a new paper.

    May return a team of all first-time authors if no seed with history is found.
    In that case, prior work prediction will fall back to random sampling.
    """
    sd2publications = state["sd2publications"]
    all_papers_dict = state["all_papers_dict"]

    author_ids = []
    author_set = set()

    # Sample first author
    first_author = sample_author_candidate(state, allow_new=True)
    author_ids.append(first_author)
    author_set.add(first_author)

    if num_authors == 1:
        return author_ids

    # Find a seed author with publication history for coauthor prediction
    seed_author = first_author if author_has_history(first_author, date, sd2publications, all_papers_dict) else None

    # If first author has no history, keep sampling until we find a seed or fill all slots
    while seed_author is None and len(author_ids) < num_authors:
        candidate = sample_author_candidate(state, allow_new=True)
        if candidate not in author_set:
            author_ids.append(candidate)
            author_set.add(candidate)
            if author_has_history(candidate, date, sd2publications, all_papers_dict):
                seed_author = candidate

    # Use coauthor prediction to fill remaining slots if we have a seed
    if seed_author is not None and len(author_ids) < num_authors:
        prediction_pool_size = max(num_authors * 2, num_authors + 10)
        predicted_candidates = predict_coauthors(
            seed_author, num_authors - len(author_ids), prediction_pool_size, date, state, args, author_set
        )
        for candidate in predicted_candidates:
            if candidate not in author_set:
                if candidate not in sd2publications:
                    sd2publications[candidate] = []
                author_ids.append(candidate)
                author_set.add(candidate)
                if len(author_ids) >= num_authors:
                    break

    return author_ids


def create_author_index(cutoff_date, num_recent_papers, sd2publications, all_papers_dict, all_embeddings, distance_metric, use_gpu=True):
    """Create FAISS index of author embeddings from their recent publications."""
    author_index_embeddings = {}

    for author_id, pub_ids in tqdm(sd2publications.items(), desc="Creating author embeddings"):
        if pub_ids is None:
            continue
        # Filter to publications on or before cutoff date
        valid_pubs = [cid for cid in pub_ids if cid in all_papers_dict and all_papers_dict[cid]["date"] <= cutoff_date]
        if len(valid_pubs) > 0:
            recent_pubs = valid_pubs[-num_recent_papers:]
            author_index_embeddings[author_id] = create_mean_author_embedding(recent_pubs, all_embeddings, distance_metric)

    utils.log(f"Created embeddings for {len(author_index_embeddings)} authors")
    return utils.create_index(author_index_embeddings, distance_metric, use_gpu=use_gpu)


def create_paper_index(cutoff_date, all_papers_dict, all_embeddings, distance_metric, use_gpu=True):
    """Create FAISS index of paper embeddings for papers on or before cutoff date."""
    paper_index_embeddings = {}

    for corpus_id, paper in all_papers_dict.items():
        if paper["date"] <= cutoff_date and corpus_id in all_embeddings:
            paper_index_embeddings[corpus_id] = get_paper_embedding(all_embeddings, corpus_id)

    utils.log(f"Created paper index with {len(paper_index_embeddings)} papers")
    return utils.create_index(paper_index_embeddings, distance_metric, use_gpu=use_gpu)


def create_initial_state(args):
    """Create initial simulation state from input data."""
    utils.log(f"Loading data from HuggingFace: {args.hf_repo_id} ({args.calibration_split} split)")

    # Load corpus with paper embeddings
    all_papers, sd2publications, paper_index_embeddings = utils.load_corpus(
        hf_repo_id=args.hf_repo_id,
        split=args.calibration_split,
        embeddings_dir=args.embeddings_dir,
        embedding_type=args.paper_embedding_type,
        load_sd2publications=True
    )

    # Load second embedding type for author index
    author_index_embeddings, _ = utils.load_pkl(
        os.path.join(args.embeddings_dir, f"all_papers.{args.author_embedding_type}_embeddings.pkl")
    )

    # Deep copy sd2publications (simulate.py mutates it)
    sd2publications = copy.deepcopy(sd2publications)

    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    target_papers = utils.filter_by_roles(all_papers, ["target"])
    synthetic_papers = utils.filter_by_roles(all_papers, ["synthetic"])
    final_date = max(p["date"] for p in all_papers)

    author_distance_metric = "l2" if args.author_embedding_type == "specter2" else "cosine"
    paper_distance_metric = "l2" if args.paper_embedding_type == "specter2" else "cosine"

    utils.log(f"Loaded {len(all_papers)} papers ({len(target_papers)} target, {len(synthetic_papers)} synthetic), {len(sd2publications)} authors")
    utils.log(f"Author embeddings ({args.author_embedding_type}): {len(author_index_embeddings)}, Paper embeddings ({args.paper_embedding_type}): {len(paper_index_embeddings)}")
    utils.log(f"Final date in corpus: {final_date}")

    # Create indexes with their respective embeddings
    utils.log(f"Creating author index with {args.author_embedding_type} embeddings...")
    author_index = create_author_index(final_date, args.num_recent_papers, sd2publications, all_papers_dict, author_index_embeddings, author_distance_metric, use_gpu=not args.use_cpu_index)

    utils.log(f"Creating paper index with {args.paper_embedding_type} embeddings...")
    paper_index = create_paper_index(final_date, all_papers_dict, paper_index_embeddings, paper_distance_metric, use_gpu=not args.use_cpu_index)

    # Compute statistics for sampling
    utils.log("Computing statistical distributions...")
    daily_paper_stats = compute_daily_paper_statistics(target_papers)
    author_counts = [len(p["authors"]) for p in target_papers]
    key_reference_counts = [len(p["key_references"]) for p in target_papers]
    first_time_author_prob = compute_first_time_author_probability(target_papers)

    utils.log(f"Daily paper count: mean={daily_paper_stats[0]:.2f}, std={daily_paper_stats[1]:.2f}")
    utils.log(f"First-time author probability: {first_time_author_prob:.4f}")

    # Find highest existing synthetic corpus ID
    existing_synthetic_ids = [int(cid.replace(SYNTHETIC_CORPUS_ID_PREFIX, "")) for cid in all_papers_dict.keys() if cid.startswith(SYNTHETIC_CORPUS_ID_PREFIX)]
    next_synthetic_corpus_id = max(existing_synthetic_ids, default=0) + 1

    # Find highest existing synthetic author ID
    existing_synthetic_author_ids = [int(aid.replace(SYNTHETIC_AUTHOR_ID_PREFIX, "")) for aid in sd2publications.keys() if aid.startswith(SYNTHETIC_AUTHOR_ID_PREFIX)]
    next_synthetic_author_id = max(existing_synthetic_author_ids, default=0) + 1

    state = {
        "all_papers_dict": all_papers_dict,
        "author_index_embeddings": author_index_embeddings,
        "paper_index_embeddings": paper_index_embeddings,
        "sd2publications": sd2publications,
        "author_index": author_index,
        "paper_index": paper_index,
        "final_date": final_date,
        "daily_paper_stats": daily_paper_stats,
        "author_count_distribution": compute_smoothed_count_distribution(author_counts),
        "key_reference_distribution": compute_smoothed_count_distribution(key_reference_counts),
        "first_time_author_prob": first_time_author_prob,
        "next_synthetic_corpus_id": next_synthetic_corpus_id,
        "next_synthetic_author_id": next_synthetic_author_id,
    }

    return state


def fold_in_papers(state, new_papers, date, args):
    """Fold newly generated papers into the simulation state."""
    if len(new_papers) == 0:
        return state

    all_papers_dict = state["all_papers_dict"]
    author_index_embeddings = state["author_index_embeddings"]
    paper_index_embeddings = state["paper_index_embeddings"]
    sd2publications = state["sd2publications"]
    paper_index = state["paper_index"]
    author_index = state["author_index"]
    author_distance_metric = get_distance_metric(author_index)

    # Batch compute embeddings for new papers
    paper_strings = [utils.get_title_abstract_string(p) for p in new_papers]
    new_author_index_embeddings = utils.embed_on_gpus_parallel(paper_strings, args.author_embedding_type)
    new_paper_index_embeddings = copy.deepcopy(new_author_index_embeddings) if \
                                    args.author_embedding_type == args.paper_embedding_type else \
                                    utils.embed_on_gpus_parallel(paper_strings, args.paper_embedding_type)

    # Process each paper
    affected_authors = set()
    for i, paper in enumerate(new_papers):
        corpus_id = paper["corpus_id"]
        author_ids = [a["author_id"] for a in paper["authors"]]

        # Add to all_papers_dict
        all_papers_dict[corpus_id] = paper

        # Add embeddings
        author_index_embeddings[corpus_id] = new_author_index_embeddings[i]
        paper_index_embeddings[corpus_id] = new_paper_index_embeddings[i]

        # Update sd2publications for each author
        for author_id in author_ids:
            if author_id not in sd2publications or sd2publications[author_id] is None:
                sd2publications[author_id] = []
            sd2publications[author_id].append(corpus_id)
            affected_authors.add(author_id)

        # Add to paper index
        paper_index = utils.add_vector_to_index(paper_index, corpus_id, get_paper_embedding(paper_index_embeddings, corpus_id))

        # Mark key references
        for ref in paper["key_references"]:
            ref_cid = ref["corpus_id"]
            if ref_cid in all_papers_dict:
                ref_paper = all_papers_dict[ref_cid]
                if "synthetic.key_reference" not in ref_paper["roles"]:
                    ref_paper["roles"] = sorted(list(set(ref_paper["roles"] + ["synthetic.key_reference"])))

        # Mark author publication history papers and their key references
        for author in paper["authors"]:
            # publication_history may be None for authors with untracked history; treat as []
            for pub_cid in (author.get("publication_history") or []):
                if pub_cid in all_papers_dict:
                    pub_paper = all_papers_dict[pub_cid]
                    if "synthetic.author.publication_history" not in pub_paper["roles"]:
                        pub_paper["roles"] = sorted(list(set(pub_paper["roles"] + ["synthetic.author.publication_history"])))
                    for pub_ref in pub_paper.get("key_references", []):
                        pub_ref_cid = pub_ref["corpus_id"]
                        if pub_ref_cid in all_papers_dict:
                            pub_ref_paper = all_papers_dict[pub_ref_cid]
                            if "synthetic.author.publication_history.key_reference" not in pub_ref_paper["roles"]:
                                pub_ref_paper["roles"] = sorted(list(set(pub_ref_paper["roles"] + ["synthetic.author.publication_history.key_reference"])))

    # Update author embeddings for affected authors
    for author_id in affected_authors:
        pub_ids = sd2publications[author_id]
        valid_pubs = [cid for cid in pub_ids if cid in author_index_embeddings]
        if len(valid_pubs) > 0:
            recent_pubs = valid_pubs[-args.num_recent_papers:]
            author_embedding = create_mean_author_embedding(recent_pubs, author_index_embeddings, author_distance_metric)
            author_index = utils.replace_vector_in_index(author_index, author_id, author_embedding)

    # Update state (explicit assignments for clarity even if it happens by reference implicitly)
    state["all_papers_dict"] = all_papers_dict
    state["author_index_embeddings"] = author_index_embeddings
    state["paper_index_embeddings"] = paper_index_embeddings
    state["sd2publications"] = sd2publications
    state["paper_index"] = paper_index
    state["author_index"] = author_index
    state["final_date"] = date
    # next_synthetic_corpus_id and next_synthetic_author_id are updated by get_new_synthetic_corpus_id() and get_new_synthetic_author_id() respectively

    return state


def sample_new_papers_for_day(state, date, args, client, system_prompt):
    """Sample and generate new papers for a single day."""
    mean_daily, std_daily = state["daily_paper_stats"]
    num_new_papers = np.random.normal(mean_daily, std_daily) if std_daily > 0 else mean_daily
    num_new_papers = max(int(round(num_new_papers)), 0)

    if num_new_papers == 0:
        return []

    all_papers_dict = state["all_papers_dict"]
    sd2publications = state["sd2publications"]

    # Generate author teams and key references for all papers
    all_author_ids = []
    all_key_ref_ids = []
    for _ in range(num_new_papers):
        # Sample author count and generate team
        author_count = sample_from_distribution(state["author_count_distribution"])
        author_ids = generate_author_team(author_count, date, state, args)

        # Sample key reference count and predict references
        key_ref_count = sample_from_distribution(state["key_reference_distribution"])
        key_ref_ids = predict_prior_work(author_ids, key_ref_count, date, state, args)

        all_author_ids.append(author_ids)
        all_key_ref_ids.append(key_ref_ids)

    # Generate title/abstracts in parallel
    title_abstracts = generate_followups_batch(
        all_key_ref_ids, all_papers_dict, system_prompt, client,
        args.generation_model, args.generation_backend, args.num_workers
    )

    # Create paper records
    new_papers = []
    for author_ids, key_ref_ids, title_abstract in zip(all_author_ids, all_key_ref_ids, title_abstracts):
        if title_abstract is not None:
            paper_record = {
                "corpus_id": get_new_synthetic_corpus_id(state),
                "date": date,
                "title": title_abstract["title"],
                "abstract": title_abstract["abstract"],
                "roles": ["synthetic"],
                "key_references": [{"corpus_id": cid} for cid in key_ref_ids],
                "authors": [
                    # sd2publications[aid] may be None for authors with untracked history; normalize to []
                    {"author_id": aid, "publication_history": copy.deepcopy(sd2publications.get(aid) or [])}
                    for aid in author_ids
                ],
            }
            new_papers.append(paper_record)

    return new_papers


def simulate(state, args):
    """Run the main simulation loop."""
    initial_date = state["final_date"]
    first_sim_date = (datetime.datetime.strptime(initial_date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    utils.log(f"Starting simulation from {first_sim_date} for {args.depth} days")

    # Initialize API client
    client = openai.OpenAI() if args.generation_backend == "openai" else anthropic.Anthropic()
    
    # Load system prompt
    with open("task_followup_prediction/templates/prediction_system.prompt", "r") as f:
        system_prompt = f.read()

    total_papers_generated = 0
    pbar = tqdm(range(1, args.depth + 1), desc="Simulating days")
    for d in pbar:
        current_date = (datetime.datetime.strptime(initial_date, "%Y-%m-%d") + datetime.timedelta(days=d)).strftime("%Y-%m-%d")
        pbar.set_description(f"Day {d}/{args.depth} ({current_date})")

        # Generate new papers for this day
        new_papers = sample_new_papers_for_day(state, current_date, args, client, system_prompt)
        state = fold_in_papers(state, new_papers, current_date, args)
        total_papers_generated += len(new_papers)

        # Checkpoint
        if d % args.checkpoint_every == 0:
            utils.log(f"Day {d}/{args.depth} ({current_date}): {len(new_papers)} papers, {total_papers_generated} total")
            save_state(state, args)

    utils.log(f"Simulation complete: {total_papers_generated} papers generated over {args.depth} days")
    return state


def save_state(state, args):
    """Save simulation state to output directory."""
    all_papers = sorted(list(state["all_papers_dict"].values()), key=lambda p: p["date"])
    metadata = utils.update_metadata([], args)

    utils.save_json(all_papers, os.path.join(args.output_dir, "all_papers.json"), metadata=metadata, overwrite=True)
    utils.save_pkl(state["author_index_embeddings"], os.path.join(args.output_dir, f"all_papers.{args.author_embedding_type}_embeddings.pkl"), overwrite=True)
    utils.save_pkl(state["paper_index_embeddings"], os.path.join(args.output_dir, f"all_papers.{args.paper_embedding_type}_embeddings.pkl"), overwrite=True)
    utils.save_json(state["sd2publications"], os.path.join(args.output_dir, "sd2publications.json"), overwrite=True)


def main():
    parser = argparse.ArgumentParser(description="Simulate scientific literature trajectory")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--calibration_split", type=str, default="train", choices=["train", "test"], help="Dataset split to use as initial corpus for simulation")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing embedding files")
    parser.add_argument("--output_dir", type=str, default="data/multiturn/simulated", help="Directory to save simulated output")
    parser.add_argument("--depth", type=int, default=365, help="Number of days to simulate")
    parser.add_argument("--author_embedding_type", type=str, default="grit", choices=["gtr", "grit", "specter2"], help="Embedding type for author index")
    parser.add_argument("--paper_embedding_type", type=str, default="grit", choices=["gtr", "grit", "specter2"], help="Embedding type for paper index")
    parser.add_argument("--coauthor_baseline", type=str, default="embedding_fusion", choices=["frequency", "embedding_fusion", "rank_fusion"], help="Baseline for coauthor prediction")
    parser.add_argument("--coauthor_strategy", type=str, default="one_shot", choices=["one_shot", "iterated"], help="Strategy for coauthor prediction")
    parser.add_argument("--priorwork_baseline", type=str, default="embedding_fusion", choices=["frequency", "embedding_fusion", "rank_fusion"], help="Baseline for prior work prediction")
    parser.add_argument("--generation_backend", type=str, default="openai", choices=["openai", "anthropic"], help="Backend for followup generation")
    parser.add_argument("--generation_model", type=str, default="gpt-5-2025-08-07", help="Model for followup generation")
    parser.add_argument("--num_recent_papers", type=int, default=10, help="Number of recent papers to consider per author")
    parser.add_argument("--num_workers", type=int, default=256, help="Number of parallel workers for generation")
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Save checkpoint every N days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_cpu_index", action="store_true", help="Use CPU FAISS index instead of GPU (slower but avoids GPU memory limits)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    state = create_initial_state(args)
    state = simulate(state, args)
    save_state(state, args)

    utils.log("Done!")


if __name__ == "__main__":
    main()
