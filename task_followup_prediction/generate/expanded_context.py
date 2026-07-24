"""Expanded-context helpers for followup prediction baselines (author history + related papers via FAISS)."""

from itertools import groupby

import utils


def get_embedding(all_embeddings, corpus_id):
    return all_embeddings[corpus_id]["key"].reshape(-1)


def select_author_papers(target_record, all_papers_dict, per_author_cap, total_cap):
    """Greedy left-to-right fill across authors with per-author and total caps. Returns list of paper dicts."""
    target_id = target_record["corpus_id"]
    target_date = target_record["date"]
    seen = set()
    selected = []
    for author in target_record["authors"]:
        pubs = author.get("publication_history") or []
        dated = sorted(((all_papers_dict[cid]["date"], cid) for cid in pubs if cid in all_papers_dict), reverse=True)
        taken = 0
        for date, cid in dated:
            if taken >= per_author_cap or len(selected) >= total_cap:
                break
            if cid == target_id or cid in seen:
                continue
            assert date < target_date, f"Leakage: author paper {cid} date {date} >= target {target_id} date {target_date}"
            seen.add(cid)
            selected.append(all_papers_dict[cid])
            taken += 1
        if len(selected) >= total_cap:
            break
    return selected


def build_faiss_index(all_papers_dict, all_embeddings, cutoff_date, distance_metric):
    """Build FAISS index over papers strictly predating cutoff_date. Returns index tuple or None if empty."""
    paper_embeddings = {cid: get_embedding(all_embeddings, cid) for cid, p in all_papers_dict.items() if p["date"] < cutoff_date and cid in all_embeddings}
    if len(paper_embeddings) == 0:
        return None
    return utils.create_index(paper_embeddings, distance_metric)


def retrieve_related_papers(target_record, index, all_embeddings, all_papers_dict, k, exclude_ids, distance_metric):
    """Retrieve top-k related papers via per-reference rank fusion over key_references (mirrors task_priorwork_prediction rank-fusion pattern). Returns list of paper dicts."""
    if index is None:
        return []
    key_ref_embs = [get_embedding(all_embeddings, r["corpus_id"]) for r in target_record["key_references"] if r["corpus_id"] in all_embeddings]
    if len(key_ref_embs) == 0:
        return []
    oversample_k = max(k * 4, k + len(exclude_ids) + 10)
    retrieved_ids_lists, _ = utils.query_index(index, key_ref_embs, oversample_k)
    summed_ranks = {}
    for retrieved in retrieved_ids_lists:
        for rank, cid in enumerate(retrieved):
            if cid is None or cid in exclude_ids or cid not in all_papers_dict:
                continue
            summed_ranks[cid] = summed_ranks.get(cid, 0) + rank
    selected = []
    for cid, _ in sorted(summed_ranks.items(), key=lambda x: x[1]):
        assert all_papers_dict[cid]["date"] < target_record["date"], f"Leakage: related paper {cid} date >= target {target_record['corpus_id']} date"
        selected.append(all_papers_dict[cid])
        if len(selected) >= k:
            break
    return selected


def format_section(papers, label):
    out = ""
    for i, p in enumerate(papers):
        out += f"{label} {i+1}:\nTitle: {p['title']}\nAbstract: {p['abstract']}\n\n"
    return out


def get_target_impact(target_record):
    """Return (citations, horizon_months) for the target's eventual impact, or (None, None) if no trajectory."""
    traj = target_record.get("citation_trajectory") or []
    if not traj:
        return None, None
    if len(traj) >= 12:
        return traj[11], 12
    return traj[-1], len(traj)


def format_impact_line(target_record):
    """Format the oracle-impact line. Returns empty string if no trajectory data."""
    cites, horizon = get_target_impact(target_record)
    if cites is None:
        return ""
    return f"Target impact (intentional oracle signal): the followup paper you must predict went on to receive {cites} citations within {horizon} months of its publication.\n\n"


def format_expanded_user_prompt(target_record, all_papers_dict, author_papers, related_papers, include_impact=False):
    """Assemble user prompt from optional impact line + key_references + optional author history + optional related papers."""
    key_ref_papers = [all_papers_dict[r["corpus_id"]] for r in target_record["key_references"]]
    for p in key_ref_papers:
        assert p["date"] < target_record["date"], f"Leakage: key_ref {p['corpus_id']} date >= target {target_record['corpus_id']} date"
    prompt = format_impact_line(target_record) if include_impact else ""
    prompt += format_section(key_ref_papers, "Background Paper")
    if author_papers:
        prompt += format_section(author_papers, "Recent Author Publication")
    if related_papers:
        prompt += format_section(related_papers, "Related Paper")
    return prompt.rstrip() + "\n"


def iter_date_groups(all_papers_sorted):
    """Yield (date, list_of_papers_with_that_date) in date order. Caller must pass a date-sorted list."""
    for date, group in groupby(all_papers_sorted, key=lambda p: p["date"]):
        yield date, list(group)


def build_messages_with_faiss_walk(query_papers_dict, all_papers_sorted, all_papers_dict, all_embeddings, distance_metric, system_prompt, num_author_papers_per_author, max_total_author_papers, num_related_papers, include_author_history, include_related_papers, include_impact=False):
    """Date-grouped walk building per-target messages. Queries FAISS before adding same-date papers to avoid same-day leakage. Returns list of (corpus_id, messages, n_author_used, n_related_used)."""
    if include_related_papers:
        earliest_query_date = min(p["date"] for p in query_papers_dict.values())
        pre_cutoff = {cid: p for cid, p in all_papers_dict.items() if p["date"] < earliest_query_date}
        index = build_faiss_index(pre_cutoff, all_embeddings, earliest_query_date, distance_metric)
    else:
        index = None

    results = []
    for date, group in iter_date_groups(all_papers_sorted):
        for paper in group:
            cid = paper["corpus_id"]
            if cid not in query_papers_dict:
                continue
            target = query_papers_dict[cid]
            author_papers = select_author_papers(target, all_papers_dict, num_author_papers_per_author, max_total_author_papers) if include_author_history else []
            exclude_ids = {r["corpus_id"] for r in target["key_references"]} | {cid} | {p["corpus_id"] for p in author_papers}
            related_papers = retrieve_related_papers(target, index, all_embeddings, all_papers_dict, num_related_papers, exclude_ids, distance_metric) if include_related_papers else []
            user_prompt = format_expanded_user_prompt(target, all_papers_dict, author_papers, related_papers, include_impact=include_impact)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            results.append((cid, messages, len(author_papers), len(related_papers)))
        if include_related_papers:
            for paper in group:
                pid = paper["corpus_id"]
                if pid not in all_embeddings:
                    continue
                vec = get_embedding(all_embeddings, pid)
                if index is None:
                    index = utils.create_index({pid: vec}, distance_metric)
                else:
                    index = utils.add_vector_to_index(index, pid, vec)
    return results
