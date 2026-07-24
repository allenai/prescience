"""Build per-ablation system prompt templates by replacing the 3 few-shot demos with real train-split targets, optionally expanded with author history, related papers, and oracle impact line."""

import argparse

import utils
from task_followup_prediction.generate.expanded_context import select_author_papers, build_faiss_index, retrieve_related_papers, format_section, get_target_impact


DEMO_CORPUS_IDS = ["272525247", "271601035", "271600824"]
MODES = ("vanilla", "authors", "related", "full")
IMPACT_INSTRUCTION = "\nEach example also includes a 'Target impact' line at the top, an oracle signal stating the eventual citation count of the target paper. This is provided as conditioning to help calibrate the ambition, scope, and framing of your prediction. High-impact papers tend to be broader, propose new framings, or address widely-applicable problems; low-impact papers are typically narrower in scope. Use this signal accordingly. Do NOT generate a 'Target impact' line in your output.\n"


def load_base_prompt():
    with open("task_followup_prediction/templates/prediction_system.prompt", "r") as f:
        return f.read()


def preamble(base_prompt):
    idx = base_prompt.find("<example 1>")
    return base_prompt[:idx]


def postamble(base_prompt):
    idx = base_prompt.rfind("</example 3>")
    return base_prompt[idx + len("</example 3>"):]


def build_instruction_block(mode, include_impact):
    sections = []
    if mode in ("full", "authors"):
        sections.append("'Recent Author Publication' entries (recent work by the authors of the to-be-predicted paper, capturing their research thread)")
    if mode in ("full", "related"):
        sections.append("'Related Paper' entries (other recent papers in the field, retrieved by semantic similarity to each background paper)")
    block = ""
    if sections:
        label = " and ".join(sections)
        block += f"\nIn addition to 'Background Paper' entries, each solved example below includes {label}. Use these auxiliary sections for context, but ground the predicted followup primarily in the Background Papers.\n"
    if include_impact:
        block += IMPACT_INSTRUCTION
    return block


def format_demo(demo_num, target, key_ref_papers, author_papers, related_papers, mode, include_impact):
    body = ""
    if include_impact:
        cites, horizon = get_target_impact(target)
        if cites is not None:
            body += f"Target impact (intentional oracle signal): the followup paper you must predict went on to receive {cites} citations within {horizon} months of its publication.\n\n"
    for i, p in enumerate(key_ref_papers):
        body += f"Background Paper {i+1}:\nTitle: {p['title']}\nAbstract: {p['abstract']}\n\n"
    if mode in ("full", "authors") and author_papers:
        body += format_section(author_papers, "Recent Author Publication")
    if mode in ("full", "related") and related_papers:
        body += format_section(related_papers, "Related Paper")
    body += f"Predicted Followup Paper:\nTitle: {target['title']}\nAbstract: {target['abstract']}\n"
    return f"<example {demo_num}>\n{body.rstrip()}\n</example {demo_num}>"


def expand_one_demo(corpus_id, all_papers_dict, all_embeddings, distance_metric, args):
    target = all_papers_dict[corpus_id]
    key_ref_papers = [all_papers_dict[r["corpus_id"]] for r in target["key_references"]]
    author_papers = select_author_papers(target, all_papers_dict, args.num_author_papers_per_author, args.max_total_author_papers)
    pre_cutoff = {cid: p for cid, p in all_papers_dict.items() if p["date"] < target["date"]}
    index = build_faiss_index(pre_cutoff, all_embeddings, target["date"], distance_metric)
    exclude_ids = {r["corpus_id"] for r in target["key_references"]} | {target["corpus_id"]} | {p["corpus_id"] for p in author_papers}
    related_papers = retrieve_related_papers(target, index, all_embeddings, all_papers_dict, args.num_related_papers, exclude_ids, distance_metric)
    cites, horizon = get_target_impact(target)
    utils.log(f"  {corpus_id}: {len(key_ref_papers)} key refs, {len(author_papers)} author pubs, {len(related_papers)} related papers, impact=({cites},{horizon}mo)")
    return target, key_ref_papers, author_papers, related_papers


def write_prompt_variant(mode, include_impact, base_prompt, expansions, output_dir):
    pre = preamble(base_prompt).replace("Below are a few solved examples for this prediction problem where we provide only one possible followup.", "Below are a few solved examples for this prediction problem where we provide only one possible followup." + build_instruction_block(mode, include_impact))
    demos = "\n\n\n".join(format_demo(i + 1, *exp, mode=mode, include_impact=include_impact) for i, exp in enumerate(expansions))
    post = postamble(base_prompt)
    prompt = pre + demos + post
    suffix = "_impact" if include_impact else ""
    mode_str = mode if mode == "vanilla" else f"expanded_{mode}"
    output_path = f"{output_dir}/prediction_system_{mode_str}{suffix}.prompt"
    with open(output_path, "w") as f:
        f.write(prompt)
    utils.log(f"Wrote {output_path} ({len(prompt)} chars)")


def main():
    parser = argparse.ArgumentParser(description="Regenerate per-ablation few-shot prompt templates from train-split demos")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience")
    parser.add_argument("--embeddings_dir", type=str, default="data/corpus/train")
    parser.add_argument("--embedding_type", type=str, default="grit", choices=["gtr", "grit", "specter2"])
    parser.add_argument("--num_author_papers_per_author", type=int, default=3)
    parser.add_argument("--max_total_author_papers", type=int, default=15)
    parser.add_argument("--num_related_papers", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="task_followup_prediction/templates")
    args = parser.parse_args()

    distance_metric = "l2" if args.embedding_type == "specter2" else "cosine"
    utils.log(f"Loading train split corpus + {args.embedding_type} embeddings")
    all_papers, _, all_embeddings = utils.load_corpus(hf_repo_id=args.hf_repo_id, split="train", embeddings_dir=args.embeddings_dir, embedding_type=args.embedding_type, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    utils.log(f"Expanding {len(DEMO_CORPUS_IDS)} demos")
    expansions = [expand_one_demo(cid, all_papers_dict, all_embeddings, distance_metric, args) for cid in DEMO_CORPUS_IDS]

    base_prompt = load_base_prompt()
    for mode in MODES:
        for include_impact in (False, True):
            write_prompt_variant(mode, include_impact, base_prompt, expansions, args.output_dir)


if __name__ == "__main__":
    main()
