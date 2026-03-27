"""Multi-GPU local baseline for followup work prediction. Supports vanilla, LoRA, and full-finetuned models."""

import os
import argparse
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from tqdm import tqdm

import utils
from task_followup_prediction.dataset import get_query_papers, create_messages_local, create_messages
from task_followup_prediction.templates.fewshot_examples import FEWSHOT_EXAMPLES

MODEL_CONFIGS = {
    "olmo3-7b": "allenai/Olmo-3-7B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def setup_distributed():
    """Initialize distributed process group if launched via torchrun."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    return rank, world_size


def load_model(model_key, adapter_path=None, model_path=None, device=None):
    """Load model with optional LoRA adapter or from a local checkpoint."""
    device_map = {"": device} if device is not None else "auto"
    model_name = MODEL_CONFIGS[model_key]
    if model_path is not None:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    return model, tokenizer


def clean_response(text):
    """Remove example tags and other artifacts from parsed text."""
    import re
    text = re.sub(r'</example\s*\d*>', '', text)
    return text.strip()


def truncate_at_degenerate_markers(text):
    """Truncate text at degenerate markers that indicate hallucinated content."""
    degenerate_markers = ["Background Paper", "Paper 1:", "Paper 2:", "Example 1:", "Example 2:"]
    for marker in degenerate_markers:
        if marker in text:
            text = text.split(marker)[0].strip()
    return text


def parse_response(answer):
    """Parse model response into reasoning, title, abstract."""
    try:
        reasoning = answer.split("Reasoning:")[1].split("Title:")[0].strip()
        title = answer.split("Title:")[1].split("Abstract:")[0].strip()
        abstract = answer.split("Abstract:")[1].strip()
        abstract = truncate_at_degenerate_markers(abstract)
        return {"reasoning": clean_response(reasoning), "title": clean_response(title), "abstract": clean_response(abstract), "raw_response": answer}
    except Exception:
        if "Title:" in answer and "Abstract:" in answer:
            title = answer.split("Title:")[1].split("Abstract:")[0].strip()
            abstract = answer.split("Abstract:")[1].strip()
            abstract = truncate_at_degenerate_markers(abstract)
            return {"reasoning": "", "title": clean_response(title), "abstract": clean_response(abstract), "raw_response": answer}
        utils.log(f"Error parsing response")
        print(f"Raw response: {answer}")
        return {"reasoning": "", "title": "", "abstract": "", "raw_response": answer}


def generate_batch(model, tokenizer, batch_messages, model_key, max_new_tokens=1000):
    """Generate responses for a batch of message lists."""
    texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in batch_messages]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)

    stop_strings = ["Background Paper", "Paper 1:", "Paper 2:", "\n\n\n\n"]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, stop_strings=stop_strings, tokenizer=tokenizer)

    results = []
    for i, output in enumerate(outputs):
        response_ids = output[len(inputs.input_ids[i]):]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        results.append(parse_response(response_text))
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU local baseline for followup prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/generations", help="Output directory")
    parser.add_argument("--model", type=str, default="llama3.1-8b", choices=list(MODEL_CONFIGS.keys()), help="Model to use")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (None for vanilla)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to local model checkpoint (overrides --model for loading)")
    parser.add_argument("--max_query_papers", type=int, default=5000, help="Maximum papers to evaluate")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Maximum new tokens to generate")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N samples")
    args = parser.parse_args()

    rank, world_size = setup_distributed()
    is_main = (rank == 0)

    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    query_papers = get_query_papers(all_papers, max_papers=args.max_query_papers)

    if args.model_path is not None:
        system_prompt_path = "task_followup_prediction/templates/prediction_system_finetune.prompt"
    else:
        system_prompt_path = "task_followup_prediction/templates/prediction_system_local.prompt"
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()

    # Shard query papers across GPUs (interleaved for balanced distribution)
    query_papers_local = query_papers[rank::world_size]
    if is_main:
        utils.log(f"Evaluating {len(query_papers)} papers across {world_size} GPU(s) ({len(query_papers_local)} per GPU), batch_size={args.batch_size}")
        utils.log(f"Using system prompt: {system_prompt_path}")

    device = f"cuda:{rank}" if world_size > 1 else None
    if is_main:
        utils.log(f"Loading model {args.model}")
    model, tokenizer = load_model(args.model, args.adapter_path, args.model_path, device=device)

    if is_main:
        mode = "finetuned" if args.model_path else ("lora" if args.adapter_path else "vanilla")
        output_filename = f"generations.{args.model}.{mode}.json"
        output_path = os.path.join(args.output_dir, output_filename)
        os.makedirs(args.output_dir, exist_ok=True)
        utils.log("Running inference")

    next_checkpoint = args.save_every
    for i in tqdm(range(0, len(query_papers_local), args.batch_size), desc="Generating", disable=not is_main):
        batch_records = query_papers_local[i:i + args.batch_size]
        if args.model_path is not None:
            batch_messages = [create_messages_local(r, system_prompt, all_papers_dict, [], reasoning_trace="none") for r in batch_records]
        else:
            batch_messages = [create_messages_local(r, system_prompt, all_papers_dict, FEWSHOT_EXAMPLES) for r in batch_records]
        results = generate_batch(model, tokenizer, batch_messages, args.model, args.max_new_tokens)

        for record, result in zip(batch_records, results):
            record["title"] = result["title"]
            record["abstract"] = result["abstract"]
            record["reasoning"] = result["reasoning"]
            record["raw_response"] = result["raw_response"]

        # Intermediate checkpoints only in single-GPU mode (multi-GPU saves after gather)
        if world_size == 1:
            processed = i + len(batch_records)
            if processed >= next_checkpoint:
                utils.save_json(query_papers, output_path, metadata=utils.update_metadata([], args), overwrite=True)
                utils.log(f"Checkpoint: saved {processed} predictions")
                next_checkpoint += args.save_every

    # Gather results from all GPUs
    if world_size > 1:
        all_shards = [None] * world_size
        dist.all_gather_object(all_shards, query_papers_local)
        if is_main:
            # Interleave shards back to original order (reverse of rank::world_size sharding)
            query_papers = []
            max_shard_len = max(len(s) for s in all_shards)
            for i in range(max_shard_len):
                for shard in all_shards:
                    if i < len(shard):
                        query_papers.append(shard[i])

    if is_main:
        utils.save_json(query_papers, output_path, metadata=utils.update_metadata([], args), overwrite=True)
        utils.log(f"Saved {len(query_papers)} predictions to {output_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
