"""Local baseline for followup work prediction. Supports vanilla and LoRA-finetuned models."""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

import utils
from task_followup_prediction.dataset import get_query_papers, create_messages_local
from task_followup_prediction.templates.fewshot_examples import FEWSHOT_EXAMPLES

MODEL_CONFIGS = {
    "olmo3-7b": "allenai/Olmo-3-7B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def load_model(model_key, adapter_path=None):
    """Load model with optional LoRA adapter."""
    model_name = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if adapter_path is not None:
        utils.log(f"Loading LoRA adapter from {adapter_path}")
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
    parser = argparse.ArgumentParser(description="Local baseline for followup prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/generations", help="Output directory")
    parser.add_argument("--model", type=str, default="llama3.1-8b", choices=list(MODEL_CONFIGS.keys()), help="Model to use")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (None for vanilla)")
    parser.add_argument("--max_query_papers", type=int, default=5000, help="Maximum papers to evaluate")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Maximum new tokens to generate")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N samples")
    args = parser.parse_args()

    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    query_papers = get_query_papers(all_papers, max_papers=args.max_query_papers)

    with open("task_followup_prediction/templates/prediction_system_local.prompt", "r") as f:
        system_prompt = f.read()

    utils.log(f"Evaluating {len(query_papers)} papers with batch_size={args.batch_size}")
    utils.log(f"Using {len(FEWSHOT_EXAMPLES)} few-shot examples in multi-turn format")

    utils.log(f"Loading model {args.model}")
    model, tokenizer = load_model(args.model, args.adapter_path)

    mode = "lora" if args.adapter_path else "vanilla"
    output_filename = f"generations.{args.model}.{mode}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    os.makedirs(args.output_dir, exist_ok=True)

    utils.log("Running inference")
    next_checkpoint = args.save_every
    for i in tqdm(range(0, len(query_papers), args.batch_size), desc="Generating"):
        batch_records = query_papers[i:i + args.batch_size]
        batch_messages = [create_messages_local(r, system_prompt, all_papers_dict, FEWSHOT_EXAMPLES) for r in batch_records]
        results = generate_batch(model, tokenizer, batch_messages, args.model, args.max_new_tokens)

        for record, result in zip(batch_records, results):
            record["title"] = result["title"]
            record["abstract"] = result["abstract"]
            record["reasoning"] = result["reasoning"]
            record["raw_response"] = result["raw_response"]

        processed = i + len(batch_records)
        if processed >= next_checkpoint:
            utils.save_json(query_papers, output_path, metadata=utils.update_metadata([], args), overwrite=True)
            utils.log(f"Checkpoint: saved {processed} predictions")
            next_checkpoint += args.save_every

    utils.save_json(query_papers, output_path, metadata=utils.update_metadata([], args), overwrite=True)
    utils.log(f"Saved {len(query_papers)} predictions to {output_path}")


if __name__ == "__main__":
    main()
