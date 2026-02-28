"""Full parameter finetuning for followup work prediction with FSDP and wandb logging."""

import os
import random
import argparse
import numpy as np
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
from tqdm import tqdm

import utils
from task_followup_prediction.dataset import create_evaluation_instances, create_train_val_split, format_instance_for_training

MODEL_CONFIGS = {
    "olmo3-7b": {"model_name": "allenai/Olmo-3-7B-Instruct", "fsdp_layer_cls": "Olmo3DecoderLayer"},
    "llama3.1-8b": {"model_name": "meta-llama/Llama-3.1-8B-Instruct", "fsdp_layer_cls": "LlamaDecoderLayer"},
}


def load_model_and_tokenizer(model_key):
    """Load model and tokenizer for full finetuning."""
    config = MODEL_CONFIGS[model_key]
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def tokenize_instance(example, tokenizer, model_key, max_length=4096):
    """Tokenize a single training instance with completion-only label masking."""
    formatted = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(formatted, truncation=True, max_length=max_length, padding=False, return_tensors=None)

    # Tokenize prompt only (without assistant response) to find where completion starts
    prompt_formatted = tokenizer.apply_chat_template(example["messages"][:-1], tokenize=False, add_generation_prompt=True)
    prompt_length = len(tokenizer(prompt_formatted, truncation=True, max_length=max_length, padding=False, return_tensors=None)["input_ids"])

    # Mask prompt tokens in labels so loss is only on the completion
    labels = tokenized["input_ids"].copy()
    labels[:prompt_length] = [-100] * prompt_length
    tokenized["labels"] = labels
    return tokenized


def create_hf_dataset(instances, all_papers_dict, system_prompt, tokenizer, model_key, fewshot_examples, reasoning_trace):
    """Convert instances to HuggingFace Dataset."""
    formatted_data = [format_instance_for_training(inst, all_papers_dict, system_prompt, fewshot_examples, reasoning_trace) for inst in tqdm(instances, desc="Formatting instances")]
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.map(lambda x: tokenize_instance(x, tokenizer, model_key), remove_columns=["messages", "corpus_id"])
    return dataset


def get_fsdp_config(model_key):
    """Return FSDP TrainingArguments kwargs if multiple GPUs available."""
    if torch.cuda.device_count() <= 1:
        return {}
    return {
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": [MODEL_CONFIGS[model_key]["fsdp_layer_cls"]],
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
            "limit_all_gathers": True,
            "use_orig_params": True,
            "sync_module_states": True,
            "cpu_ram_efficient_loading": True,
            "activation_checkpointing": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Full parameter finetuning for followup prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace dataset repo ID")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/train/full_models", help="Output directory for models")
    parser.add_argument("--model", type=str, default="llama3.1-8b", choices=list(MODEL_CONFIGS.keys()), help="Model to finetune")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps (effective batch = batch_size * grad_accum * num_gpus)")
    parser.add_argument("--reasoning_trace", type=str, default="none", choices=["generic", "none"], help="Reasoning trace format: 'none' omits reasoning (recommended), 'generic' includes placeholder reasoning")
    parser.add_argument("--wandb_project", type=str, default="prescience-finetune", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name (auto-generated if not provided)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    utils.log(f"Loading data from {args.hf_repo_id} ({args.split})")
    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    system_prompt_path = "task_followup_prediction/templates/prediction_system_finetune.prompt"
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()

    utils.log("Creating evaluation instances")
    instances = create_evaluation_instances(all_papers, all_papers_dict)
    train_instances, val_instances = create_train_val_split(instances, args.val_ratio, args.seed)
    utils.log(f"Created {len(train_instances)} train, {len(val_instances)} val instances")

    utils.log(f"Loading tokenizer for {args.model}")
    _, tokenizer = load_model_and_tokenizer(args.model)

    utils.log(f"Creating HuggingFace datasets (reasoning_trace={args.reasoning_trace}, no few-shot demonstrations)")
    train_dataset = create_hf_dataset(train_instances, all_papers_dict, system_prompt, tokenizer, args.model, [], args.reasoning_trace)
    val_dataset = create_hf_dataset(val_instances, all_papers_dict, system_prompt, tokenizer, args.model, [], args.reasoning_trace)

    model_output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_output_dir, exist_ok=True)

    # Initialize wandb on main process
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        run_name = args.wandb_run_name if args.wandb_run_name else f"{args.model}_lr{args.learning_rate}_wd{args.weight_decay}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    num_gpus = torch.cuda.device_count()
    fsdp_kwargs = get_fsdp_config(args.model)
    use_fsdp = len(fsdp_kwargs) > 0
    utils.log(f"Detected {num_gpus} GPU(s), {'using FSDP' if use_fsdp else 'single-GPU mode'}")

    utils.log(f"Loading model {args.model} for full finetuning")
    model, _ = load_model_and_tokenizer(args.model)
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=model_output_dir, num_train_epochs=args.num_train_epochs, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, bf16=True,
        gradient_checkpointing=not use_fsdp, gradient_checkpointing_kwargs={"use_reentrant": False} if not use_fsdp else None,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, report_to="wandb",
        **fsdp_kwargs,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)

    trainer.train()

    model_path = os.path.join(model_output_dir, "model")
    if use_fsdp:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    utils.log(f"Saved full model to {model_path}")

    result = {"model": args.model, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "model_path": model_path, "train_instances": len(train_instances), "val_instances": len(val_instances)}
    utils.save_json([result], os.path.join(model_output_dir, "training_summary.json"), metadata=utils.update_metadata([], args))

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.finish()
    utils.log("Done")


if __name__ == "__main__":
    main()
