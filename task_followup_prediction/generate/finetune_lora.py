"""LoRA finetuning for followup work prediction with Optuna hyperparameter search."""

import os
import gc
import random
import argparse
import numpy as np
import torch
import optuna
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from tqdm import tqdm

import utils
from task_followup_prediction.dataset import create_evaluation_instances, create_train_val_split, format_instance_for_training
from task_followup_prediction.templates.fewshot_examples import FEWSHOT_EXAMPLES

MODEL_CONFIGS = {
    "olmo3-7b": {"model_name": "allenai/Olmo-3-7B-Instruct", "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]},
    "llama3.1-8b": {"model_name": "meta-llama/Llama-3.1-8B-Instruct", "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]},
}

HP_SEARCH_SPACE = {
    "learning_rate": (1e-5, 5e-4),
    "lora_r": [8, 16, 32, 64],
}  # lora_alpha = 2 * lora_r (fixed ratio), always train for 3 epochs


def load_model_and_tokenizer(model_key, lora_config=None):
    """Load model and tokenizer, optionally applying LoRA."""
    config = MODEL_CONFIGS[model_key]
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_config is not None:
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    return model, tokenizer


def create_lora_config(model_key, lora_r, lora_alpha):
    """Create LoRA configuration for the specified model."""
    config = MODEL_CONFIGS[model_key]
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, target_modules=config["target_modules"])


def tokenize_instance(example, tokenizer, model_key, max_length=4096):
    """Tokenize a single training instance using the model's chat template."""
    formatted = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(formatted, truncation=True, max_length=max_length, padding=False, return_tensors=None)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def create_hf_dataset(instances, all_papers_dict, system_prompt, tokenizer, model_key, fewshot_examples, reasoning_trace):
    """Convert instances to HuggingFace Dataset."""
    formatted_data = [format_instance_for_training(inst, all_papers_dict, system_prompt, fewshot_examples, reasoning_trace) for inst in tqdm(instances, desc="Formatting instances")]
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.map(lambda x: tokenize_instance(x, tokenizer, model_key), remove_columns=["messages", "corpus_id"])
    return dataset


def is_main_process():
    """Check if this is the main process in distributed training."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def is_distributed():
    """Check if running in distributed mode."""
    return "LOCAL_RANK" in os.environ


def sync_hyperparams(lr, lora_r):
    """Synchronize hyperparameters across all processes in distributed training."""
    if not is_distributed():
        return lr, lora_r
    import torch.distributed as dist
    params = torch.tensor([lr, float(lora_r)], device="cuda")
    dist.broadcast(params, src=0)
    return params[0].item(), int(params[1].item())


def objective(trial, train_dataset, val_dataset, model_key, tokenizer, output_dir, batch_size, gradient_accumulation_steps, num_epochs):
    """Optuna objective function for hyperparameter search."""
    if is_main_process():
        lr = trial.suggest_float("learning_rate", HP_SEARCH_SPACE["learning_rate"][0], HP_SEARCH_SPACE["learning_rate"][1], log=True)
        lora_r = trial.suggest_categorical("lora_r", HP_SEARCH_SPACE["lora_r"])
    else:
        lr, lora_r = 0.0, 0
    lr, lora_r = sync_hyperparams(lr, lora_r)
    lora_alpha = 2 * lora_r

    lora_config = create_lora_config(model_key, lora_r, lora_alpha)
    model, _ = load_model_and_tokenizer(model_key, lora_config)

    trial_output_dir = os.path.join(output_dir, f"trial_{trial.number}")
    training_args = TrainingArguments(
        output_dir=trial_output_dir, num_train_epochs=num_epochs, per_device_train_batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr, bf16=True, gradient_checkpointing=True, logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, report_to="none", dataloader_num_workers=4,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)

    trainer.train()
    eval_results = trainer.evaluate()
    val_loss = eval_results["eval_loss"]

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return val_loss


def run_hp_search(train_dataset, val_dataset, model_key, tokenizer, output_dir, n_trials, batch_size, gradient_accumulation_steps, num_epochs):
    """Run Optuna hyperparameter search."""
    study = optuna.create_study(direction="minimize", study_name=f"lora_{model_key}")
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, model_key, tokenizer, output_dir, batch_size, gradient_accumulation_steps, num_epochs), n_trials=n_trials, show_progress_bar=True)

    utils.log(f"Best trial: {study.best_trial.number}")
    utils.log(f"Best params: {study.best_params}")
    utils.log(f"Best val loss: {study.best_value}")

    return study.best_params


def train_final_model(train_dataset, val_dataset, model_key, tokenizer, best_params, output_dir, batch_size, gradient_accumulation_steps, num_epochs):
    """Train final model with best hyperparameters."""
    lora_alpha = 2 * best_params["lora_r"]

    lora_config = create_lora_config(model_key, best_params["lora_r"], lora_alpha)
    model, _ = load_model_and_tokenizer(model_key, lora_config)

    final_output_dir = os.path.join(output_dir, "final")
    training_args = TrainingArguments(
        output_dir=final_output_dir, num_train_epochs=num_epochs, per_device_train_batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=best_params["learning_rate"], bf16=True, gradient_checkpointing=True, logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)

    trainer.train()

    adapter_path = os.path.join(final_output_dir, "adapter")
    model.save_pretrained(adapter_path)
    utils.log(f"Saved adapter to {adapter_path}")

    return adapter_path


def main():
    parser = argparse.ArgumentParser(description="LoRA finetuning for followup prediction")
    parser.add_argument("--input_dir", type=str, default="data/corpus/train", help="Training corpus directory")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/train/lora_models", help="Output directory for models")
    parser.add_argument("--model", type=str, default="llama3.1-8b", choices=list(MODEL_CONFIGS.keys()), help="Model to finetune")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_hp_search", action="store_true", help="Skip HP search and use provided params")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate (if skipping HP search)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (if skipping HP search, alpha = 2*r)")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--reasoning_trace", type=str, default="generic", choices=["generic", "none"], help="Reasoning trace format: 'generic' includes placeholder reasoning, 'none' omits reasoning")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    utils.log(f"Loading data from {args.input_dir}")
    all_papers, metadata = utils.load_json(os.path.join(args.input_dir, "all_papers.json"))
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    system_prompt_path = "task_followup_prediction/templates/prediction_system_local.prompt"
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()

    utils.log("Creating evaluation instances")
    instances = create_evaluation_instances(all_papers, all_papers_dict)
    train_instances, val_instances = create_train_val_split(instances, args.val_ratio, args.seed)
    utils.log(f"Created {len(train_instances)} train, {len(val_instances)} val instances")

    utils.log(f"Loading tokenizer for {args.model}")
    _, tokenizer = load_model_and_tokenizer(args.model, lora_config=None)

    utils.log(f"Creating HuggingFace datasets (reasoning_trace={args.reasoning_trace})")
    train_dataset = create_hf_dataset(train_instances, all_papers_dict, system_prompt, tokenizer, args.model, FEWSHOT_EXAMPLES, args.reasoning_trace)
    val_dataset = create_hf_dataset(val_instances, all_papers_dict, system_prompt, tokenizer, args.model, FEWSHOT_EXAMPLES, args.reasoning_trace)

    model_output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_output_dir, exist_ok=True)

    if args.skip_hp_search:
        best_params = {"learning_rate": args.learning_rate, "lora_r": args.lora_r}
        utils.log(f"Using provided hyperparameters: {best_params} (lora_alpha = 2 * lora_r = {2 * args.lora_r}, epochs = {args.num_train_epochs})")
    else:
        utils.log(f"Running hyperparameter search with {args.n_trials} trials ({args.num_train_epochs} epochs per trial)")
        best_params = run_hp_search(train_dataset, val_dataset, args.model, tokenizer, model_output_dir, args.n_trials, args.batch_size, args.gradient_accumulation_steps, args.num_train_epochs)

    utils.log("Training final model with best hyperparameters")
    adapter_path = train_final_model(train_dataset, val_dataset, args.model, tokenizer, best_params, model_output_dir, args.batch_size, args.gradient_accumulation_steps, args.num_train_epochs)

    result = {"model": args.model, "best_params": best_params, "adapter_path": adapter_path, "train_instances": len(train_instances), "val_instances": len(val_instances)}
    utils.save_json([result], os.path.join(model_output_dir, "training_summary.json"), metadata=utils.update_metadata(metadata, args))
    utils.log("Done")


if __name__ == "__main__":
    main()
