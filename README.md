<div align="center">

# PreScience

**A Benchmark for Forecasting Scientific Contributions**

[Paper](https://arxiv.org/abs/TODO) | [Dataset](https://huggingface.co/datasets/allenai/prescience)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

</div>

## Overview

Can AI systems trained on the scientific record up to a fixed point in time forecast the scientific advances that follow? PreScience decomposes the research process into four interdependent generative tasks and evaluates models on a curated dataset of 98,000 AI-related arXiv papers (Oct 2023 -- Oct 2025) with disambiguated author identities, temporally aligned scholarly metadata, and a structured graph of 502,000 total papers. We decompose the prediction of a single scientific advance into the following four interdependent prediction problems:

<img width="721" height="326" alt="image" src="https://github.com/user-attachments/assets/1698f74b-fd49-4312-a39c-5429b362a3ed" />


| Task | Description | Metrics |
|------|-------------|---------|
| Collaborator Prediction | Predict the remaining authors on a future paper given a seed author | nDCG, R-Precision |
| Prior Work Selection | Predict the key references of a future paper given its authors | nDCG, R-Precision |
| Contribution Generation | Generate a paper's title and abstract given its authors and key references | LACERScore, ROUGE-L, BERTScore |
| Impact Prediction | Predict a paper's 12-month cumulative citation count | MAE, Pearson, Spearman |

## Dataset

The dataset is hosted on [HuggingFace](https://huggingface.co/datasets/allenai/prescience) and contains 98k target papers across train (Oct 2023--2024) and test (Oct 2024--2025) splits, along with 400k+ companion papers (references and author publication histories). See the [dataset card](https://huggingface.co/datasets/allenai/prescience) for full schema and statistics.

## Setup

```bash
git clone https://github.com/allenai/prescience.git
cd prescience
pip install -r requirements.txt
```

**Prerequisites:**
- Python 3.10+
- GPU with 24 GB+ VRAM for GritLM-7B embeddings and 8B model fine-tuning (BF16, no quantization); CUDA 12 required (`faiss-gpu-cu12`)
- Embedding-based task baselines and `multiturn/simulate.py` require precomputed embeddings (see [Computing Embeddings](#computing-embeddings))
- `export OPENAI_API_KEY=<your_key>` — required for GPT baselines, LACERScore evaluation, and gold paraphrase generation
- `export ANTHROPIC_API_KEY=<your_key>` — required for Claude baselines
- SPECTER2 and Qwen3 have dependency conflicts with the main environment; use `requirements_specter2.txt` and `requirements_qwen3.txt` in separate environments


## Codebase Structure

```
prescience/
├── task_coauthor_prediction/     # Collaborator prediction baselines, evaluation and analyses
├── task_priorwork_prediction/    # Prior work selection baselines, evaluation and analyses
├── task_followup_prediction/
│   ├── generate/                 # Contribution generation baselines (GPT, Claude, OLMo, LLaMA, Qwen)
│   └── evaluate/                 # LACERScore, BERTScore, ROUGE-L evaluation
│   └── analysis/
├── task_impact_prediction/       # Impact prediction baselines, evaluation and analyses
├── multiturn/
│   ├── simulate.py               # End-to-end multi-turn simulation
│   └── analysis/                 # Simulation analysis scripts
├── dataset/
│   ├── corpus/                   # 7-stage corpus creation pipeline
│   ├── embeddings/               # Embedding computation
│   └── s2and_prep/               # S2AND author disambiguation features
├── utils.py                      # Shared utilities (I/O, embeddings, FAISS, S2 API)
└── requirements.txt
```


## Running Baselines

All baseline scripts save predictions to `data/<task_dir>/<split>/predictions/` and generation scripts save to `data/task_followup_prediction/<split>/generations/`, which are the inputs to the evaluation scripts.

### Collaborator Prediction

```bash
# Frequency
python3 -m task_coauthor_prediction.baseline_frequency --split test

# Rank Fusion (requires precomputed embeddings)
python3 -m task_coauthor_prediction.baseline_rank_fusion --split test --embeddings_dir data/corpus/test --embedding_type grit

# Embedding Fusion
python3 -m task_coauthor_prediction.baseline_embedding_fusion --split test --embeddings_dir data/corpus/test --embedding_type grit

# Hierarchical Clustering
python3 -m task_coauthor_prediction.baseline_hierarchical --split test --embeddings_dir data/corpus/test --embedding_type grit

# Embedding Fusion Projected (train projection, then run inference)
python3 -m task_coauthor_prediction.train_projection --split train --embeddings_dir data/corpus/train --embedding_type grit --output_dir data/task_coauthor_prediction/train/projection_models
python3 -m task_coauthor_prediction.baseline_mean_pooling_projected --split test --embeddings_dir data/corpus/test --embedding_type grit --checkpoint data/task_coauthor_prediction/train/projection_models/projection.grit.pt

# Evaluate
python3 -m task_coauthor_prediction.evaluate --predictions_path data/task_coauthor_prediction/test/predictions/predictions.frequency.one_shot.first.json
```

Table 2 results are reported across `--embedding_type` values: `gtr`, `specter2`, `grit`. The `--seed_author_type` flag defaults to `first` (matching Table 2); other options (`last`, `random`, `highest_h_index`) are explored in Appendix C.1.

### Prior Work Selection

```bash
# Frequency
python3 -m task_priorwork_prediction.baseline_frequency --split test

# Embedding Fusion (mean pool of authored papers; "Emb. Fusion" in Table 2)
python3 -m task_priorwork_prediction.baseline_mean_pooling --split test --embeddings_dir data/corpus/test --embedding_type grit

# Embedding Fusion Refs (mean pool of cited references; "Emb. Fusion Refs" in Table 2)
python3 -m task_priorwork_prediction.baseline_embedding_fusion --split test --embeddings_dir data/corpus/test --embedding_type grit

# Rank Fusion
python3 -m task_priorwork_prediction.baseline_rank_fusion --split test --embeddings_dir data/corpus/test --embedding_type grit

# Hierarchical Clustering
python3 -m task_priorwork_prediction.baseline_hierarchical --split test --embeddings_dir data/corpus/test --embedding_type grit

# Embedding Fusion Projected (train projection, then run inference)
python3 -m task_priorwork_prediction.train_projection --split train --embeddings_dir data/corpus/train --embedding_type grit --output_dir data/task_priorwork_prediction/train/projection_models
python3 -m task_priorwork_prediction.baseline_mean_pooling_projected --split test --embeddings_dir data/corpus/test --embedding_type grit --checkpoint data/task_priorwork_prediction/train/projection_models/projection.grit.pt

# Evaluate
python3 -m task_priorwork_prediction.evaluate --predictions_path data/task_priorwork_prediction/test/predictions/predictions.frequency.json
```

Table 2 results are reported across `--embedding_type` values: `gtr`, `specter2`, `grit`.

### Contribution Generation

```bash
# GPT models (requires OPENAI_API_KEY)
python3 -m task_followup_prediction.generate.baseline_gpt_parallel --split test --model gpt-4o-2024-11-20

# Claude models (requires ANTHROPIC_API_KEY)
python3 -m task_followup_prediction.generate.baseline_claude_parallel --split test --model claude-sonnet-4-5-20250929

# Local models — vanilla
python3 -m task_followup_prediction.generate.baseline_local --split test --model llama3.1-8b

# Local models — fine-tuned (LoRA)
python3 -m task_followup_prediction.generate.baseline_local --split test --model llama3.1-8b --adapter_path data/task_followup_prediction/train/lora_models/llama3.1-8b/final/adapter

# Same-topic baseline
python3 -m task_followup_prediction.generate.baseline_same_topic --split test

# Key reference baseline
python3 -m task_followup_prediction.generate.baseline_key_reference --split test

# Gold paraphrase (requires OPENAI_API_KEY)
python3 -m task_followup_prediction.generate.baseline_paraphrased_gold --split test --model gpt-5-2025-08-07
```

**Model name mapping** (paper name → `--model` value):

| Paper Name | `--model` | Script |
|---|---|---|
| GPT 4o | `gpt-4o-2024-11-20` | `baseline_gpt_parallel` |
| GPT 4.1 | `gpt-4.1-2025-04-14` | `baseline_gpt_parallel` |
| GPT o3 | `o3-2025-04-16` | `baseline_gpt_parallel` |
| GPT 5 | `gpt-5-2025-08-07` | `baseline_gpt_parallel` |
| GPT 5.1 | `gpt-5.1-chat-latest` | `baseline_gpt_parallel` |
| GPT 5.2 | `gpt-5.2-2025-12-11` | `baseline_gpt_parallel` |
| Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | `baseline_claude_parallel` |
| Claude Opus 4.5 | `claude-opus-4-5-20251101` | `baseline_claude_parallel` |
| LLaMA 3.1 8B | `llama3.1-8b` | `baseline_local` |
| OLMo 3 7B | `olmo3-7b` | `baseline_local` |

Table 3 reports only fine-tuned (FT) results for LLaMA and OLMo.

**LoRA fine-tuning:**

```bash
# Hyperparameter search + final training
python3 -m task_followup_prediction.generate.finetune_lora --input_dir data/corpus/train --output_dir data/task_followup_prediction/train/lora_models --model llama3.1-8b --n_trials 20 --val_ratio 0.15 --num_train_epochs 3

# Skip HP search with known-good params
python3 -m task_followup_prediction.generate.finetune_lora --input_dir data/corpus/train --output_dir data/task_followup_prediction/train/lora_models --model llama3.1-8b --skip_hp_search --learning_rate 2e-4 --lora_r 16
```

**Evaluate:**

```bash
# LACERScore (default judge: gpt-5-2025-08-07; also supports Anthropic via --judge anthropic)
python3 -m task_followup_prediction.evaluate.lacer_score --input_path <generations_file>

# BERTScore
python3 -m task_followup_prediction.evaluate.bert_score --input_path <generations_file>

# ROUGE-L
python3 -m task_followup_prediction.evaluate.rouge_score --input_path <generations_file>
```

### Impact Prediction

**Feature-to-flag mapping** (for understanding Table 4 feature set combinations):

| Paper Feature Set | CLI Flags |
|---|---|
| Target Text | `--use_followup_work_paper` |
| Context Text | `--use_author_papers --use_prior_work_papers` |
| Bibliometrics | `--use_author_numbers --use_prior_work_numbers` |

```bash
# Ridge — Target Text (GRIT)
python3 -m task_impact_prediction.baseline_embedding_ridge --split test --train_embeddings_dir data/corpus/train --test_embeddings_dir data/corpus/test --embedding_type grit --use_followup_work_paper

# XGBoost — Target Text only (GRIT)
python3 -m task_impact_prediction.baseline_xgboost_regressor --split test --train_embeddings_dir data/corpus/train --test_embeddings_dir data/corpus/test --embedding_type grit --use_followup_work_paper

# XGBoost — Target + Context Text
python3 -m task_impact_prediction.baseline_xgboost_regressor --split test --train_embeddings_dir data/corpus/train --test_embeddings_dir data/corpus/test --embedding_type grit --use_followup_work_paper --use_author_papers --use_prior_work_papers

# XGBoost — Target + Context + Bibliometrics (full model)
python3 -m task_impact_prediction.baseline_xgboost_regressor --split test --train_embeddings_dir data/corpus/train --test_embeddings_dir data/corpus/test --embedding_type grit --use_followup_work_paper --use_author_papers --use_prior_work_papers --use_author_numbers --use_prior_work_numbers

# XGBoost — Bibliometrics only
python3 -m task_impact_prediction.baseline_xgboost_regressor --split test --train_embeddings_dir data/corpus/train --test_embeddings_dir data/corpus/test --embedding_type grit --use_author_numbers --use_prior_work_numbers

# Evaluate
python3 -m task_impact_prediction.evaluate --predictions_path data/task_impact_prediction/test/predictions/predictions.ridge_grit.json
```

Replace `grit` with `gtr` or `specter2` to reproduce other embedding rows in Table 4.


## Computing Embeddings

Embedding-based baselines require precomputed embeddings. Compute them for each split you need:

```bash
python3 -m dataset.embeddings.compute_paper_embeddings --split test --embedding_type grit --output_dir data/corpus/test
python3 -m dataset.embeddings.compute_paper_embeddings --split train --embedding_type grit --output_dir data/corpus/train
```

| Type | Model | Dimension |
|------|-------|-----------|
| `gtr` | sentence-transformers/gtr-t5-large | 768 |
| `specter2` | allenai/specter2_base | 768 |
| `grit` | GritLM/GritLM-7B | 4096 |

SPECTER2 embeddings require a separate environment; install with `pip install -r requirements_specter2.txt`.


## Multi-turn Simulation

The simulation composes collaborator prediction, prior work selection, and contribution generation into a pipeline that generates a synthetic corpus day-by-day over a specified time horizon.

```bash
python3 -m multiturn.simulate --calibration_split train --embeddings_dir data/corpus/train \
  --author_embedding_type grit --paper_embedding_type grit \
  --coauthor_baseline embedding_fusion --priorwork_baseline embedding_fusion \
  --generation_backend openai --generation_model gpt-5-2025-08-07 \
  --output_dir data/multiturn/simulated --depth 365 --seed 42
```

Key arguments:
- `--coauthor_baseline`: `frequency`, `embedding_fusion`, `rank_fusion`
- `--priorwork_baseline`: `frequency`, `embedding_fusion`, `rank_fusion`
- `--generation_backend`: `openai`, `anthropic`
- `--generation_model`: any supported model name (e.g., `gpt-5-2025-08-07`)
- `--seed`: random seed for reproducibility

The paper uses `embedding_fusion` for both collaborator and prior work prediction with GPT-5 for generation, runs the simulation 6 times with different `--seed` values, and reports 95% confidence intervals.

**Analysis.** The analysis pipeline has two stages: compute metrics, then plot. For example, to analyze author diversity:

```bash
# Compute diversity for natural and synthetic corpora
python3 -m multiturn.analysis.compute_author_diversity --compute_on natural --role target --output_path data/multiturn/analysis/author_diversity_natural.json
python3 -m multiturn.analysis.compute_author_diversity --compute_on synthetic --synthetic_dir simulated --role synthetic --output_path data/multiturn/analysis/author_diversity_synthetic.json

# Plot
python3 -m multiturn.analysis.plot_author_diversity --natural_path data/multiturn/analysis/author_diversity_natural.json --synthetic_paths data/multiturn/analysis/author_diversity_synthetic.json
```

Other available analysis scripts follow the same compute→plot pattern:

| Analysis | Compute Script | Plot Script |
|----------|---------------|-------------|
| Author diversity | `compute_author_diversity` | `plot_author_diversity` |
| Key reference diversity | `compute_key_references_diversity` | `plot_key_references_diversity` |
| LACER diversity | `compute_diversity_neighbors` → `compute_lacer_diversity` | `plot_lacer_diversity` |
| LACER novelty | `compute_novelty_neighbors` → `compute_lacer_novelty` | `plot_lacer_novelty` |
| Topic distribution | `classify_synthetic_primary_categories` | `plot_topic_distribution` |


## Reproducing Paper Results

| Paper Reference | Tasks / Section | How to Reproduce |
|---|---|---|
| Table 2 | Collaborator Prediction + Prior Work Selection | Run all embedding baselines (Frequency, Rank Fusion, Emb. Fusion, Hier. Clustering, Projection) × 3 embedding types (`gtr`, `specter2`, `grit`) |
| Table 3 | Contribution Generation | Run all LLM baselines (see model mapping table) and evaluate with LACERScore, BERTScore, and ROUGE-L |
| Table 4 | Impact Prediction | Run XGBoost with feature combinations (Target Text, Context Text, Bibliometrics) × 3 embedding types |
| Figure 4 | Multi-turn Simulation | Run `multiturn.simulate` with `--coauthor_baseline embedding_fusion --priorwork_baseline embedding_fusion --generation_model gpt-5-2025-08-07`, 6 seeds |


## Building the Dataset from Scratch

Most users should use the [HuggingFace dataset](https://huggingface.co/datasets/allenai/prescience) directly. The dataset we released was built using internal versions of the Semantic Scholar API for efficiency; the scripts below use the equivalent public APIs, but results may vary slightly.

To rebuild from scratch, you need:
- **S2 API key**: Get one at https://www.semanticscholar.org/product/api#api-key and set `export S2_API_KEY=<your_key>`
- **arXiv snapshot**: Download from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) to `data/arxiv_snapshot/arxiv-metadata-oai-snapshot.json`
- **S2AND**: For author disambiguation (Stage 4), follow [S2AND setup](https://github.com/allenai/S2AND)

<details>
<summary><b>Full pipeline commands</b></summary>

Run each stage for both train and test splits. Commands below show the train split; for test, replace `2023-10-01`/`2024-10-01` with `2024-10-01`/`2025-10-01` and `train` with `test`.

```bash
# Stage 1: Download target papers from arXiv snapshot + S2 API
python3 -m dataset.corpus.download_target_papers --start_date 2023-10-01 --end_date 2024-10-01 --output_dir data/corpus/train

# Stage 2: Add key (highly influential) references
python3 -m dataset.corpus.add_key_references --input_dir data/corpus/train --output_dir data/corpus/train

# Stage 3: Add author rosters and publication histories
python3 -m dataset.corpus.add_authors --input_dir data/corpus/train --output_dir data/corpus/train

# Stage 4a: Prepare S2AND input features
python3 -m dataset.s2and_prep.download_s2and_features --input_dirs data/corpus/train data/corpus/test --output_dir data/corpus/s2and_prescience

# Stage 4b: Run S2AND clustering (see https://github.com/allenai/S2AND)

# Stage 4c: Merge disambiguated author identities into corpus
python3 -m dataset.corpus.merge_authors_in_corpus --input_dir data/corpus/train --output_dir data/corpus/train --s2and_data_dir data/corpus/s2and_prescience

# Stage 5: Add citation metadata (citation counts, h-index, trajectories)
python3 -m dataset.corpus.add_citation_metadata --input_dir data/corpus/train --output_dir data/corpus/train

# Stage 6: Replace titles/abstracts with official arXiv versions
python3 -m dataset.corpus.replace_title_abstracts_using_snapshot --input_dir data/corpus/train --output_dir data/corpus/train

# Stage 7: Compute embeddings
python3 -m dataset.embeddings.compute_paper_embeddings --split train --embedding_type gtr --output_dir data/corpus/train
python3 -m dataset.embeddings.compute_paper_embeddings --split train --embedding_type specter2 --output_dir data/corpus/train
python3 -m dataset.embeddings.compute_paper_embeddings --split train --embedding_type grit --output_dir data/corpus/train
```

</details>


## Citation

```bibtex
@article{prescience2026,
  title={PreScience: A Benchmark for Forecasting Scientific Contributions},
  author={Anirudh Ajith and Amanpreet Singh and Jay DeYoung and Nadav Kunievsky and Austin C. Kozlowski and Oyvind Tafjord and James Evans and Daniel S. Weld and Tom Hope and Doug Downey},
  journal={arXiv preprint arXiv:TODO},
  year={2026}
}
```


## License

Code is licensed under Apache 2.0 — see [LICENSE](LICENSE) for details.

The [dataset](https://huggingface.co/datasets/allenai/prescience) is released under [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1.0/).

We welcome bug fixes and improvements — please submit a pull request. For questions or suggestions, open an issue on [GitHub](https://github.com/allenai/prescience/issues). For other inquiries, contact anirudha@allenai.org.
