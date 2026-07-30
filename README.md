# Qwen2.5-1.5B LLM Instruction Fine-Tuning (PEFT) Pipeline

An end-to-end, production-grade pipeline to parameter-efficiently fine-tune **Qwen2.5-1.5B-Instruct** using Low-Rank Adaptation (LoRA) on the Alpaca dataset. This repository contains the complete engineering workflow, including data preprocessing, stable memory-optimized training, local inference validation, and final adapter weight fusion.

---

## 🚀 Model & Weights Deployment
The fine-tuned adapter weights have been successfully merged into the base model layers to create a unified, standalone checkpoint. The production-ready model is hosted on the Hugging Face Hub:

👉 **[Access the Fused Model Checkpoint on Hugging Face]([Your Hugging Face URL Go Here])**

---

## 🛠️ Project Architecture & Layout
This project is engineered to run deterministically and safely on a single local GPU setup by mitigating common Windows multi-processing and out-of-memory overheads.

* `finetuning.py`: The core training pipeline utilizing Hugging Face's `SFTTrainer` configured for standard half-precision (`Float16`) matrix tracking.
* `merge_model.py`: An optimization script that mathematically fuses the learned LoRA delta weights permanently back into the underlying base model layers.
* `inference.py`: A local testing module to load the checkpoints onto the GPU and validate instruction-following performance.
* `requirements.txt`: Pins specific, cross-compatible library dependencies to prevent underlying CUDA driver or memory allocation faults.

---

## 📊 Training Pipeline Specifications

| Hyperparameter / Component | Configuration Value |
| :--- | :--- |
| **Base Model Architecture** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Fine-Tuning Method** | LoRA (Low-Rank Adaptation) |
| **LoRA Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **LoRA Config** | Rank ($r$) = `16`, Alpha ($\alpha$) = `32`, Dropout = `0.05` |
| **Dataset** | Stanford Alpaca (`tatsu-lab/alpaca` - 2,000 tokenized subset pairs) |
| **Precision Mode** | Native `Float16` |
| **Effective Batch Size** | `16` (Per Device Train Batch: `1` × Gradient Accumulation Steps: `16`) |
| **Optimizer** | `adamw_torch` (Native PyTorch Engine) |

---

## ⚙️ Local Environment Replication

To stand up this pipeline locally and guarantee compilation safety without hitting common dependency version collisions, establish a Python virtual environment and run:

```bash
# Force the download of CUDA-enabled PyTorch binaries
pip install -r requirements.txt --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
