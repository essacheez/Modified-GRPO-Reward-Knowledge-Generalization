<div align="center">

# 🧠 Reward Might Be All You Need For Generalization

**Investigating whether reinforcement learning can help LLMs generalize newly injected knowledge**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg)]()

</div>

---

## 🎯 Overview

Small language models often struggle to generalize from new information.

For example, if you finetune a 3B model on a specific fact and then ask the exact question it was trained on, it may answer correctly. But if you rephrase the question slightly, the model usually fails. The issue is that supervised finetuning typically teaches the model a mapping between specific question patterns and answers. The model may memorize the surface forms of the training data rather than fully internalizing the underlying fact. This limitation becomes evident when testing with held-out rephrasings: a model that performs well on the training distribution can perform much worse on semantically equivalent questions expressed differently.

Consider a simple fact:

> "A 6.2 magnitude earthquake struck Myanmar on March 28, 2025."

A finetuned model might correctly answer:

- "What magnitude earthquake hit Myanmar in March 2025?"

But struggle with variations such as:

- **Linguistic variations:** "How strong was the seismic event in Myanmar during late March 2025?"
- **Structural differences:** "Myanmar experienced an earthquake in March 2025, what was its magnitude?"

All of these refer to the same fact, but the model may treat them differently due to surface-level differences.

This effect is more pronounced in smaller models. While very large models generalize across some variations, smaller models are more likely to overfit to the patterns seen during training. Since smaller models are often the ones used in practical deployments, finding a way around these limitations is important when considering their reliability.

> **Core Hypothesis**: Updating the reward structure to add a group reward for generalization inaddtion to consistency across multiple phrasings of the same question in GRPO will force models to build robust internal representations and enable OOD generalization.

---

## 🔬 The Approach

### Stage 1: Novel Data Collection

We source news from the **GDELT Project** published after April 2025, information models have definitely never seen. We filter for unpredictable events using keywords like _abrupt_, _earthquake_, _unprecedented_, _sinkhole_ to ensure genuine novelty. We collect a total of **3,200 atomic facts** from Yahoo News

### Stage 2: Question Diversification

For each fact, we generate **7 distinct question phrasings** with minimal lexical overlap using GPT-4o mini (Average BLEU: **0.2**, Average Jaccard: **0.31**). This confirms real linguistic diversity, not surface-level rewording.

### Stage 3: Supervised Finetuning

We finetune **Qwen 2.5** (3B and 7B) on 5 of 7 phrasings per fact, holding out 2 for evaluation. This establishes baseline internalization levels.

### Stage 4: GRPO Training

We apply GRPO by geting our reward signal from a lightweight judge model (3B params) and perform it sequentially across three configurations, each building on the SFT models from Stage 3:

**Configuration 1 — Single Rephrasing**

GRPO training using only 1 question phrasing per fact. This tests whether RL alone can push a model toward generalization when given minimal linguistic diversity during the RL phase.

**Configuration 2 — 5 Rephrasings**

GRPO training using all 5 question phrasings per fact. Tests whether exposing the model to diverse phrasings during RL improves generalization beyond what SFT achieved.

**Configuration 3 — Modified Reward Structure**

Our core contribution. Same 5 rephrasings, but we modify GRPO's reward to explicitly add a generalization reward. The higher the correctly answered rephrases in the group by the model the higher the reward

**🚧 Currently in development**

Work on configuration 2 and 3 is currently under progress.

</td>
</tr>
</table>

---

## 📊 Results (Step 3)

Evaluation on **300 held-out facts** with 2 question phrasings each:

<div align="center">

| Model & Setting                            | Both Correct ↑ | One Correct |
| :----------------------------------------- | :------------: | :---------: |
| Qwen 3B Base                               |     0.67%      |    2.00%    |
| **Qwen 3B SFT Questions (5 rephrasings)**  |   **74.00%**   |   24.00%    |
| Qwen 3B SFT Question 1 Repeat (same q × 5) |     10.00%     |   25.33%    |
| Qwen 3B SFT Question No Repeat             |     4.00%      |   16.33%    |
| Qwen 3B SFT Question No Repeat Equal Steps |     7.00%      |   29.00%    |
| Qwen 3B SFT Fact 1 Repeat                  |     4.33%      |   14.67%    |
| Qwen 3B SFT Fact No Repeat                 |     1.67%      |   11.33%    |
| **Qwen 7B SFT Questions**                  |   **81.33%**   |   17.00%    |

</div> <p>

> **Both Correct** = model answered both held-out phrasings correctly (true generalization)  
> **One Correct** = model got exactly one right (partial internalization)

### 💡 Key Insights

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔥 Linguistic variety crushes repetition                                   │
│     → 5 diverse rephrasings: 74% generalization                             │
│     → Same question repeated 5×: only 10%                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  📈 Scale matters                                                           │
│     → 7B model reaches 81% vs 74% for 3B                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ❓ QA format > raw facts                                                   │
│     → Training on atomic facts directly performs poorly                     │
│     → Question-answer structure supports better retention                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### 1. Start the Reference Model Server

```bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model Essacheez/Qwen2.5-3B-RG-SFT-1-Repeat \
    --port 8003 \
    --gpu-memory-utilization 0.30 \
    --max-model-len 512 \
    --dtype bfloat16 \
    --host 0.0.0.0
```

### 2. Run Reward Model

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-3B-Instruct \
    --port 8002 \
    --gpu-memory-utilization 0.30 \
    --max-model-len 2056 \
    --dtype bfloat16
```

### 3. Run GRPO

```bash
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 --mixed_precision=bf16 train.py
```

### 💻 Hardware Requirements

| Task          | GPU Memory |
| :------------ | :--------- |
| GRPO Training | ~100GB+    |
| SFT Training  | ~24GB      |
| Inference     | ~8GB       |

---

## 📁 Project Structure

```
📦 Reward-Generalization/
│
├── 📂 Data/
│   ├── 📂 extracted_and_rephrased_QA/
│   │   ├── qa_pairs.json                    # Original QA pairs
│   │   └── Rephrased_QA.json                # All 7 rephrasings per fact
│   │
│   ├── 📂 extracted_facts/
│   │   ├── atomic_facts_yahoo_abrupt_6m_extracted.csv.json
│   │   ├── atomic_facts_yahoo_collapse_6m.csv.json
│   │   ├── atomic_facts_yahoo_crash_6m.csv.json
│   │   ├── atomic_facts_yahoo_discovery_6m.csv.json
│   │   ├── atomic_facts_yahoo_earthquake_6m.csv.json
│   │   ├── atomic_facts_yahoo_emergency_6m.csv.json
│   │   ├── atomic_facts_yahoo_erupt_6m.csv.json
│   │   ├── atomic_facts_yahoo_outbreak_6m.csv.json
│   │   ├── atomic_facts_yahoo_sinkhole_6m.csv.json
│   │   ├── atomic_facts_yahoo_strike_6m.csv.json
│   │   └── atomic_facts_yahoo_unprecedented_6m.csv.json
│   │
│   └── 📂 extracted_news/
│       ├── combined_sampled.csv
│       └── yahoo_[keyword]_6m.csv           # Raw scraped articles
│
├── 📂 Data Scripts/
│   ├── 📓 Extract_news_from_links.ipynb     # Scraping pipeline
│   ├── 📓 QA_generation_from_news.ipynb     # Fact extraction + QA gen
│   ├── 📓 QA_Rephraser.ipynb                # Generate diverse rephrasings
│   └── 📓 Similarity_calculation.ipynb      # Verify rephrasing diversity
│
├── 📂 GRPO Scripts/
│   ├── 🐍 train.py                          # Main GRPO training loop
│   ├── 🐍 ref_server.py                     # vLLM reference model server
│   ├── 📄 judge_prompt.txt                  # LLM judge prompt template
│   ├── 🐍 checkpoint_handler.py             # Merging FSDP Checkpoints
│   ├── 🐍 merging_sft_model.py              # Merge LoRA → base model
│   ├── 🐍 converter.py                      # Merging FSDP Checkpoints
│   ├── 📄 trl_grpo.sbatch                   # SLURM batch script
│   └── 🐍 upload_to_hf.py                   # Push to Hugging Face Hub
│
├── 📂 SFT Scripts/
│   ├── 🐍 Qwen-2.5-sft-facts.py             # SFT training script
│   └── 🐍 judge.py                          # Evaluation correctness judge
│
└── 📄 requirements.txt
```

## 📚 References

This work builds on:

> **Data Doping or True Intelligence? Evaluating the Transferability of Injected Knowledge in LLMs**  
> E. Jan, M. Ali, M. S. Hassan, F. Zaffar, and Y. Zaki (2025)  
> 📎 [arxiv.org/abs/2505.17140](https://arxiv.org/abs/2505.17140)

> FSDP to PyTorch Checkpoint converstion code was from github I can't find the repo as of now will update as soon as I find it.

---

<div align="center">

## 📬 Contact

Questions? Open an issue or reach out directly.

**Made with ❤️ and mass GPU hours**

</div>
