# Embeddings and Data Preparation for LLMs

This repository implements the core concepts from **Chapter 2** of *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

The objective is to reproduce and understand the data preparation pipeline required to train a Large Language Model (LLM), focusing on tokenization, training sample generation, and embeddings.

---

## 📚 Assignment Overview

The notebook follows the official guide and uses only:

- `ch02.ipynb` (reference implementation)
- `the-verdict.txt` (training text)

The notebook runs end-to-end using:

- `torch`
- `tiktoken`

---

## ⚙️ What Is Implemented

### 1. Text Loading
The raw text is loaded and inspected before processing.

### 2. Byte Pair Encoding (BPE) Tokenization
We use the GPT-2 tokenizer via `tiktoken` to:
- Convert text into subword tokens
- Generate token IDs

### 3. Next-Token Prediction Data Preparation
A sliding window approach is used to generate input-target pairs:
- Input sequence of length `max_length`
- Target sequence shifted by one token

### 4. PyTorch Dataset and DataLoader
The token sequences are wrapped in a custom `Dataset` class to:
- Enable batching
- Prepare for training loops
- Ensure compatibility with neural network pipelines

---

## 🧠 Conceptual Explanations

The notebook includes original explanations covering:

- Why tokenization is necessary for LLMs
- Why BPE is used instead of word-level tokenization
- Why sliding windows are required for next-token prediction
- Why embeddings encode meaning
- How embeddings relate to neural network concepts

### Why Do Embeddings Encode Meaning?

Embeddings are learnable vector representations stored in an embedding matrix.  
Each token ID indexes a row in this matrix.

During training, vectors are updated via backpropagation to minimize next-token prediction loss.  
Tokens appearing in similar contexts receive similar gradient updates, leading to geometrically similar representations.

Thus, meaning emerges as spatial structure in high-dimensional vector space — not as symbolic rules.

---

## 🔬 Experiment: Effect of `max_length` and `stride`

We conducted a small experiment modifying:

- `max_length`
- `stride`

We measured the number of generated samples for different configurations.

### Observations:

- Smaller stride → more overlapping windows → more training samples
- Larger stride → fewer samples → lower computational cost
- Larger max_length → fewer valid starting positions → fewer samples

This demonstrates the tradeoff between:
- Dataset richness
- Context size
- Computational efficiency

Overlap is useful because it increases training diversity and preserves contextual continuity.

---

## ▶️ How to Run

Install dependencies:

```bash
pip install torch tiktoken numpy
