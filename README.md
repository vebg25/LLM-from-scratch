# 🧠 LLM From Scratch

> A complete, from-scratch PyTorch implementation of a GPT-124M style Large Language Model — built component by component, from raw text tokenisation to a fully trained transformer generating text on Edith Wharton's *The Verdict*.

---

## 📖 What This Project Is

This repository implements every internal component of a modern GPT-style language model from scratch in pure PyTorch — no high-level wrappers, no pretrained checkpoints, no shortcuts. Every class, every matrix multiplication, every design decision is written and explained explicitly.

The project follows a deliberate learning progression across four chapters:

1. **How raw text becomes numbers** — tokenisation, vocabulary, embeddings
2. **How tokens attend to each other** — from basic dot-product similarity to causal multi-head attention
3. **How the full transformer is assembled** — GELU, LayerNorm, FeedForward, residual connections, TransformerBlock, GPTModel
4. **How the model learns** — training loop, cross-entropy loss, AdamW, text generation

The training corpus is **"The Verdict"** by Edith Wharton (~20KB) — compact enough to train on a laptop CPU, meaningful enough to verify the pipeline works end to end.

---

## 🗂️ Repository Structure

```
LLM-from-scratch/
│
├── preprocessing_chapter_1/
│   ├── _1_retrieving-verdict-file.py     # Downloads the-verdict.txt from GitHub
│   ├── _2_SimpleTokenizerV1.py           # Vocabulary-based tokenizer (encode/decode)
│   ├── _3_SimpleTokenizerV2.py           # V2: adds <|UNK|> and <|endoftext|> tokens
│   ├── _4_BytePairEncodingTokenizer.py   # GPT-2 BPE via tiktoken
│   ├── _5_slidingWindowSampling.py       # Next-token prediction dataset demonstration
│   ├── _6_CreatingDataset.py             # PyTorch Dataset + DataLoader (sliding window)
│   └── _7_EmbeddingCreation.py           # Token embeddings + positional embeddings
│
├── Attention_chapter_2/
│   ├── SimplifiedAttention.py            # Dot-product attention for a single query (manual loop)
│   ├── SimplifiedAttention_2.py          # Same, rewritten as matrix multiplication (inputs @ inputs.T)
│   ├── SelfAttention.py                  # Full Q/K/V projections with nn.Parameter
│   ├── SelfAttention_2.py                # Same with nn.Linear (better weight init)
│   ├── SelfAttention_3.py                # Compact version — same architecture, cleaner code
│   ├── CausalAttention.py                # Masked causal attention + MultiHeadAttentionWrapper
│   └── MultiHeadAttention.py             # Efficient single-matrix multi-head attention
│
├── Transformer_Architecture_chapter_3/
│   ├── GELU.py                           # GELU activation + FeedForward module (expand 4×, project back)
│   ├── LayerNormalization.py             # LayerNorm with learnable scale and shift
│   ├── dummy_llm_architecture.py         # GPT-124M config + DummyGPTModel (shape verification)
│   └── transformer_block_1.py            # TransformerBlock (pre-LN) + full GPTModel
│
├── Training_chapter_4/
│   └── 1_generate_simple_text.py         # Training loop, loss functions, text generation
│
├── the-verdict.txt                        # Training corpus (~20KB, Edith Wharton)
└── README.md
```

---

## 📚 Chapter Breakdown

### Chapter 1 — Text Preprocessing & Tokenisation

**What it covers:** The complete pipeline from raw `.txt` file to batched tensor inputs ready for a model.

**Files and what each one does:**

`_2_SimpleTokenizerV1.py` — Builds a vocabulary from *The Verdict* by splitting on punctuation and whitespace with regex, assigns each unique token an integer, and implements `encode()` (text → IDs) and `decode()` (IDs → text). Fails on any word not seen during training.

`_3_SimpleTokenizerV2.py` — Extends V1 with two special tokens: `<|UNK|>` (handles unknown words gracefully instead of crashing) and `<|endoftext|>` (separator between independent text documents). Demonstrates encoding two separate texts joined by the separator.

`_4_BytePairEncodingTokenizer.py` — Drops the custom vocabulary entirely and plugs in OpenAI's `tiktoken` GPT-2 BPE tokenizer (50,257 vocab). Shows that even unknown words like `"Akwirw ier"` get handled via byte-level fallback — no `<|UNK|>` ever needed.

`_5_slidingWindowSampling.py` — Shows the core self-supervised training signal: given tokens `[t0, t1, t2, t3]`, the model learns `t0→t1`, `t0,t1→t2`, `t0,t1,t2→t3`. Each input predicts its immediate successor.

`_6_CreatingDataset.py` — Implements `DatasetV1(Dataset)` using a sliding window with configurable `max_length` and `stride`. Creates parallel `input_ids` and `target_ids` tensors (target is input shifted right by one). Wraps in `DataLoader` with batching, shuffling, and drop-last.

`_7_EmbeddingCreation.py` — Shows how token IDs become continuous vectors via `nn.Embedding(vocab_size=50257, output_dim=256)`, and how learnable positional embeddings (`nn.Embedding(context_length, output_dim)`) encode position. The two are added element-wise to produce the final input to the transformer.

**Key insight from this chapter:** The training data requires no labels — the text supervises itself. Every token is simultaneously an input (to predict the next) and a target (the answer for the previous).

---

### Chapter 2 — Attention Mechanisms

**What it covers:** A six-file progression that builds the attention mechanism from first principles — starting with a single dot-product similarity score and ending with efficient multi-head causal attention.

**The progression:**

`SimplifiedAttention.py` — Attention explained for a single query token ("journey") using a manual Python loop. Computes `torch.dot(query, x_i)` for each token, applies softmax, and sums weighted value vectors into a context vector. No learnable parameters.

`SimplifiedAttention_2.py` — The same computation rewritten as `inputs @ inputs.T` — one matrix multiplication replaces the entire loop. Same result, dramatically faster.

`SelfAttention.py` — Introduces trainable weight matrices `W_query`, `W_key`, `W_value` (as `nn.Parameter`). The model now learns *what* to pay attention to, not just computing raw similarity. Wraps everything in an `nn.Module` with a proper `forward()`.

`SelfAttention_2.py` — Replaces `nn.Parameter(torch.rand(...))` with `nn.Linear(d_in, d_out, bias=False)`. Functionally identical but benefits from PyTorch's Kaiming uniform weight initialisation — more stable training from the start.

`CausalAttention.py` — The critical step for autoregressive language modelling. Adds an upper-triangular mask registered as a buffer via `register_buffer()`. Before softmax, future positions are filled with `-inf`, making their softmax weight exactly 0. The model can only attend to past and current tokens — never future ones. Also adds dropout on attention weights. Includes `MultiHeadAttentionWrapper` that runs multiple `CausalAttention` heads independently and concatenates their outputs.

`MultiHeadAttention.py` — The production version. Instead of running N separate attention modules, projects Q, K, V to full `d_out` dimensions, then reshapes into `(batch, num_heads, seq_len, head_dim)` using `.view()` and `.transpose()`. All heads computed in a single batched matrix multiplication. Adds an output projection `nn.Linear(d_out, d_out)` to mix information across heads.

**Key insight from this chapter:** The causal mask is what makes a transformer a *language model* rather than just an encoder. Without it, the model would trivially learn to copy the next token by attending directly to it.

---

### Chapter 3 — Transformer Architecture

**What it covers:** Assembling GELU, LayerNorm, FeedForward, and MultiHeadAttention into a full GPT-style model.

**Components:**

`LayerNormalization.py` — Implements LayerNorm from scratch:
```
norm_x = (x - mean) / sqrt(var + eps)
output  = norm_x * scale + shift
```
where `scale` (ones) and `shift` (zeros) are learnable parameters. The `eps=1e-5` prevents division by zero. Placed *before* attention and FFN (pre-LN) rather than after — the placement used in modern GPT implementations for more stable training.

`GELU.py` — Implements GELU using the tanh approximation:
```
GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```
and wraps it in a `FeedForward` module: `Linear(emb_dim → 4×emb_dim) → GELU → Linear(4×emb_dim → emb_dim)`. The 4× expansion gives the model capacity to combine features before projecting back. GELU is preferred over ReLU in transformers because it has non-zero gradient for negative inputs.

`dummy_llm_architecture.py` — Establishes the **GPT-124M configuration**:
```python
GPT_CONFIG_124M = {
    "vocab_size":     50257,   # GPT-2 BPE vocabulary
    "context_length": 1024,    # Max tokens in a single forward pass
    "emb_dim":        768,     # Embedding dimension
    "n_heads":        12,      # Attention heads
    "n_layers":       12,      # Stacked transformer blocks
    "drop_rate":      0.1,     # Dropout probability
    "qkv_bias":       False,   # No bias in QKV projections
}
```
Then builds `DummyGPTModel` — a full model scaffold with real embedding layers and a real output head, but with placeholder `DummyTransformerBlock` (identity) and `DummyLayerNorm` (identity). This verifies the model's input/output shapes are correct before filling in the real components.

`transformer_block_1.py` — Assembles the full working model:

`TransformerBlock` wires together all components:
```
x → LayerNorm → MultiHeadAttention → Dropout → + residual → 
  → LayerNorm → FeedForward       → Dropout → + residual
```
The two residual (shortcut) connections — one around attention, one around the feed-forward — are critical: they give gradients a direct path back through the network during backpropagation, making 12+ layer models trainable.

`GPTModel` stacks `n_layers=12` of `TransformerBlock`, preceded by token + positional embeddings and followed by a final `LayerNorm` and a linear output head projecting to `vocab_size=50257` logits.

**Key insight from this chapter:** The pre-norm placement (`LayerNorm` before each sub-layer rather than after) is a small but important difference from the original 2017 "Attention Is All You Need" paper. Modern GPT models use pre-norm because it produces more stable gradients at initialisation.

---

### Chapter 4 — Pretraining & Text Generation

**What it covers:** Training the GPT model end-to-end on *The Verdict* and generating coherent text.

**The training setup:**
- **Model:** GPT-124M architecture, but `context_length` reduced to `256` (from 1024) for memory efficiency
- **Corpus:** *The Verdict* — 20,479 characters, ~5,145 tokens after BPE encoding
- **Split:** 90% training / 10% validation (character-level split)
- **DataLoader:** `batch_size=2`, `stride=context_length` (non-overlapping windows, no data leakage)
- **Optimiser:** AdamW — `lr=0.0004`, `weight_decay=0.1`
- **Training:** 10 epochs

**Key functions implemented:**

`calc_loss_batch()` — Runs a forward pass, flattens `(batch, seq_len, vocab_size)` logits to `(batch×seq_len, vocab_size)`, and computes cross-entropy against the target token IDs. This is next-token prediction: the model is penalised for every token it gets wrong.

`calc_loss_loader()` — Averages loss across batches in a DataLoader, with optional `num_batches` limit for fast mid-epoch evaluation.

`evaluate_model()` — Switches to `model.eval()` + `torch.no_grad()` to compute train/val loss without updating weights. Switches back to `model.train()` after.

`generate_text_simple()` — Greedy text generation: takes a prompt as token IDs, runs a forward pass, takes `argmax` of the last position's logit distribution, appends the new token, and repeats for `max_new_tokens` steps.

`train_model_simple()` — The full training loop: iterates epochs and batches, computes loss, calls `.backward()`, steps the optimiser, logs train/val loss every `eval_freq` steps, and prints a generated sample at the end of each epoch.

The trained model is saved as `model_and_optimizer.pth` via `torch.save()`.

**Key insight from this chapter:** Training on just 5,000 tokens for 10 epochs is enough to see the model's outputs shift from random tokens to fragments that resemble the training corpus — proof that gradient descent through 124M parameters (even untrained) is finding structure in the data.

---

## 🏗️ Architecture at a Glance

```
Token IDs  (batch, seq_len)
     │
     ├─── Token Embedding     nn.Embedding(50257, 768)
     ├─── Position Embedding  nn.Embedding(1024, 768)    ← learned, not sinusoidal
     └─── Sum + Dropout(0.1)
                │
    ┌───────────┴───────────────────────────────┐
    │  TransformerBlock × 12                     │
    │                                            │
    │  x → LayerNorm                             │
    │    → MultiHeadAttention (12 heads, d=768)  │
    │    → Dropout(0.1)                          │
    │    → + residual                            │
    │                                            │
    │  x → LayerNorm                             │
    │    → FeedForward (768 → 3072 → 768, GELU)  │
    │    → Dropout(0.1)                          │
    │    → + residual                            │
    └───────────────────────────────────────────┘
                │
          Final LayerNorm
                │
          Linear (768 → 50257)     ← output logits, no bias
                │
          Softmax → next token probability distribution
```

**Total parameters (GPT-124M):** ~124 million

---

## ⚡ Quick Start

```bash
# Install dependencies
pip install torch tiktoken

# Chapter 1: Tokenisation
python preprocessing_chapter_1/_2_SimpleTokenizerV1.py
python preprocessing_chapter_1/_6_CreatingDataset.py

# Chapter 2: Attention
python Attention_chapter_2/CausalAttention.py
python Attention_chapter_2/MultiHeadAttention.py

# Chapter 3: Full Model
python Transformer_Architecture_chapter_3/transformer_block_1.py

# Chapter 4: Train + Generate
python Training_chapter_4/1_generate_simple_text.py
```

---

## 🔧 Model Configuration

```python
GPT_CONFIG_124M = {
    "vocab_size":     50257,  # GPT-2 BPE tokenizer vocabulary
    "context_length": 256,    # Tokens per forward pass (reduced for training efficiency)
    "emb_dim":        768,    # Embedding and hidden dimension
    "n_heads":        12,     # Attention heads (head_dim = 768 / 12 = 64)
    "n_layers":       12,     # Stacked transformer blocks
    "drop_rate":      0.1,    # Dropout applied after attention and FFN
    "qkv_bias":       False,  # No additive bias in Q, K, V linear projections
}

# Training hyperparameters
optimizer  = AdamW(lr=0.0004, weight_decay=0.1)
num_epochs = 10
batch_size = 2
stride     = context_length  # Non-overlapping windows
```

---

## 🧪 What This Repository Teaches

After working through every file in order, you will be able to:

- Implement a BPE tokenizer from scratch and explain why vocabulary size affects token fertility
- Derive the scaled dot-product attention formula `softmax(QK^T / √d_k) V` from first principles
- Explain why causal masking (`-inf` before softmax) is required for autoregressive generation
- Describe the difference between `nn.Parameter` and `nn.Linear` for weight matrices and when each is preferred
- Implement LayerNorm from scratch and explain why pre-norm placement stabilises deep network training
- Explain the role of residual connections in enabling gradient flow through 12+ layer networks
- Build a training loop with proper train/eval switching, loss computation, and checkpoint saving
- Generate text from a trained model using greedy decoding and explain its limitations

---

## 📂 Based On

**Build a Large Language Model (From Scratch)**
Sebastian Raschka — Manning Publications, 2024
[https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

---
