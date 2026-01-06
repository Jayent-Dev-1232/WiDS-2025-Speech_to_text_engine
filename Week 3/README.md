# **WEEK 3 — Transformers + Modern NLP**

### **Concepts**

* 2017 Transformer architecture (Q/K/V, self-attention)
* BERT vs GPT vs encoder–decoder models
* Tokenizers (WordPiece, BPE, SentencePiece)

### **Coding Tasks**

* Fine-tune a BERT model for text classification
* Train a custom SentencePiece tokenizer
* Use Hugging Face `transformers` for inference

### **Mini-Project**

* Build a simple Q&A or text summarizer using a pretrained model


---

## 1️⃣ Transformer Architecture (2017)

### What problem did Transformers solve?

Earlier NLP models (RNNs, LSTMs) processed text **sequentially**, which:

* Was slow
* Struggled with long-range dependencies

Transformers replaced recurrence with **self-attention**, allowing **parallel processing** and better context modeling.

---

### Core Components

#### 🔹 Self-Attention (Q / K / V)

Each token is projected into:

* **Query (Q)** – what am I looking for?
* **Key (K)** – what do I contain?
* **Value (V)** – what information do I give?

Attention score:

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

**Intuition**
Every word asks:

> “Which other words are important for me?”

Example:

> *“The animal didn’t cross the street because it was tired”*
> → “it” attends strongly to “animal”

---

#### 🔹 Multi-Head Attention

Instead of one attention mechanism:

* Use **multiple heads**
* Each head learns a different relationship (syntax, meaning, position)

---

#### 🔹 Positional Encoding

Transformers don’t have sequence order inherently, so we add **positional encodings**:

* Sinusoidal or learned embeddings
* Inject word order information

---

### 📚 Best Resources

* 🔗 **Attention Is All You Need (original paper)**
  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
* 🎥 Jay Alammar — *Illustrated Transformer* (must-read)
  [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
* 🎥 3Blue1Brown — *Attention explained visually*
  [https://www.youtube.com/watch?v=eMlx5fFNoYc](https://www.youtube.com/watch?v=eMlx5fFNoYc)

---

## 2️⃣ BERT vs GPT vs Encoder–Decoder

### 🔹 Encoder-Only — **BERT**

* Reads text **bidirectionally**
* Best for **understanding tasks**
* Masked Language Modeling (MLM)

**Used for:**

* Text classification
* NER
* Sentiment analysis
* Semantic search

---

### 🔹 Decoder-Only — **GPT**

* Reads **left-to-right**
* Autoregressive generation

**Used for:**

* Text generation
* Chatbots
* Story writing
* Code generation

---

### 🔹 Encoder–Decoder — T5, BART

* Encoder understands input
* Decoder generates output

**Used for:**

* Translation
* Summarization
* Question answering

---

### 📊 Quick Comparison

| Model   | Direction     | Strength          |
| ------- | ------------- | ----------------- |
| BERT    | Bidirectional | Understanding     |
| GPT     | Left → Right  | Generation        |
| T5/BART | Both          | Transforming text |

---

### 📚 Resources

* BERT paper: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
* Illustrated BERT: [https://jalammar.github.io/illustrated-bert/](https://jalammar.github.io/illustrated-bert/)
* GPT explained: [https://jalammar.github.io/illustrated-gpt2/](https://jalammar.github.io/illustrated-gpt2/)

---

## 3️⃣ Tokenizers (WordPiece, BPE, SentencePiece)

### Why Tokenizers Matter

Neural models don’t see words — they see **numbers**.
Tokenizers:

* Break text into subwords
* Handle unknown words
* Control vocabulary size

---

### 🔹 WordPiece (BERT)

Example:

```
unbelievable → un ##bel ##iev ##able
```

---

### 🔹 BPE (GPT)

* Merge frequent character pairs
* Efficient and simple

---

### 🔹 SentencePiece

* Language-agnostic
* Treats text as raw bytes
* No need for whitespace

Used heavily in multilingual models.

---

### 📚 Resources

* SentencePiece paper: [https://arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)
* Hugging Face tokenizer course:
  [https://huggingface.co/course/chapter2](https://huggingface.co/course/chapter2)

---

## 4️⃣ Coding Tasks — How to Do Them

---

### ✅ Fine-tune BERT for Text Classification

**Steps**

1. Load pretrained BERT
2. Add classification head
3. Fine-tune on your dataset

**Key Library:** **Hugging Face**

📘 Tutorial:

* [https://huggingface.co/docs/transformers/training](https://huggingface.co/docs/transformers/training)

📺 Video:

* [https://www.youtube.com/watch?v=8NslS5V5kz0](https://www.youtube.com/watch?v=8NslS5V5kz0)

---

### ✅ Train a Custom SentencePiece Tokenizer

**Why do this?**

* Domain-specific text (medical, legal, Indian languages)

📘 Official guide:

* [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)

📺 Practical walkthrough:

* [https://www.youtube.com/watch?v=9d7R8lD4Q2M](https://www.youtube.com/watch?v=9d7R8lD4Q2M)

---

### ✅ Inference Using Transformers

Typical flow:

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
classifier("Transformers are amazing")
```

📘 Docs:

* [https://huggingface.co/docs/transformers/pipeline_tutorial](https://huggingface.co/docs/transformers/pipeline_tutorial)

---

## 5️⃣ Mini-Project Ideas (Pick One)

### 🔹 Q&A System

* Model: `distilbert-base-cased-distilled-squad`
* Input: question + paragraph
* Output: answer span

Tutorial:

* [https://huggingface.co/tasks/question-answering](https://huggingface.co/tasks/question-answering)

---

### 🔹 Text Summarizer

* Model: `facebook/bart-large-cnn`
* Input: long article
* Output: summary

Tutorial:

* [https://huggingface.co/tasks/summarization](https://huggingface.co/tasks/summarization)

---

### ⭐ Suggested Deliverable

* CLI or notebook demo
* Explain:

  * Model choice
  * Tokenizer
  * Limitations

---

## 🔚 Week 3 Summary

You should finish this week knowing:

* How self-attention actually works
* Why BERT ≠ GPT
* How tokenization impacts models
* How to fine-tune and deploy Transformers

---


