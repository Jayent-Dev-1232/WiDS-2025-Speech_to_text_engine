

# 🚀 Weeks 5–6: Speech-to-Text Model + Deployment

*(Whisper / Wav2Vec 2.0 Track — Recommended)*

---

## WEEK 5 — Build the Speech-to-Text Model

### 🎯 Goal

Fine-tune a **pretrained STT model** that transcribes short phrases with **reasonable WER**.

---

## 1️⃣ Dataset Preparation (Common Voice)

### What you do

* Download **Mozilla Common Voice**
* Filter by language, duration (≤15 s)
* Remove empty / noisy transcripts

### 📚 Resources

**Dataset**

* 🔗 [https://commonvoice.mozilla.org](https://commonvoice.mozilla.org)
* HuggingFace loader:

```python
from datasets import load_dataset
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en")
```

**Videos**

* 🎥 *Fine-tuning Speech Models with HuggingFace* — HuggingFace
  [https://www.youtube.com/watch?v=TksaY_FDgnk](https://www.youtube.com/watch?v=TksaY_FDgnk)

**Reading**

* 📄 Common Voice Paper:
  *“Common Voice: A Massively-Multilingual Speech Corpus”*

---

## 2️⃣ Audio Preprocessing

### Concepts

* Resampling → **16 kHz**
* Normalization
* Silence trimming (optional)

### Tools

* `torchaudio`
* `librosa`

```python
import torchaudio
waveform, sr = torchaudio.load(path)
waveform = torchaudio.functional.resample(waveform, sr, 16000)
```

### 📚 Resources

* 📘 *Speech and Language Processing* — Jurafsky & Martin (Ch. 25)
* 🎥 *Audio preprocessing for ML* — Valerio Velardo
  [https://www.youtube.com/c/ValerioVelardo](https://www.youtube.com/c/ValerioVelardo)

---

## 3️⃣ Train / Test Splits

```python
dataset = dataset["train"].train_test_split(test_size=0.1)
```

### Best practice

* Keep **speaker-independent splits**
* Validate on unseen accents

---

## 4️⃣ Load Pretrained STT Model

### Recommended

| Model                  | When to Use                  |
| ---------------------- | ---------------------------- |
| **Whisper-small/base** | Best accuracy, multilingual  |
| **Wav2Vec2-base**      | Lightweight, faster training |

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
```

### 📚 Resources

* 📄 **Whisper Paper** (OpenAI, 2022)
* 📄 **Wav2Vec 2.0 Paper** (Baevski et al.)
* 🎥 *How Whisper Works* — AssemblyAI
  [https://www.youtube.com/watch?v=UeT6U5v8Z3A](https://www.youtube.com/watch?v=UeT6U5v8Z3A)

---

## 5️⃣ Tokenize Transcripts

### Whisper

* Built-in tokenizer
* Handles punctuation + casing

### Wav2Vec2

* Character-level tokenizer
* CTC loss

### 📚 Resources

* HuggingFace Docs:

  * [https://huggingface.co/docs/transformers/tasks/asr](https://huggingface.co/docs/transformers/tasks/asr)

---

## 6️⃣ Fine-Tune the Model

### Training Setup

* Optimizer: AdamW
* Batch size: 8–16
* LR: 1e-5 (Whisper), 3e-4 (Wav2Vec2)

```python
from transformers import Trainer, TrainingArguments
```

### 📚 Resources

* 🎥 *Fine-tuning Whisper End-to-End*
  [https://www.youtube.com/watch?v=ZpZ_7yqF2qE](https://www.youtube.com/watch?v=ZpZ_7yqF2qE)
* 📘 *Deep Learning for Audio* — MIT 6.S191

---

## 7️⃣ Track Metrics (WER)

```python
from evaluate import load
wer = load("wer")
```

### 📚 Resources

* 📄 *Word Error Rate Explained* — NIST
* 🎥 *ASR Evaluation Metrics* — CMU

---

## 8️⃣ Save & Evaluate Model

```python
trainer.save_model("stt_model")
```

### Milestone

✅ Transcribes **short phrases**
✅ WER improves vs baseline

---

# WEEK 6 — Final Project + Deployment

## 🎯 Goal

Turn your model into a **real-world STT application**

---

## 1️⃣ Model Optimization

### Techniques

* Quantization (INT8)
* Pruning
* ONNX export

```python
torch.quantization.quantize_dynamic(...)
```

### 📚 Resources

* 📘 *Efficient Deep Learning* — MIT
* 📄 ONNX Runtime ASR Guide

---

## 2️⃣ Real-Time Inference Pipeline

### Flow

```
Mic → Audio Buffer → Feature Extractor → Model → Text
```

### Tools

* `sounddevice`
* `pyaudio`

### 📚 Resources

* 🎥 *Real-time Speech Recognition in Python*
  [https://www.youtube.com/watch?v=JYfHq7j6y6U](https://www.youtube.com/watch?v=JYfHq7j6y6U)

---

## 3️⃣ Model Serving

### Recommended Stack

| Tool          | Why              |
| ------------- | ---------------- |
| **Gradio**    | Fast demos       |
| **FastAPI**   | Production-ready |
| **Streamlit** | UI-first         |

```python
import gradio as gr
```

### 📚 Resources

* 🎥 *Deploy Whisper with Gradio*
* 📘 *Designing Machine Learning Systems* — Chip Huyen

---

## 4️⃣ Final Project Deliverables

### ✔ What You Submit

* Fine-tuned Whisper/Wav2Vec2 model
* Web app (upload + mic input)
* Evaluation report:

  * Dataset
  * Architecture
  * Training pipeline
  * WER
  * Demo examples

---

## 🧠 Optional Extensions (Advanced)

### Speaker Diarization

* pyannote.audio
  📄 *pyannote: Neural Speaker Diarization*

### Punctuation Restoration

* BERT / T5 post-processing

### Mobile Deployment

* ONNX → CoreML / TFLite

---

# 📚 MASTER REFERENCE LIST

### Books

1. **Speech and Language Processing** — Jurafsky & Martin
2. **Deep Learning** — Goodfellow et al.
3. **Designing ML Systems** — Chip Huyen

### Courses

* 🎓 Stanford CS224N (Speech lectures)
* 🎓 MIT 6.S191 (Audio DL)

### Repos

* [https://github.com/openai/whisper](https://github.com/openai/whisper)
* [https://github.com/facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)
* [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

---

