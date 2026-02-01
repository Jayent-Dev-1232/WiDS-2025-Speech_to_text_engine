# WIDS – Speech to Text Engine

This repository is created as part of the **Winter in Data Science (WiDS)** project under the **Analytics Club, IIT Bombay** and is being mentored by [Aditya Sanapala](https://github.com/adityasanapala).

**Mentee Name:** [Jayent Dev](https://github.com/Jayent-Dev-1232)  
**Roll Number:** 24B1232  

---

## 📌 Program Overview

This project documents my structured journey through **Natural Language Processing (NLP)**, **Deep Learning for NLP** and **Speech Processing**, culminating in building a **Speech-to-Text Engine** and deploying it as a real world application.  

The repository contains:
- Conceptual learning notes
- Hands-on coding implementations
- Mini-projects and comparisons
- Reports and visualizations
- End-to-End Speech-to-Text Model + Deployment

---

## 📚 Table of Contents

- [Week 1 – Foundations of NLP](#week-1--foundations-of-nlp)
  - Learning
  - Tasks Completed
- [Week 2 – Deep Learning for NLP](#week-2--deep-learning-for-nlp)
  - Learning
  - Tasks Completed
- [Week 3 - Transformers + Modern NLP](#week-3--transformers-+-modern-nlp)
  - Learning
  - Tasks Completed
  - Mini Project
- [Week 4 - Introduction to Speech Processing](#week-4--introduction-to-speech-processing)
  - Learning
  - Tasks Completed
- [Week 5 - Speech-to-Text Model Traning and Deployment](#week-5--speech-to-text-model-training-and-deployment)
- [Repository Structure](#repository-structure)
- [Key Learnings](#key-learnings)
- [Acknowledgements](#acknowledgements)

---

## Week 1 – Foundations of NLP

### 📖 Learning

In Week 1, I focused on building a strong theoretical and practical foundation in classical NLP. The key concepts studied include:

- What is NLP: tokens, types, and corpora
- Text preprocessing techniques:
  - Tokenization
  - Stemming vs lemmatization
  - Stopword removal
  - N-grams
- Classical machine learning methods for NLP:
  - Bag-of-Words
  - TF-IDF
- Evaluation metrics overview:
  - Accuracy
  - F1-score
  - BLEU score
- Introduction to NLP libraries:
  - NLTK
  - spaCy
- Exploring datasets using Hugging Face

---

### 💻 Tasks Completed

- Implemented tokenization and stemming using **Python + NLTK / spaCy**
- Built a **TF-IDF based sentiment classifier** (IMDb / SST-2 style)
- Explored NLP datasets using **Hugging Face Datasets**

---

## Week 2 – Deep Learning for NLP

### 📖 Learning

Week 2 focused on modern NLP techniques using neural networks and representation learning. The following topics were covered:

- Word embeddings:
  - Word2Vec
  - GloVe (theoretical understanding)
- Neural networks for NLP:
  - RNN
  - GRU
  - LSTM
- Attention mechanism (conceptual understanding)
- Introduction to Transformers
- Dimensionality reduction and visualization:
  - PCA
  - t-SNE

### 💻 Tasks Completed

- Trained **Word2Vec embeddings** on a text corpus
- Visualized word embeddings using **t-SNE**
- Built a **TF-IDF + Logistic Regression** baseline model
- Built an **LSTM-based sentiment classifier**
- Compared classical ML vs deep learning approaches

#### 🔗 Code & Reports

- **Word2Vec Training:**  
  `notebooks/01_word2vec_training.ipynb`

- **Embedding Visualization (t-SNE):**  
  `notebooks/02_embedding_visualization.ipynb`

- **TF-IDF + Logistic Regression:**  
  `notebooks/03_tfidf_logreg.ipynb`

- **LSTM Sentiment Classifier:**  
  - Model definition: `src/lstm_model.py`  
  - Training script: `src/train_lstm.py`

- **Performance Comparison Report:**  
  `report/WiDS_Speech_to_Text_Engine_Week_2_Report.pdf`

---

## Week 3 - Transformers + Modern NLP

### Learning
- Transformer Architecture (Self-Attention, Q/K/V)
- Encoder vs Decoder vs Encoder-Decoder
- BERT vs GPT vs T5/BART
- Tokenizers
  - WordPiece
  - BPE
  - SentencePiece

### Tasks Completed
- Fine-tuned BERT for Text Classification
- Trained Custom SentencePiece Tokenizer
- Transformer inference pipelines

### Mini Project
- Built Text Summarizer / Q&A System using pretrained Transformers

---

## Week 4 - Introduction to Speech Processing

### Learning
- Speech as Digital Signal
- Waveforms & Frequency Analysis
- Spectrograms & Log Spectrograms
- Mel Spectrograms & MFCCs
- Classical vs Neural Speech Recognition
- CTC Loss Concept

### Tasks Completed
- Audio loading & normalization
- Waveform visualization
- Spectrogram & MFCC extraction
- Feature comparison
- Keyword Spotting Model (Yes/No classification using CNN)

---

## Week 5 - Speech-to-Text Model Traning and Deployment
- Fine-tuned a pretrained STT model to transcribe short phrases with reasonable WER
- Turn the trained model into a real-time Speech-to-Text Application

---

## 📁 Repository Structure

```text
WIDS-2025-SPEECH_TO_TEXT_ENGINE/
│
├── Week 1/
│   ├── Week-1_Solutions/
│   ├── Coding_Tasks_Solution.ipynb
│   ├── imdb_train.csv
│   ├── imdb_test.csv
│   ├── Notes.md
│   ├── WiDS_Speech_to_Text_Engine_Report.pdf
│   └── README.md
│
├── Week 2/
│   ├── Coding_Tasks_Solutions/
│   │   ├── data/
│   │   ├── notebooks/
│   │   ├── report/
│   │   └── src/
│   └── README.md
├──Week 3/
|  ├──README.md
|  ├──mini_project_summarizer.ipynb
|  ├──Report.md
|  ├──tokenizer_sentencepiece.ipynb
|  └──transformer_basics_and_inference.ipynb
|
├──Week 4/
|  ├──data/
|  ├──notebooks/
|  ├──src/
|  ├──README.md
|  └──requirements.txt
|
├──Week 5/
|  ├──README.md
|  ├──stt_deployment.ipynb
|  └──stt_model_training.ipynb
|
└──README.md
```

---

## 🧠 Key Learnings

- Classical NLP pipelines provide strong and fast baselines
- TF-IDF + Logistic Regression is an effective benchmark
- Word embeddings capture semantic relationships beyond bag-of-words
- LSTMs improve contextual understanding through sequence modeling
- Deep learning models offer better performance but require careful tuning
- Transformers revolutionize NLP with self-attention
- Speech signals require time-frequency analysis
- MFCCs and Mel Spectrograms are critical audio features
- Whisper/Wav2Vec2 enable end-to-end STT
- Deployment skills are as important as model accuracy

---

## 🙏 Acknowledgements

- **Winter in Data Science (WiDS)** – Analytics Club, IIT Bombay  
- Mentor: [Aditya Sanapala](https://github.com/adityasanapala)  
- Open-source tools and libraries:
  - PyTorch
  - Gensim
  - scikit-learn
  - Hugging Face
  - NLTK
  - Librosa

---

This repository will continue to be updated as the project progresses toward building a complete **Speech-to-Text Engine** using advanced NLP and deep learning techniques.
