# WIDS – Speech to Text Engine

This repository is created as part of the **Winter in Data Science (WiDS)** project under the **Analytics Club, IIT Bombay** and is being mentored by [Aditya Sanapala](https://github.com/adityasanapala).

**Mentee Name:** [Jayent Dev](https://github.com/Jayent-Dev-1232)  
**Roll Number:** 24B1232  

---

## 📌 Program Overview

This project documents my structured journey through **Natural Language Processing (NLP)** and **Deep Learning for NLP**, with a focus on building strong foundations required for developing a **Speech-to-Text Engine** by the end of the project.  

The repository contains:
- Conceptual learning notes
- Hands-on coding implementations
- Mini-projects and comparisons
- Reports and visualizations

---

## 📚 Table of Contents

- [Week 1 – Foundations of NLP](#week-1--foundations-of-nlp)
  - Learning
  - Tasks Completed
- [Week 2 – Deep Learning for NLP](#week-2--deep-learning-for-nlp)
  - Learning
  - Tasks Completed
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

---

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
|
├── README.md

---

## 🧠 Key Learnings

- Classical NLP pipelines provide strong and fast baselines
- TF-IDF + Logistic Regression is an effective benchmark
- Word embeddings capture semantic relationships beyond bag-of-words
- LSTMs improve contextual understanding through sequence modeling
- Deep learning models offer better performance but require careful tuning

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

---

This repository will continue to be updated as the project progresses toward building a complete **Speech-to-Text Engine** using advanced NLP and deep learning techniques.
