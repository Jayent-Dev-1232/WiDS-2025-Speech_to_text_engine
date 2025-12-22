# Week 2

# **WEEK 2 — Deep Learning for NLP**

### **Concepts**

* Word embeddings: Word2Vec, GloVe
* Neural networks for NLP:

  * RNN, GRU, LSTM
  * Attention mechanism
* Intro to transformers

### **Coding Tasks**

* Train Word2Vec/GloVe on a small text corpus
* Build an LSTM sentiment classifier
* Visualize embeddings using PCA/TSNE

### **Mini-Project**

* Compare performance of:

  * TF-IDF + Logistic Regression
  * LSTM classifier

---

Perfect—Week 2 is where NLP starts to feel *alive*. Here’s a **clean, high-quality resource stack** for each topic, mixing theory + hands-on. I’ll flag **must-do** items so you don’t drown in links.

---

## 🧠 Word Embeddings (Word2Vec, GloVe)

### 📖 Theory

* **Word2Vec (CBOW & Skip-Gram)**

  * Original paper: *“Efficient Estimation of Word Representations in Vector Space”* — Mikolov et al.
  * Clear explanation + math:
    👉 [https://jalammar.github.io/illustrated-word2vec/](https://jalammar.github.io/illustrated-word2vec/) (**must-read**)

* **GloVe**

  * Stanford NLP overview:
    👉 [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
  * Intuition-focused explanation:
    👉 [https://towardsdatascience.com/introduction-to-glove-embeddings-2a7f84d0e8d8](https://towardsdatascience.com/introduction-to-glove-embeddings-2a7f84d0e8d8)

---

### 💻 Coding

* **Train Word2Vec**

  * `gensim` tutorial (simple & powerful):
    👉 [https://radimrehurek.com/gensim/models/word2vec.html](https://radimrehurek.com/gensim/models/word2vec.html) (**must-do**)

* **Train GloVe**

  * Simple Python implementation:
    👉 [https://github.com/stanfordnlp/GloVe](https://github.com/stanfordnlp/GloVe)
  * Lightweight tutorial:
    👉 [https://medium.com/@martinpella/how-to-use-glove-word-embeddings-in-python-d7b7f7e5d7](https://medium.com/@martinpella/how-to-use-glove-word-embeddings-in-python-d7b7f7e5d7)

---

## 🔁 RNN, GRU, LSTM (for NLP)

### 📖 Conceptual Understanding

* **Why RNNs fail → why LSTM/GRU exist**
  👉 [https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (**legendary explanation**)

* GRU vs LSTM (intuitive):
  👉 [https://towardsdatascience.com/gru-vs-lstm-86e8f79b2a7c](https://towardsdatascience.com/gru-vs-lstm-86e8f79b2a7c)

---

### 💻 Coding

* **LSTM Sentiment Classifier (PyTorch)**
  👉 [https://www.kaggle.com/code/affand20/imdb-with-pytorch-lstm](https://www.kaggle.com/code/affand20/imdb-with-pytorch-lstm)
  👉 [https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)

* **TensorFlow/Keras version**
  👉 [https://www.tensorflow.org/text/tutorials/text_classification_rnn](https://www.tensorflow.org/text/tutorials/text_classification_rnn)

> 🔑 Tip: Start with **IMDB dataset**, then reduce size to experiment faster.

---

## 🎯 Attention Mechanism

### 📖 Understanding Attention

* **Visual + intuitive explanation**
  👉 [https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) (**must-read**)

* Slightly more math-focused:
  👉 [https://towardsdatascience.com/attention-in-neural-networks-e66920838742](https://towardsdatascience.com/attention-in-neural-networks-e66920838742)

---

### 💻 Coding (Optional but Powerful)

* Attention in PyTorch (Seq2Seq):
  👉 [https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

---

## 🤖 Intro to Transformers

### 📖 Conceptual

* **The gold standard intro**
  👉 [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/) (**absolutely must-read**)

* Transformer vs RNN (clean comparison):
  👉 [https://towardsdatascience.com/transformers-vs-rnns-4e3b5c1e21f7](https://towardsdatascience.com/transformers-vs-rnns-4e3b5c1e21f7)

---

### 💻 Practical

* Hugging Face beginner NLP course:
  👉 [https://huggingface.co/learn/nlp-course/chapter1](https://huggingface.co/learn/nlp-course/chapter1)
* Simple Transformer text classification:
  👉 [https://huggingface.co/docs/transformers/tasks/sequence_classification](https://huggingface.co/docs/transformers/tasks/sequence_classification)

---

## 📊 Embedding Visualization (PCA / t-SNE)

### 📖 + 💻

* PCA vs t-SNE explained:
  👉 [https://distill.pub/2016/misread-tsne/](https://distill.pub/2016/misread-tsne/)

* Visualizing Word Embeddings (Python):
  👉 [https://www.kaggle.com/code/jeffd23/visualizing-word-vectors-with-t-sne](https://www.kaggle.com/code/jeffd23/visualizing-word-vectors-with-t-sne)
  👉 [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

---

## 🧪 Mini-Project: TF-IDF vs LSTM

### TF-IDF + Logistic Regression

* End-to-end tutorial:
  👉 [https://www.kaggle.com/code/sudalairajkumar/simple-exploration-notebook-imdb-review](https://www.kaggle.com/code/sudalairajkumar/simple-exploration-notebook-imdb-review)

### LSTM Classifier

* IMDB sentiment with LSTM (clean):
  👉 [https://www.kaggle.com/code/bansalshubham907/lstm-sentiment-analysis](https://www.kaggle.com/code/bansalshubham907/lstm-sentiment-analysis)

### Comparison Guidance

* Metrics to compare:

  * Accuracy
  * F1-score
  * Training time
  * Overfitting behavior
* Expected result:

  * **TF-IDF + LR** → strong baseline, fast
  * **LSTM** → better context, slower, needs tuning

---

## 📚 Optional (If You Want Depth)

* **Speech and Language Processing** - Chapters on embeddings & neural NLP
* Stanford CS224N (videos + notes):
  👉 [https://web.stanford.edu/class/cs224n/](https://web.stanford.edu/class/cs224n/)

---

## 🔥 Suggested Week 2 Flow

1. Read **Illustrated Word2Vec**
2. Train Word2Vec on a small corpus
3. Build **TF-IDF + LR**
4. Build **LSTM classifier**
5. Visualize embeddings (t-SNE)
6. Read about **Illustrated Transformer**


