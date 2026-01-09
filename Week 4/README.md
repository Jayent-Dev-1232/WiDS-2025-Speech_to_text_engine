# **WEEK 4 — Introduction to Speech Processing**

## **Learning Objectives**

By the end of this week, you should be able to:

* Understand how speech is represented as a digital signal
* Extract common speech features (spectrograms, Mel, MFCCs)
* Build an end-to-end audio preprocessing pipeline
* Understand how classical and neural speech recognition systems differ
* Implement a simple keyword spotting model

---

## **1. Core Concepts**

### **1.1 What Is Speech?**

* Speech as a **time-domain signal**
* Waveform representation
* Frequency content of speech
* Speech vs noise

**Key ideas**

* Periodicity (voiced vs unvoiced sounds)
* Formants
* Temporal structure

📚 **Resources**

* *Speech and Language Processing* — Jurafsky & Martin (Ch. 9)
* MIT OpenCourseWare: *Digital Signal Processing* (Intro lectures)

---

### **1.2 Audio Signal Basics**

* Sampling rate (Nyquist theorem)
* Amplitude & dynamic range
* Mono vs stereo
* Bit depth

**Key formulas**

* Sampling frequency
* Discrete-time signal representation

📚 **Resources**

* Stanford CS224S: *Speech Processing* (Lecture 1)
* Librosa documentation: *Audio Basics*

---

### **1.3 Time–Frequency Representations**

* Short-Time Fourier Transform (STFT)
* Spectrograms
* Log spectrograms

**Why spectrograms?**

* Speech is non-stationary
* Frequency content changes over time

📚 **Resources**

* Coursera: *Audio Signal Processing for Music Applications*
* Librosa STFT tutorial

---

### **1.4 Mel Scale & MFCCs**

* Human auditory perception
* Mel frequency scale
* Mel spectrograms
* MFCC pipeline:

  1. STFT
  2. Mel filterbanks
  3. Log compression
  4. DCT

📚 **Resources**

* Andrew Ng’s Deep Learning Specialization (Speech section)
* “MFCC Tutorial” — Dan Ellis (Columbia)

---

### **1.5 Speech Recognition Approaches**

#### **Classical STT**

* GMM-HMM
* Feature engineering
* Lexicons & language models

#### **Neural STT**

* End-to-end models
* CNNs / RNNs / Transformers
* Learned features

📚 **Resources**

* Jurafsky & Martin (Ch. 9–10)
* DeepSpeech paper (overview level)

---

### **1.6 CTC Loss (Conceptual)**

* Why alignment is hard
* Blank tokens
* Many-to-one mappings
* How CTC enables end-to-end training

📚 **Resources**

* Distill.pub: *Sequence Modeling with CTC*
* Baidu DeepSpeech blog posts

---

## **2. Coding Tasks**

### **2.1 Audio Loading & Inspection**

* Load `.wav` files
* Check duration, sampling rate
* Normalize amplitude

**Tools**

* `librosa`
* `pydub`

---

### **2.2 Visualization**

* Plot waveform
* Plot spectrogram
* Plot Mel spectrogram
* Plot MFCCs

**Skills practiced**

* Time–frequency intuition
* Feature comparison

---

### **2.3 Feature Extraction Pipeline**

* Pre-emphasis (optional)
* Framing & windowing
* Feature normalization
* Save features to disk

---

### **2.4 Keyword Spotting Model**

**Task**

* Binary classification: “yes” vs “no”

**Suggested approach**

* Input: Mel spectrograms or MFCCs
* Model:

  * CNN (preferred)
  * Or small MLP for baseline
* Loss: Cross-entropy
* Dataset:

  * Google Speech Commands (subset)

📚 **Resources**

* TensorFlow Speech Commands tutorial
* PyTorch audio classification examples

---

## **3. Mini-Project**

### **Speech Feature Extraction Notebook**

**Goal**
Build a reusable notebook that:

* Loads raw audio
* Visualizes waveforms & spectrograms
* Extracts:

  * Spectrogram
  * Mel spectrogram
  * MFCCs
* Compares features across different words

**Deliverables**

* Jupyter Notebook
* Clean visualizations
* Short markdown explanations

**Optional Extensions**

* Noise augmentation
* Silence trimming
* Feature comparison across speakers

---

## **4. Suggested Weekly Schedule**

| Day     | Focus                          |
| ------- | ------------------------------ |
| Day 1   | Audio basics & waveforms       |
| Day 2   | Spectrograms & Mel features    |
| Day 3   | MFCCs + preprocessing pipeline |
| Day 4   | STT concepts & CTC             |
| Day 5   | Keyword spotting model         |
| Day 6–7 | Mini-project                   |

---

## **5. Tools Checklist**

* Python
* Librosa
* NumPy / SciPy
* Matplotlib
* PyTorch or TensorFlow
* Jupyter Notebook

---

Here are **relevant links** you can use 
---

## 📘 **Concepts & Theory (Speech, Spectrograms, MFCCs, CTC, etc.)**

### **Audio Signal & Speech Basics**

* **Introduction to Audio Data (Hugging Face Audio Course)** – covers sampling rate, amplitude, waveform, spectrograms, mel spectrograms, etc.
  👉 [https://huggingface.co/learn/audio-course/en/chapter1/audio_data](https://huggingface.co/learn/audio-course/en/chapter1/audio_data) ([Hugging Face][1])

### **MFCC / STFT / Feature Explanation**

* **Mel-Frequency Cepstral Coefficients (MFCC) for Speech Recognition (GeeksforGeeks)** – explains MFCCs and how they work.
  👉 [https://www.geeksforgeeks.org/nlp/mel-frequency-cepstral-coefficients-mfcc-for-speech-recognition/](https://www.geeksforgeeks.org/nlp/mel-frequency-cepstral-coefficients-mfcc-for-speech-recognition/) ([GeeksforGeeks][2])

* **Audio Deep Learning Made Simple (ASR, CTCloss, and spectrogram explanation)** – plain-English overview covering mel spectrograms, MFCC, CTC’s role in ASR.
  👉 [https://ketanhdoshi.github.io/Audio-ASR/](https://ketanhdoshi.github.io/Audio-ASR/) ([Ketan Doshi Blog][3])

### **Librosa Documentation & Tutorials (Feature Extraction)**

* **Librosa Tutorial (Official docs)** – covers audio loading, feature extraction (spectrogram, mel, MFCC).
  👉 [https://librosa.org/doc/0.11.0/tutorial.html](https://librosa.org/doc/0.11.0/tutorial.html) ([librosa.org][4])

* **Librosa MFCC Reference** – documentation for `librosa.feature.mfcc()` usage.
  👉 [https://librosa.org/doc/main/generated/librosa.feature.mfcc.html](https://librosa.org/doc/main/generated/librosa.feature.mfcc.html) ([librosa.org][5])

* **Librosa Mel Spectrogram Reference** – docs for computing mel spectrograms.
  👉 [https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html](https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html) ([librosa.org][6])

---

## 🧪 **Hands-On Tutorials & Practical Guides**

### **TensorFlow / Speech Commands Keyword Spotting**

* **TensorFlow Simple Audio Recognition (Keyword Spotter Tutorial)** – preprocess audio and train a basic speech/keyword classifier with Speech Commands dataset.
  👉 [https://www.tensorflow.org/tutorials/audio/simple_audio](https://www.tensorflow.org/tutorials/audio/simple_audio) ([TensorFlow][7])

* **GeeksforGeeks — Audio Recognition in TensorFlow** – example including spectrogram feature extraction and classification.
  👉 [https://www.geeksforgeeks.org/audio-recognition-in-tensorflow/](https://www.geeksforgeeks.org/audio-recognition-in-tensorflow/) ([GeeksforGeeks][8])

### **PyTorch / torchaudio Speech Recognition**

* **PyTorch Speech Command Classification (official tutorial)** – how to load data and train a classifier with torchaudio.
  👉 [https://docs.pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html](https://docs.pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html) ([docs.pytorch.org][9])

* **Real Python — TorchAudio Audio Loading & Spectrograms** – shows loading audio and turning it into spectrogram features with torchaudio.
  👉 [https://realpython.com/python-torchaudio/](https://realpython.com/python-torchaudio/) ([realpython.com][10])

* **Medium Tutorial (PyTorch + torchaudio Speech Commands)** – community walkthrough.
  👉 [https://medium.com/@aminul.huq11/speech-command-classification-using-pytorch-and-torchaudio-c844153fce3b](https://medium.com/@aminul.huq11/speech-command-classification-using-pytorch-and-torchaudio-c844153fce3b) ([Medium][11])

---

## 🧪 **Extras (Visualization & Basics)**

* **PostNetwork Academy — Understanding Speech Data with Python & Librosa** – basic waveform and spectrogram plotting in Python.
  👉 [https://www.postnetwork.co/understanding-speech-data-using-python/](https://www.postnetwork.co/understanding-speech-data-using-python/) ([postnetwork.co][12])

---

## 🎯 **Bonus Resources (Videos / Interactive)**

* **YouTube — Mel Spectrograms Explained** *(video)* – visual intuition behind mel scales & spectrograms.
  👉 [https://www.youtube.com/watch?v=9GHCiiDLHQ4](https://www.youtube.com/watch?v=9GHCiiDLHQ4) ([YouTube][13])

* **YouTube — Librosa Audio & Music Signal Analysis (SciPy)** *(classic tutorial)*
  👉 [https://www.youtube.com/watch?v=MhOdbtPhbLU](https://www.youtube.com/watch?v=MhOdbtPhbLU) ([YouTube][14])

---

[1]: https://huggingface.co/learn/audio-course/en/chapter1/audio_data?utm_source=chatgpt.com "Introduction to audio data - Hugging Face Audio Course"
[2]: https://www.geeksforgeeks.org/nlp/mel-frequency-cepstral-coefficients-mfcc-for-speech-recognition/?utm_source=chatgpt.com "Mel-frequency Cepstral Coefficients (MFCC) for Speech Recognition"
[3]: https://ketanhdoshi.github.io/Audio-ASR/?utm_source=chatgpt.com "Automatic Speech Recognition (ASR), How it Works"
[4]: https://librosa.org/doc/0.11.0/tutorial.html?utm_source=chatgpt.com "Tutorial — librosa 0.11.0 documentation"
[5]: https://librosa.org/doc/main/generated/librosa.feature.mfcc.html?utm_source=chatgpt.com "librosa.feature.mfcc — librosa 0.11.0 documentation"
[6]: https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html?utm_source=chatgpt.com "librosa.feature.melspectrogram — librosa 0.11.0 documentation"
[7]: https://www.tensorflow.org/tutorials/audio/simple_audio?utm_source=chatgpt.com "Simple audio recognition: Recognizing keywords  |  TensorFlow Core"
[8]: https://www.geeksforgeeks.org/audio-recognition-in-tensorflow/?utm_source=chatgpt.com "Audio Recognition in Tensorflow - GeeksforGeeks"
[9]: https://docs.pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html?utm_source=chatgpt.com "Speech Command Classification with torchaudio — PyTorch Tutorials 1.13.1+cu117 documentation"
[10]: https://realpython.com/python-torchaudio/?utm_source=chatgpt.com "Use TorchAudio to Prepare Audio Data for Deep Learning – Real Python"
[11]: https://medium.com/%40aminul.huq11/speech-command-classification-using-pytorch-and-torchaudio-c844153fce3b?utm_source=chatgpt.com "Speech Command Classification using PyTorch and torchaudio | by Aminul Huq | Medium"
[12]: https://www.postnetwork.co/understanding-speech-data-using-python/?utm_source=chatgpt.com "Understanding Speech Data using Python - PostNetwork Academy"
[13]: https://www.youtube.com/watch?v=9GHCiiDLHQ4&utm_source=chatgpt.com "Mel Spectrograms Explained Easily - YouTube"
[14]: https://www.youtube.com/watch?v=MhOdbtPhbLU&utm_source=chatgpt.com "Librosa Audio and Music Signal Analysis in Python | SciPy 2015"

