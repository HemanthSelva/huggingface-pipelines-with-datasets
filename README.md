# 📘 HuggingFace Pipelines with Datasets

## 📌 Overview
This project demonstrates **8 different Natural Language Processing (NLP) tasks** using Hugging Face Transformers pipelines along with real-world datasets from the `datasets` library.

It showcases how to quickly apply pretrained models for various NLP applications without building models from scratch.

---

## 👨‍💻 Author
**HEMANTHSELVA A K**  
**Company:** Sourcesys Technologies  

---

## 🚀 Features / Tasks Covered

1. **Question Answering**
   - Model: `distilbert-base-cased-distilled-squad`
   - Dataset: SQuAD  

2. **Token Classification (NER)**
   - Model: `bert-large-cased-finetuned-conll03-english`
   - Dataset: CoNLL-2003  

3. **Text Classification (Sentiment Analysis)**
   - Model: `distilbert-base-uncased-finetuned-sst-2-english`
   - Dataset: IMDB  

4. **Zero-Shot Classification**
   - Model: `facebook/bart-large-mnli`
   - Dataset: AG News  

5. **Summarization**
   - Model: `distilbart-cnn-12-6`
   - Dataset: CNN/DailyMail  

6. **Text Generation**
   - Model: `gpt2`
   - Dataset: WikiText-2  

7. **Sentence Similarity**
   - Model: `all-MiniLM-L6-v2`
   - Dataset: SICK  

8. **Feature Extraction**
   - Model: `distilbert-base-uncased`
   - Dataset: Emotion  

---

## 🛠️ Installation

pip install transformers torch sentence-transformers datasets

---

## ▶️ How to Run

python huggingface_pipelines_with_datasets.py

---

## 📂 Project Structure

.
├── huggingface_pipelines_with_datasets.py
└── README.md

---

## ⚠️ Notes

- Internet connection required for downloading models
- First run may take time
- Some datasets may require `trust_remote_code=True`

---

## 💡 Learning Outcomes

- Understanding Hugging Face pipelines
- Working with NLP datasets
- Using pretrained models effectively
