# 🧠 SurgiGuide – PDF-Based Surgical Knowledge Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-blue)](https://docs.langchain.com/)
[![Model: FLAN-T5](https://img.shields.io/badge/Model-FLAN--T5--Small-yellow)](https://huggingface.co/google/flan-t5-small)

---

**SurgiGuide** is a lightweight Retrieval-Augmented Generation (RAG) assistant that answers surgical and medical queries by retrieving information from a local PDF (e.g., instruction manuals, medical guides). It uses LangChain with Sentence Transformers for chunk embedding, ChromaDB for vector storage, and FLAN-T5 as the generative model.

---

## 🚀 Features

- 🔍 **Ask questions** from any medical PDF
- 📑 **Preprocessing** and **chunking** for optimized retrieval
- 🤖 Built with HuggingFace `FLAN-T5` and LangChain
- 🧠 Embeds using `all-MiniLM-L6-v2`
- 💾 Persistent vector storage with `ChromaDB`
- ✅ Clean input and formatted bullet-point answers
- 🖥️ Terminal-based interactive chatbot

---

## 📂 Project Structure
### SurgiGuide/

- Instructions.pdf # Input medical instruction PDF
- surgiguides_db/ # ChromaDB vector storage
- genai_model4.py # Main RAG pipeline script
- requirements.txt # Python dependencies
- README.md # You are here

---

## 🛠️ How it Works

### 1. Load & Chunk PDF
- Loads the PDF using `PyMuPDFLoader`
- Chunks it with `RecursiveCharacterTextSplitter`

### 2. Embed & Store
- Uses `all-MiniLM-L6-v2` SentenceTransformer to embed chunks
- Stores vectors in local `ChromaDB`

### 3. Build LLM Pipeline
- Loads `FLAN-T5` via HuggingFace pipeline
- Chains with LangChain’s `RetrievalQA`

### 4. Answer Queries
- Preprocesses input
- Retrieves relevant chunks
- Generates answer via LLM
- Postprocesses the response for readability

---

## 🧪 Example Interaction

```bash
you: What is abdominal surgery?

 Generating Your answer...

📘 Surgiguide says:
Abdominal surgery is a procedure involving the abdominal cavity. It includes operations on organs such as the stomach, intestines, liver, or kidneys.
```

---

## 📦 Installation

``` bash
git clone https://github.com/yourusername/surgiguide.git
cd surgiguide
pip install -r requirements.txt

⚠️ Ensure Instructions.pdf is present in the root directory before running.
```

---

## ▶️ Run the Application

```bash
python genai_model4.py

To exit, type: exit or quit.
```

## 🔧 Dependencies

```bash
pip install langchain pymupdf transformers sentence-transformers chromadb
```

---

<p align="center">
  Made with ❤️ for patients, doctors, and developers by <strong>Ashley Mathias</strong>
</p>



