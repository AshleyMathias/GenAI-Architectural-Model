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

