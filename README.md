# 📚 Chat PDF Intelligent

Interact with multiple PDF files using powerful AI models like **Gemini 1.5 (Google AI)** to extract insights, analyze financial data, and answer questions based on uploaded documents. This app is especially useful for analyzing **annual reports** and **financial statements** of Indian stock market companies.

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange?style=flat-square&logo=streamlit)

---

## 🚀 Features

- 📄 Upload multiple PDF files
- 🤖 Ask questions based on the content of the PDFs
- 🧠 LangChain avec LLM HuggingFace FLAN‑T5 via transformers
- 🗃️ Embeddings `all-MiniLM-L6-v2` et index FAISS local
- 📊 Specialized for analyzing financial reports, related-party transactions, and remuneration
- 🗨️ Chat-like interface with user/bot avatars
- 📥 Export conversation history as CSV

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rakshithsantosh/pdf-chatbot-gemini.git
cd pdf-chatbot-gemini
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using uv (recommended)
uv sync
```

### 4. Run the App

```bash
streamlit run app.py

### 5. Advanced App

streamlit run app_advanced.py
```

---

## 🔐 Google AI API Key

To use Gemini models and embeddings:

1. Visit [Google AI Studio](https://ai.google.dev/)
2. Generate your API key
3. Enter the key in the **Streamlit sidebar**

---

## 📦 Tech Stack

| Tech       | Purpose                                  |
| ---------- | ---------------------------------------- |
| Streamlit  | UI framework for interactive web apps    |
| LangChain  | Managing LLM chains and embeddings       |
| Gemini 1.5 | Large Language Model (via Google AI API) |
| PyPDF2     | PDF text extraction                      |
| FAISS      | Vector database for similarity search    |
| Pandas     | Exporting conversation as CSV            |
| HTML/CSS   | Custom chat UI inside Streamlit          |

---

## 📁 File Structure

```
├── app.py
├── app_advanced.py
├── advanced_rag.py
├── adaptive_chunking.py
├── multimodal_extractor.py
├── faiss_index/
├── requirements.txt
├── GUIDE_INSTALLATION.md
└── README.md
```

---

## 🧠 Architecture Avancée

Système RAG Avancé:
- Cache de requêtes avec TTL
- Reranking par cross‑encoder
- Fusion de requêtes par RRF
- Orchestration complète et métriques

Chunking Intelligent:
- Chunking adaptatif par type de document
- Mode hybride multi‑stratégies
- Optimisation de segments

Multi‑modalité:
- Extraction d’images
- OCR Tesseract
- Tables Camelot/Tabula
- Formats DOCX/PPTX/XLSX

- Evaluate financial statements from PDFs
- Detect irregularities or red flags
- Analyze related party transactions
- Identify unusual managerial remuneration

---

## 🧪 Quick Start Avancé

1. Lancer `app_advanced.py`
2. Assurer l’index via la sidebar
3. Activer RRF et Rerank si nécessaire
4. Ajuster TTL et k

---

## 👤 Author

- [Rakshith Santosh](https://www.linkedin.com/in/rak-99-s)
- [GitHub](https://github.com/rakshithsantosh)

---

## 📄 License

MIT License – Feel free to use, modify, and share!
