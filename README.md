# 🎓 School Tutor RAG System

A Retrieval-Augmented Generation (RAG) system for school tutoring that answers questions from textbooks in **English** and **Tamil** with structured JSON responses.

## ✨ Features

- 📚 **Multi-subject support**: Science, Maths, English, Social Science, Tamil
- 🌐 **Bilingual**: Supports English and Tamil questions & responses
- 🔢 **Math step-by-step**: Detailed solutions with explanations
- 📊 **Structured output**: JSON responses with summaries, bullet points, and tables
- ⚡ **GPU-accelerated**: Uses FAISS for fast similarity search
- 🎯 **Smart classification**: Auto-detects subject and language

## 🛠️ Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Required Ollama models:
  ```bash
  ollama pull llama3.2
  ollama pull mxbai-embed-large
  ```

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/njk0/tutor_rag.git
   cd tutor_rag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Ollama** (if not already running)
   ```bash
   ollama serve
   ```

## 🚀 Usage

### Step 1: Ingest Documents

First, ingest your PDF documents to create the vector store:

```bash
python -m src.ingest
```

This will:
- Read all PDFs from `data/` folder
- Extract text and metadata
- Create embeddings using `mxbai-embed-large`
- Store vectors in `vector_stores/` folder

### Step 2: Run the Application

**Option A: Streamlit Web UI (Recommended)**
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

**Option B: Command Line Interface**
```bash
python main.py
```

## 📁 Project Structure

```
tutor_rag/
├── src/
│   ├── ingest.py           # Document ingestion pipeline
│   ├── vector_store.py     # FAISS vector store management
│   ├── rag_chain.py        # Main RAG pipeline
│   ├── query_classifier.py # Subject classification
│   ├── language_detector.py# Tamil/English detection
│   ├── metadata_extractor.py # PDF metadata extraction
│   └── output_formatter.py # JSON response formatting
├── data/                   # Place your PDF files here
├── vector_stores/          # Generated vector indexes
├── app.py                  # Streamlit web interface
├── main.py                 # CLI interface
├── config.py               # Configuration settings
└── requirements.txt        # Python dependencies
```

## 📤 JSON Output Format

### General Response (Science, Social Science, English, Tamil)
```json
{
  "summary": "Detailed explanation...",
  "caption": "Topic Title",
  "bullet_points": [{"point": "Key fact 1"}, {"point": "Key fact 2"}],
  "table": [{"header": "Properties", "rows": [{"property": "Name", "value": "Value"}]}],
  "_metadata": {"subject": "Science", "language": "English"}
}
```

### Math Response (Step-by-step)
```json
{
  "problem": "Original problem",
  "caption": "Problem Type",
  "steps": [
    {"step_number": 1, "action": "...", "explanation": "...", "expression": "...", "result": "..."}
  ],
  "final_answer": "42",
  "concept_used": ["Algebra"],
  "tips": ["Helpful tip"]
}
```

## 🌐 Language Support

Ask questions in Tamil by:
- Writing in Tamil: `பக்தி இயக்கம் பற்றி விளக்குக`
- Or adding "in tamil": `explain bhakti movement in tamil`

## ⚙️ Configuration

Edit `config.py` to customize:
- `LLM_MODEL`: Default is `llama3.2`
- `EMBEDDING_MODEL`: Default is `mxbai-embed-large`
- `CHUNK_SIZE`: Text chunk size (default: 500)
- `TOP_K_RESULTS`: Number of documents to retrieve (default: 5)

## 📝 Adding New Documents

1. Place PDF files in the `data/` folder
2. Name files with subject prefix: `Science_Chapter1.pdf`, `Maths_Algebra.pdf`
3. Re-run ingestion: `python -m src.ingest`

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama not running | Run `ollama serve` in terminal |
| Model not found | Run `ollama pull llama3.2` |
| Empty responses | Check if `vector_stores/` has data |
| GPU not used | Install `faiss-gpu` instead of `faiss-cpu` |

## 📄 License

MIT License
