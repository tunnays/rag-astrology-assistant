[README.md](https://github.com/user-attachments/files/31710831/README.md)
# ✨ Local RAG Astrology Assistant

A fully **offline**, on-device Retrieval-Augmented Generation (RAG) application that answers questions about zodiac signs, planets, astrological houses, and planetary aspects. Both the embedding model and the language model run entirely on your computer through [Foundry Local](https://github.com/microsoft/Foundry-Local) — no data ever leaves the device.

## Architecture: 3 Core Files

The project is organized into 3 core Python files, each mapping directly to one stage of the RAG pipeline:

| # | File | Responsibility |
|---|---|---|
| 1 | `veri_hazirlama.py` | **Data preparation** — chunks the source document, generates embeddings, stores them in SQLite |
| 2 | `yapay_zeka_motoru.py` | **AI engine** — hybrid retrieval (semantic + keyword) and answer generation logic |
| 3 | `kullanici_arayuzu.py` | **User interface** — Streamlit-based chat UI |

`yapay_zeka_motoru.py` has no knowledge of any UI framework; `kullanici_arayuzu.py` contains no retrieval/generation logic — it simply calls the engine. This separation makes it easy to swap the interface later (CLI, mobile, API) without touching the core logic.

```
         User Question
               │
               ▼
      Question -> Embedding
               │
               ▼
┌─────────────────────────────┐
│      Hybrid Retrieval        │
│                               │
│  Semantic Similarity          │
│          +                    │
│  Keyword / Exact Matching     │
└──────────────┬────────────────┘
               │
               ▼
        Best Text Chunk
               │
               ▼
      Strong Exact Match?
          /            \
        Yes             No
         │                │
         ▼                ▼
Return Source Text    Phi-3.5-mini
         │                │
         └────────┬───────┘
                  ▼
               Answer
```

## Project Structure

```
.
├── belgeler/
│   └── burclar.txt          # Source astrology knowledge base
│
├── veri_hazirlama.py         # (1/3) Data preparation
├── yapay_zeka_motoru.py      # (2/3) AI engine
├── kullanici_arayuzu.py      # (3/3) User interface
│
├── araclar/                  # Utility / verification scripts (not core deliverables)
│   ├── db_kontrol.py          # Verifies veritabani.db content
│   ├── test_embedding.py      # Tests that the embedding model loads correctly
│   └── test_chat_model.py     # Tests that the chat model loads correctly
│
├── requirements.txt
├── .gitignore
└── veritabani.db              # Generated locally, not included in the repo
```

> Scripts under `araclar/` are not part of the core pipeline; they are helper tools for verifying the setup and debugging.

## How It Works

### 1. Data Preparation (`veri_hazirlama.py`)

Reads `belgeler/burclar.txt` and splits it into chunks based on blank lines. Any chunk longer than `MAX_CHUNK_CHARS` (800 characters) is further split at sentence boundaries. Each chunk is converted into a vector using the `qwen3-embedding-0.6b` model (via Foundry Local) and stored as JSON in `veritabani.db` (SQLite).

### 2. AI Engine (`yapay_zeka_motoru.py`)

**Retrieval:** The user's question is converted into an embedding, and **cosine similarity** is computed against every stored chunk. In addition, the question is scanned with regular expressions for a planet name and a house number (written either numerically or as a word — e.g. "7th house" / "seventh house", and in Turkish "7. ev" / "yedinci ev") to compute a **keyword score**:

```
final_score = semantic_score * 0.70 + keyword_score * 0.30
```

**Generation:** If the top chunk's keyword score is very high (`exact_score >= 0.60`), the source text is returned directly without calling the LLM at all — this eliminates hallucination risk and is faster. Otherwise, the retrieved chunks are passed to `phi-3.5-mini` as context, and the model is instructed to answer **only** using information contained in that context. If the answer cannot be found, it replies with *"I don't have information about this topic."*

This file also exposes the single high-level function the UI calls:

```python
answer, context_chunks = soru_sor(embed_client, chat_client, question, k=1)
```

### 3. User Interface (`kullanici_arayuzu.py`)

A Streamlit app that takes a question from the user, calls `soru_sor()` from `yapay_zeka_motoru.py`, and displays the result along with the retrieval scores of the context chunks used. Models are cached with `st.cache_resource` so they are only loaded once.

## Installation

### 1. Requirements

- Python 3.10+
- [Foundry Local](https://github.com/microsoft/Foundry-Local) installed

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the knowledge database

```bash
python veri_hazirlama.py
```

If the source document changes, run this script again — it deletes old records and rebuilds the database from scratch.

### 4. (Optional) Verify the setup

```bash
python araclar/test_embedding.py     # Tests the embedding model
python araclar/test_chat_model.py    # Tests the chat model
python araclar/db_kontrol.py         # Verifies the database was populated correctly
```

### 5. (Optional) Test the AI engine from the terminal

You can try the engine directly, without the UI:

```bash
python yapay_zeka_motoru.py
```

Typing a question prints `final_score`, `semantic_score`, and `keyword_score` separately — useful for quickly evaluating retrieval quality.

### 6. Run the application

```bash
python -m streamlit run kullanici_arayuzu.py
```

Open the local address shown by Streamlit in your browser (typically `http://localhost:8501`).

## Example Questions

```
Akrep burcunun özellikleri nelerdir?
Venüs neyi temsil eder?
4. evde Mars ne anlama gelir?
7. evde Jüpiter ne anlama gelir?
Yedinci evde Jüpiter ne anlama gelir?
```

## Configuration

The main configuration values live at the top of the relevant file:

| Variable | Location | Purpose | Default |
|---|---|---|---|
| `EMBEDDING_MODEL` | `veri_hazirlama.py`, `yapay_zeka_motoru.py` | Model used for embeddings | `qwen3-embedding-0.6b` |
| `CHAT_MODEL` | `yapay_zeka_motoru.py` | Local language model | `phi-3.5-mini` |
| `MIN_SIMILARITY` | `yapay_zeka_motoru.py` | Minimum semantic similarity threshold | `0.45` |
| `TOP_K` | `yapay_zeka_motoru.py` | Default number of retrieved chunks | `5` |
| `EXACT_MATCH_ESIGI` | `yapay_zeka_motoru.py` | Threshold above which the source text is returned directly | `0.60` |
| `MAX_CHUNK_CHARS` | `veri_hazirlama.py` | Maximum chunk length | `800` |

## Privacy and Local Execution

Document processing, embedding generation, SQLite storage, retrieval, and LLM inference all happen on your machine. No external LLM API is required for normal use once the models have been downloaded.

## Technologies Used

- Python
- Microsoft Foundry Local
- Phi-3.5 Mini
- Qwen3 Embedding 0.6B
- SQLite
- Streamlit
- Cosine Similarity
- Regular Expressions (Regex)

## Current Limitations

- The knowledge base is relatively small.
- Retrieval currently performs a full scan over SQLite rather than using a dedicated vector database; this would need to be revisited for larger datasets.
- Keyword matching is customized specifically for the astrology domain, not general-purpose.
- The quality of generated answers depends on the local language model used.

## Project Purpose

This project was built as an educational exercise to explore the end-to-end implementation of a fully local Retrieval-Augmented Generation system:

```
Document -> Chunking -> Embedding -> SQLite -> Hybrid Retrieval -> Context Selection -> Local LLM -> Answer
```

It particularly focuses on improving retrieval reliability by combining **semantic similarity** with **domain-specific exact matching**.

## Author

**Ayşegül Tuna**

GitHub: `@tunnays`

## License

This project is intended for educational purposes.
