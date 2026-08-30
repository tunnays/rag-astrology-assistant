# ✨ Local RAG Astrology Assistant

A fully local **Retrieval-Augmented Generation (RAG)** application for answering questions about zodiac signs, planets, astrological houses, planetary aspects, and horoscope-related information.

The project uses **Microsoft Foundry Local** to run both the embedding model and the language model directly on the user's computer. The knowledge base is stored locally using **SQLite**, and the application provides an interactive interface built with **Streamlit**.

The main goal of this project is to demonstrate how a small, domain-specific RAG system can combine semantic search, keyword matching, local data storage, and local LLM inference to produce relevant and reliable answers.

---

## 🚀 Features

- 📚 Uses a custom astrology knowledge base containing information about:
  - Zodiac signs
  - Rising signs
  - Planets
  - Astrological houses
  - Planetary aspects
  - Daily, weekly, and monthly horoscope examples

- 🧩 Splits the source document into smaller text chunks before embedding.

- 🧠 Generates embeddings locally using:
  - `qwen3-embedding-0.6b`

- 💾 Stores text chunks and their embeddings in a local **SQLite** database.

- 🔍 Uses a **hybrid retrieval system** combining:
  - Semantic similarity using cosine similarity
  - Keyword and exact matching
  - Planet matching
  - Astrological house number matching

- 🔢 Understands house numbers written both numerically and in words:
  - `7th house`
  - `seventh house`
  
  The Turkish interface also supports forms such as `7. ev` and `yedinci ev`.

- ⚡ Returns the source text directly when a strong exact match is found, reducing unnecessary LLM generation and hallucination risk.

- 💬 Uses `phi-3.5-mini` through Foundry Local for questions that require natural-language generation.

- 🖥️ Includes a Streamlit-based user interface.

- 🔒 Runs locally. User data and prompts do not need to be sent to an external LLM API.

---

## 🏗️ Project Architecture

The application follows a standard RAG pipeline:

```text
                 User Question
                       │
                       ▼
              Question Embedding
                       │
                       ▼
        ┌─────────────────────────────┐
        │      Hybrid Retrieval       │
        │                             │
        │  Semantic Similarity        │
        │          +                  │
        │  Keyword / Exact Matching   │
        └──────────────┬──────────────┘
                       │
                       ▼
               Best Text Chunk
                       │
                       ▼
              Strong Exact Match?
                  /           \
                Yes            No
                 │              │
                 ▼              ▼
        Return Source Text    Phi-3.5-mini
                 │              │
                 └──────┬───────┘
                        ▼
                     Answer
```

---

## 📁 Project Structure

```text
.
├── belgeler/
│   └── burclar.txt
│
├── ingest.py
├── retrieval.py
├── rag_app.py
├── streamlit_app.py
├── db_kontrol.py
├── test_embedding.py
├── requirements.txt
├── .gitignore
└── veritabani.db        # Generated locally, not included in Git
```

### Main Files

**`belgeler/burclar.txt`**  
Contains the custom astrology knowledge base used by the RAG system.

**`ingest.py`**  
Reads the source document, splits it into chunks, generates embeddings, and stores them in SQLite.

**`retrieval.py`**  
Implements the hybrid retrieval system using semantic similarity and keyword/exact matching.

**`rag_app.py`**  
Main Streamlit application. It uses the advanced hybrid retrieval system and direct-answer fallback for strong exact matches.

**`streamlit_app.py`**  
A simpler Streamlit implementation based primarily on cosine similarity retrieval.

**`db_kontrol.py`**  
Utility script for checking whether the SQLite database was created correctly.

**`test_embedding.py`**  
Tests whether the embedding model can be loaded and used successfully.

---

## 🧠 How the RAG System Works

### 1. Document Chunking

The source file:

```text
belgeler/burclar.txt
```

is divided into smaller text chunks.

Paragraphs separated by blank lines are treated as natural chunks. If a paragraph exceeds the maximum chunk size, it is divided into smaller sections.

The default maximum size is:

```python
MAX_CHUNK_CHARS = 800
```

---

### 2. Embedding Generation

Each chunk is converted into a numerical vector using:

```text
qwen3-embedding-0.6b
```

The embedding model runs locally through **Foundry Local**.

---

### 3. SQLite Storage

Each chunk is stored together with its embedding in:

```text
veritabani.db
```

The database contains the text, embedding, and source information.

The database is generated locally and is intentionally excluded from the GitHub repository.

---

### 4. Hybrid Retrieval

A user question is also converted into an embedding.

The system calculates the cosine similarity between the question embedding and every stored text chunk.

However, semantic similarity alone may not always correctly distinguish structured concepts such as:

```text
1st house
4th house
7th house
11th house
```

For this reason, the advanced retrieval system also performs exact keyword matching.

The final retrieval score is calculated as:

```text
final_score =
    semantic_score × 0.70
    +
    keyword_score × 0.30
```

This allows the system to combine the flexibility of semantic search with the precision of structured keyword matching.

---

## 🎯 Exact-Match Retrieval

The retrieval system detects planet names such as:

```text
Sun
Moon
Mercury
Venus
Mars
Jupiter
Saturn
Uranus
Neptune
Pluto
```

It also detects astrological house numbers.

For example, the following Turkish questions are treated as equivalent:

```text
7. evde Jüpiter ne anlama gelir?
```

and:

```text
Yedinci evde Jüpiter ne anlama gelir?
```

When both the planet and house number match the source document, the system assigns an additional keyword score.

This significantly improves retrieval accuracy for structured astrology questions.

---

## ⚡ Direct Answer Fallback

Small language models may occasionally paraphrase a correct retrieved passage incorrectly.

To reduce this problem, `rag_app.py` uses a simple reliability mechanism.

If the best retrieved chunk has a strong exact-match score:

```python
exact_score >= 0.60
```

the application returns the original source text directly instead of asking the LLM to rewrite it.

For example:

```text
Question:
7. evde Jüpiter ne anlama gelir?

Retrieved context:
7. Evde Jüpiter: 7. evde Jüpiter evlilik ve ortaklıklarda
şans, destek ve büyüme vaat eder.
```

Because the house number and planet match strongly, the source passage can be returned directly.

This approach helps reduce hallucination and preserves the meaning of the original knowledge base.

---

## 🤖 Local Language Model

When an exact match is not strong enough for direct extraction, the retrieved context is passed to:

```text
phi-3.5-mini
```

The model is instructed to answer only using information contained in the retrieved context.

If the answer cannot be found in the context, the application can respond:

```text
Bu konuda elimde bilgi yok.
```

which means:

> "I don't have information about this topic."

---

## 🛠️ Installation

### 1. Requirements

You need:

- Python 3.10+
- Microsoft Foundry Local
- Git
- A system capable of running the selected local models

---

### 2. Clone the Repository

```bash
git clone https://github.com/tunnays/rag-astrology-assistant.git
cd rag-astrology-assistant
```

---

### 3. Create a Virtual Environment

On Windows:

```bash
python -m venv venv
```

Activate it with:

```powershell
.\venv\Scripts\Activate.ps1
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Build the Knowledge Database

Before running the application for the first time, execute:

```bash
python ingest.py
```

This script:

1. Reads `belgeler/burclar.txt`
2. Splits the document into chunks
3. Loads `qwen3-embedding-0.6b`
4. Generates an embedding for each chunk
5. Creates the SQLite database
6. Stores the chunks and embeddings in `veritabani.db`

If the source document is modified, run `ingest.py` again to rebuild the database.

---

## 🧪 Test the Embedding Model

To verify that the embedding model works correctly:

```bash
python test_embedding.py
```

---

## 🗄️ Check the Database

After running the ingestion process:

```bash
python db_kontrol.py
```

This can be used to verify that the chunks and embeddings were successfully stored.

---

## 🔎 Test Retrieval from the Terminal

The retrieval system can be tested independently:

```bash
python retrieval.py
```

Example question:

```text
Yedinci evde Jüpiter ne anlama gelir?
```

The program displays:

```text
Final Score
Semantic Score
Keyword Score
Retrieved Text
```

This is useful for debugging and evaluating retrieval quality separately from LLM generation.

---

## 🖥️ Run the Streamlit Application

The recommended application is:

```bash
python -m streamlit run rag_app.py
```

Then open the local address displayed by Streamlit in your browser.

For example:

```text
http://localhost:8501
```

---

## 💡 Example Questions

The application can answer questions such as:

```text
Akrep burcunun özellikleri nelerdir?

Venüs neyi temsil eder?

4. evde Mars ne anlama gelir?

7. evde Jüpiter ne anlama gelir?

Yedinci evde Jüpiter ne anlama gelir?
```

---

## ⚙️ Configuration

The main configuration values include:

| Variable | Purpose | Default |
|---|---|---|
| `EMBEDDING_MODEL` | Model used for embeddings | `qwen3-embedding-0.6b` |
| `CHAT_MODEL` | Local language model | `phi-3.5-mini` |
| `MIN_SIMILARITY` | Minimum semantic similarity | `0.45` |
| `TOP_K` | Maximum number of retrieved chunks | `5` |
| `MAX_CHUNK_CHARS` | Maximum chunk length | `800` |

The advanced Streamlit application can use a smaller `k` value when only the highest-ranking context is needed.

---

## 🔒 Privacy and Local Execution

One of the main goals of this project is local execution.

The following operations are performed locally:

- Document processing
- Embedding generation
- SQLite storage
- Semantic retrieval
- Keyword matching
- LLM inference

No external LLM API is required for normal inference after the required models have been downloaded.

---

## 🧰 Technologies Used

- **Python**
- **Microsoft Foundry Local**
- **Phi-3.5-mini**
- **Qwen3 Embedding 0.6B**
- **SQLite**
- **Streamlit**
- **Cosine Similarity**
- **Regular Expressions (Regex)**
- **Retrieval-Augmented Generation (RAG)**

---

## 📌 Current Limitations

This is a small educational RAG project, so there are several areas that could be improved:

- The knowledge base is relatively small.
- SQLite retrieval currently scans the stored chunks rather than using a dedicated vector database.
- Keyword matching is customized for the astrology domain.
- The quality of generated answers depends on the local language model.
- More evaluation questions could be added to measure retrieval accuracy systematically.

---

## 🔮 Future Improvements

Possible future improvements include:

- Adding more astrology documents to the knowledge base
- Supporting additional structured astrology concepts
- Adding chat history
- Adding source citations to generated answers
- Creating an automated RAG evaluation dataset
- Comparing different local embedding models
- Comparing different local LLMs
- Improving the Streamlit interface
- Adding configurable retrieval parameters
- Adding multilingual queries
- Replacing full SQLite scanning with a dedicated vector search solution for larger datasets

---

## 🎓 Project Purpose

This project was developed as an educational project to explore the practical implementation of a **local Retrieval-Augmented Generation system**.

It demonstrates the complete RAG workflow:

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
SQLite
   ↓
Hybrid Retrieval
   ↓
Context Selection
   ↓
Local LLM
   ↓
Answer
```

The project particularly focuses on improving retrieval reliability by combining **semantic similarity** with **domain-specific exact matching**.

---

## 👩‍💻 Author

**Ayşegül Tuna**

GitHub: `@tunnays`

---

## 📄 License

This project is intended for educational purposes.