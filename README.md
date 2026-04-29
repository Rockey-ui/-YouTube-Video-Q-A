# 📺 YouTube Video Q&A — Local RAG Application

> **Fully Local • Zero API Costs • Runs on Your Mac**

---

## 1. What the Code Does

This application lets you ask questions about any YouTube video using only your local machine — no OpenAI key, no cloud costs. It works by:

1. **Fetching the transcript** of a YouTube video (via captions/subtitles)
2. **Chunking** the transcript into smaller overlapping text segments
3. **Embedding** each chunk using a local sentence-transformer model and storing them in a local vector database (ChromaDB)
4. **Retrieving** the most relevant chunks when you ask a question (semantic search)
5. **Generating an answer** by sending those chunks as context to a locally running LLM via [Ollama](https://ollama.ai)

In short, it is a **Retrieval-Augmented Generation (RAG)** pipeline built entirely on local, free tools, with a Streamlit web UI.

Key features:
- **Q&A tab** — ask any question about the video
- **Summary tab** — auto-generate a multi-paragraph summary
- **Transcript tab** — view the full extracted transcript
- **Info tab** — see video metadata (title, duration, word count, etc.)

---

## 2. Code Structure

```
youtube_qa_simple.py          # Single-file Streamlit application
chroma_db/                    # Auto-created: persisted vector embeddings
```

### Function Breakdown

| Function | Purpose |
|---|---|
| `get_video_id(url)` | Parses a YouTube URL (standard or short form) and extracts the video ID |
| `get_transcript(video_id)` | Uses `youtube-transcript-api` to fetch English captions; falls back to the first available language |
| `get_video_info(youtube_url)` | Uses `yt-dlp` to extract the video title and duration |
| `extract_transcript(youtube_url)` | Orchestrates transcript + metadata extraction, returns a combined dict |
| `chunk_text(text)` | Splits transcript into 500-word chunks with 100-word overlap for context continuity |
| `load_model_and_db()` | Loads the `all-MiniLM-L6-v2` sentence-transformer and initialises ChromaDB (cached by Streamlit) |
| `store_embeddings(video_id, chunks, title)` | Encodes all chunks and upserts them into a per-video ChromaDB collection |
| `search_chunks(video_id, query)` | Encodes the question and returns the top-3 most similar chunks via vector search |
| `ask_ollama(prompt)` | POSTs a prompt to a locally running Ollama server and returns the LLM response |
| *(Streamlit UI block)* | Sidebar for URL input; four tabs for Q&A, Summary, Transcript, and Info |

### Data / Config Constants

| Constant | Value | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Local Ollama API endpoint |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `100` | Overlapping words between consecutive chunks |
| `CHROMA_DIR` | `./chroma_db` | Persistent vector store location |

---

## 3. How to Prepare to Run

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed on your machine

### Step 1 — Install Python dependencies

```bash
pip install streamlit \
            requests \
            youtube-transcript-api \
            yt-dlp \
            sentence-transformers \
            chromadb
```

### Step 2 — Install and start Ollama

```bash
# macOS / Linux
brew install ollama          # or download from https://ollama.ai

# Pull the LLaMA 2 model (one-time download, ~3.8 GB)
ollama pull llama2

# Start the Ollama server (keep this running in a separate terminal)
ollama serve
```

> **Note:** Ollama must be running on `http://localhost:11434` before you launch the app. You can verify it with `curl http://localhost:11434`.

### Step 3 — (Optional) GPU / performance note

The sentence-transformer embedding model (`all-MiniLM-L6-v2`) runs on CPU by default and is fast. The LLaMA 2 inference via Ollama will use Apple Silicon GPU (Metal) automatically on M-series Macs, giving significantly faster response times.

---

## 4. How to Run

```bash
# From the directory containing youtube_qa_simple.py
streamlit run youtube_qa_simple.py
```

Streamlit will open your browser automatically at `http://localhost:8501`.

### Usage walkthrough

1. Paste a YouTube URL into the **sidebar** (e.g. `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
2. Click **📥 Load & Process** — the app will extract the transcript and build the vector index
3. Switch to the **Q&A** tab, type your question, and click **🤖 Get Answer**
4. Optionally use the **Summary** tab for an overview or the **Transcript** tab to read the raw text

### Troubleshooting

| Error | Fix |
|---|---|
| `❌ Ollama not running!` | Run `ollama serve` in a separate terminal |
| `Invalid YouTube URL` | Make sure the URL contains `watch?v=` or is a `youtu.be` short link |
| Transcript not found | The video may have no captions; try a video with auto-generated subtitles enabled |
| Slow answers | Normal for CPU-only machines; answers typically take 10–60 seconds depending on hardware |

---

## Dependencies Summary

| Library | Role |
|---|---|
| `streamlit` | Web UI framework |
| `youtube-transcript-api` | Fetch YouTube captions |
| `yt-dlp` | Extract video metadata (title, duration) |
| `sentence-transformers` | Local text embedding model |
| `chromadb` | Local vector database |
| `requests` | HTTP calls to Ollama |
| `ollama` (server) | Local LLM inference engine |
