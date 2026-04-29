import requests
from pathlib import Path
from datetime import datetime
import re
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/generate"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHROMA_DIR = Path("./chroma_db")
CHROMA_DIR.mkdir(exist_ok=True)

def get_video_id(url):
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Invalid YouTube URL: {url}")

def get_transcript(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=['en'])
    except Exception:
        transcript_list = api.list(video_id)
        first = next(iter(transcript_list))
        fetched = first.fetch()
    result = []
    for item in fetched:
        if hasattr(item, 'text'):
            result.append({'text': item.text})
        elif isinstance(item, dict):
            result.append({'text': item.get('text', '')})
        else:
            result.append({'text': str(item)})
    return result

def get_video_info(youtube_url):
    """Get video title and duration safely"""
    try:
        from yt_dlp import YoutubeDL
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "format": "best",
            "noplaylist": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info.get("title", "Unknown"), info.get("duration", 0)
    except Exception:
        # Fallback - just extract title from URL
        video_id = get_video_id(youtube_url)
        return f"Video {video_id}", 0

def extract_transcript(youtube_url):
    video_id = get_video_id(youtube_url)
    transcript_items = get_transcript(video_id)
    full_text = " ".join([item["text"] for item in transcript_items])
    title, duration = get_video_info(youtube_url)
    return {
        "video_id": video_id,
        "title": title,
        "duration": duration,
        "transcript": full_text,
        "word_count": len(full_text.split()),
        "extracted_at": datetime.now().isoformat()
    }

def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i: i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

@st.cache_resource
def load_model_and_db():
    from sentence_transformers import SentenceTransformer
    import chromadb
    model = SentenceTransformer("all-MiniLM-L6-v2")
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return model, db

def store_embeddings(video_id, chunks, title):
    model, db = load_model_and_db()
    embeddings = model.encode(chunks).tolist()
    collection = db.get_or_create_collection(name=f"video_{video_id}")
    ids = [f"{video_id}_chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=[{"video_id": video_id, "title": title} for _ in chunks])

def search_chunks(video_id, query):
    model, db = load_model_and_db()
    collection = db.get_collection(name=f"video_{video_id}")
    query_embedding = model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return results["documents"][0] if results["documents"] else []

def ask_ollama(prompt):
    try:
        response = requests.post(OLLAMA_URL, json={"model": "llama2", "prompt": prompt, "stream": False}, timeout=120)
        if response.status_code == 200:
            return response.json().get("response", "No response")
        return f"HTTP Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ Ollama not running! Run: ollama serve"
    except Exception as e:
        return f"❌ Error: {str(e)}"

st.set_page_config(page_title="YouTube Q&A", layout="wide")
st.title("📺 YouTube Video Q&A")
st.markdown("**Fully Local • Zero API Costs • Runs on Your Mac**")

with st.sidebar:
    st.header("📥 Load a Video")
    url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("📥 Load & Process", use_container_width=True):
        if url:
            try:
                with st.spinner("⏳ Extracting transcript..."):
                    data = extract_transcript(url)
                st.session_state.video_data = data
                st.success(f"✓ {data['title']}")
                with st.spinner("⏳ Creating embeddings..."):
                    chunks = chunk_text(data["transcript"])
                    store_embeddings(data["video_id"], chunks, data["title"])
                st.session_state.video_id = data["video_id"]
                st.session_state.chunks = len(chunks)
                st.success(f"✓ Ready! ({len(chunks)} chunks)")
            except Exception as e:
                st.error(f"❌ {str(e)}")
        else:
            st.warning("Enter a YouTube URL first!")

if "video_id" not in st.session_state:
    st.info("👈 Paste a YouTube URL in the sidebar to get started!")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["Q&A", "Summary", "Transcript", "Info"])
    with tab1:
        st.subheader("Ask a Question")
        question = st.text_input("Your question:", placeholder="What is this video about?")
        if st.button("🤖 Get Answer", use_container_width=True):
            if question:
                with st.spinner("🤖 Thinking... (10-30 seconds)"):
                    chunks = search_chunks(st.session_state.video_id, question)
                    context = "\n---\n".join(chunks)
                    prompt = f"Answer this question based only on the transcript below.\nQuestion: {question}\nTranscript:\n{context}\nAnswer:"
                    answer = ask_ollama(prompt)
                st.markdown("### Answer")
                st.write(answer)
                with st.expander("📍 Sources"):
                    for i, c in enumerate(chunks, 1):
                        st.write(f"**Chunk {i}:** {c[:200]}...")
            else:
                st.warning("Enter a question!")
    with tab2:
        st.subheader("Generate Summary")
        if st.button("✨ Summarize", use_container_width=True):
            with st.spinner("Summarizing..."):
                prompt = f"Summarize this video in 3-4 paragraphs:\n{st.session_state.video_data['transcript'][:3000]}\nSummary:"
                st.write(ask_ollama(prompt))
    with tab3:
        st.subheader("Full Transcript")
        data = st.session_state.video_data
        col1, col2 = st.columns(2)
        col1.metric("Words", f"{data['word_count']:,}")
        col2.metric("Chunks", st.session_state.chunks)
        st.text_area("", value=data["transcript"], height=400)
    with tab4:
        st.subheader("Video Info")
        data = st.session_state.video_data
        col1, col2 = st.columns(2)
        col1.metric("Title", data["title"])
        col1.metric("Duration", f"{data['duration'] // 60} min")
        col2.metric("Words", f"{data['word_count']:,}")
        col2.metric("Processed", data["extracted_at"][:10])
