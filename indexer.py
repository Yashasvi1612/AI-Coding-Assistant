"""
indexer.py
----------
Clones a GitHub repo (or reads a local folder) and builds a FAISS index
over its code files, so the agent can retrieve relevant context when
answering questions about the repository.
"""

import os
import subprocess
import tempfile
import shutil
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
    ".hpp", ".go", ".rs", ".rb", ".php", ".cs", ".md", ".txt", ".json",
    ".yaml", ".yml",
}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_FILE_SIZE = 200_000  # skip huge generated files

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def clone_repo(repo_url, dest_dir=None):
    """Shallow-clones a public GitHub repo into a temp dir."""
    dest_dir = dest_dir or tempfile.mkdtemp(prefix="repo_")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, dest_dir],
        check=True,
        capture_output=True,
        text=True,
    )
    return dest_dir


def collect_files(root_dir):
    """Walk the repo, return list of (relative_path, content)."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "venv", "__pycache__")]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in CODE_EXTENSIONS:
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                if os.path.getsize(fpath) > MAX_FILE_SIZE:
                    continue
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                rel_path = os.path.relpath(fpath, root_dir)
                files.append((rel_path, content))
            except Exception:
                continue
    return files


def chunk_file(rel_path, content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk a single file's content, tagging each chunk with its source path."""
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        piece = content[start:end].strip()
        if piece:
            chunks.append({"path": rel_path, "text": piece})
        start += chunk_size - overlap
    return chunks


def build_repo_index(repo_url_or_path, is_local=False):
    """
    Clones (or reads) a repo and builds a FAISS index over its chunks.
    Returns (index, chunk_records, repo_dir, stats_message)
    """
    if is_local:
        repo_dir = repo_url_or_path
    else:
        repo_dir = clone_repo(repo_url_or_path)

    files = collect_files(repo_dir)
    if not files:
        return None, [], repo_dir, "No indexable source files found."

    all_chunks = []
    for rel_path, content in files:
        all_chunks.extend(chunk_file(rel_path, content))

    if not all_chunks:
        return None, [], repo_dir, "No text extracted from repo files."

    embedder = get_embedder()
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    stats = f"Indexed {len(files)} files / {len(all_chunks)} chunks."
    return index, all_chunks, repo_dir, stats


def retrieve_context(query, index, chunks, k=6):
    """Return top-k chunk records most relevant to the query."""
    if index is None or not chunks:
        return []
    embedder = get_embedder()
    q_emb = embedder.encode([query]).astype("float32")
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


def cleanup_repo(repo_dir):
    try:
        shutil.rmtree(repo_dir, ignore_errors=True)
    except Exception:
        pass