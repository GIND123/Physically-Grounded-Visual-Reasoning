"""
RAG knowledge base — chunk SOPs/FMEAs, embed with sentence-transformers, store in ChromaDB.

The knowledge base is queried during hypothesis generation to give GPT-4o
relevant manufacturing context (failure mechanisms, corrective actions).
"""

import os
import json
from config.settings import KB_DIR, SOP_DIR, FMEA_DIR, RAG_DIR, CFG_DIR, DEVICE


_DEFECT_KEYWORDS = [
    "crack", "scratch", "dent", "contamination", "hole", "cut", "fold",
    "bent", "broken", "missing", "color", "glue", "thread", "poke",
    "squeeze", "misplaced", "damaged", "flip", "rough", "oil", "liquid",
]


def chunk_document(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words  = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks


def extract_metadata_from_chunk(chunk: str, filename: str, doc_type: str) -> dict:
    """Tag chunk with category, doc type, and mentioned defect keywords."""
    category = (
        filename
        .replace("_sop.md", "")
        .replace("_fmea.md", "")
        .replace(".md", "")
        .replace(".json", "")
    )
    mentioned = [k for k in _DEFECT_KEYWORDS if k in chunk.lower()]
    return {
        "category":         category,
        "doc_type":         doc_type,
        "defects_mentioned": ",".join(mentioned) if mentioned else "general",
        "source_file":      filename,
    }


def build_knowledge_base(rag_dir: str = None, force: bool = False):
    """
    Build (or rebuild) the ChromaDB knowledge base from SOPs, FMEAs, and
    process context configs.

    Returns:
        (chroma_collection, embedder, total_chunks)
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    db_dir = rag_dir or RAG_DIR
    os.makedirs(db_dir, exist_ok=True)

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE.type)

    chroma_client = chromadb.PersistentClient(path=db_dir)

    if force:
        try:
            chroma_client.delete_collection("industrial_kb")
        except Exception:
            pass

    try:
        collection = chroma_client.get_collection("industrial_kb")
        if collection.count() > 0 and not force:
            print(f"  KB already has {collection.count()} chunks — skipping rebuild")
            return collection, embedder, collection.count()
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="industrial_kb",
        metadata={"hnsw:space": "cosine"},
    )

    doc_id = 0
    total  = 0

    def _add(chunks, meta_fn, prefix):
        nonlocal doc_id, total
        for chunk in chunks:
            meta      = meta_fn(chunk)
            embedding = embedder.encode(chunk).tolist()
            collection.add(
                ids=[f"{prefix}_{doc_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[meta],
            )
            doc_id += 1
            total  += 1

    # SOPs
    if os.path.exists(SOP_DIR):
        for fname in sorted(os.listdir(SOP_DIR)):
            if not fname.endswith(".md"):
                continue
            with open(os.path.join(SOP_DIR, fname)) as f:
                text = f.read()
            chunks = chunk_document(text)
            _add(chunks, lambda c, fn=fname: extract_metadata_from_chunk(c, fn, "sop"), "sop")
            print(f"  SOP {fname}: {len(chunks)} chunks")

    # FMEAs
    if os.path.exists(FMEA_DIR):
        for fname in sorted(os.listdir(FMEA_DIR)):
            if not fname.endswith(".md"):
                continue
            with open(os.path.join(FMEA_DIR, fname)) as f:
                text = f.read()
            chunks = chunk_document(text)
            _add(chunks, lambda c, fn=fname: extract_metadata_from_chunk(c, fn, "fmea"), "fmea")
            print(f"  FMEA {fname}: {len(chunks)} chunks")

    # General defect reference
    defects_txt = os.path.join(KB_DIR, "defects.txt")
    if os.path.exists(defects_txt):
        with open(defects_txt) as f:
            text = f.read()
        chunks = chunk_document(text, chunk_size=300)
        for chunk in chunks:
            meta = {"category": "general", "doc_type": "defect_reference",
                    "defects_mentioned": "general", "source_file": "defects.txt"}
            collection.add(
                ids=[f"ref_{doc_id}"],
                embeddings=[embedder.encode(chunk).tolist()],
                documents=[chunk],
                metadatas=[meta],
            )
            doc_id += 1; total += 1

    # Process context configs
    ctx_dir = os.path.join(CFG_DIR, "process_context")
    if os.path.exists(ctx_dir):
        for fname in sorted(os.listdir(ctx_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(ctx_dir, fname)) as f:
                ctx = json.load(f)
            cat  = fname.replace("_contexts.json", "")
            text = f"Process context for {cat}: " + json.dumps(ctx, indent=2)
            chunks = chunk_document(text, chunk_size=300)
            for chunk in chunks:
                meta = {"category": cat, "doc_type": "process_context",
                        "defects_mentioned": "general", "source_file": fname}
                collection.add(
                    ids=[f"ctx_{doc_id}"],
                    embeddings=[embedder.encode(chunk).tolist()],
                    documents=[chunk],
                    metadatas=[meta],
                )
                doc_id += 1; total += 1

    print(f"  Total chunks indexed: {total}")
    return collection, embedder, total


def retrieve_evidence(
    collection,
    embedder,
    category: str,
    defect_type: str,
    total_chunks: int,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve relevant KB passages for a (category, defect_type) pair.

    Returns list of dicts with keys: text, source, doc_type, relevance
    """
    query = f"{category} {defect_type} defect failure mechanism cause visual appearance"
    query_embedding = embedder.encode(query).tolist()

    where_filter = (
        {"$or": [{"category": category}, {"category": "general"}]}
        if total_chunks > 10
        else None
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,
        where=where_filter,
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    passages = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1 - dist
        if meta.get("category") == category:
            score += 0.2
        if defect_type in meta.get("defects_mentioned", ""):
            score += 0.15
        passages.append({
            "text":      doc[:500],
            "source":    meta.get("source_file", "unknown"),
            "doc_type":  meta.get("doc_type", "unknown"),
            "relevance": round(score, 3),
        })

    passages.sort(key=lambda x: x["relevance"], reverse=True)
    return passages[:top_k]
