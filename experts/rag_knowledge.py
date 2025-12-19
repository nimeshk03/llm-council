#!/usr/bin/env python3

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = str(Path("data/chroma").resolve())
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

@dataclass
class RAGChunk:
    text: str
    source: str
    page: int

class KnowledgeRAG:
    def __init__(self, collection_name: str = "knowledge"):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Light, fast embedding model (≈80MB, 384-dim)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        
        # Text splitter tuned for textbooks/notes
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n## ",
                "\n\n### ",
                "\n\nTheorem ",
                "\n\nDefinition ",
                "\n\nExample ",
                "\n\n",
                "\n",
                " ",
            ],
        )
        
        print(f"✓ KnowledgeRAG initialized with collection '{collection_name}'")
        print(f"✓ Chroma dir: {CHROMA_DIR}")
        print(f"✓ Documents in collection: {self.collection.count()}")
    
    def add_pdf(self, pdf_path: str, doc_name: Optional[str] = None, subject: str = "general") -> int:
        """Index a PDF (textbooks, lecture notes, etc.)."""
        path = Path(pdf_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        
        if doc_name is None:
            doc_name = path.stem
        
        reader = PdfReader(str(path))
        print(f"Indexing PDF: {path.name} ({len(reader.pages)} pages)")
        
        all_chunks = []
        metadatas = []
        ids = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            
            text = text.strip()
            if not text:
                continue
            
            chunks = self.splitter.split_text(text)
            for i, ch in enumerate(chunks):
                all_chunks.append(ch)
                metadatas.append(
                    {
                        "source": str(path),
                        "doc_name": doc_name,
                        "subject": subject,
                        "page": page_num,
                    }
                )
                ids.append(f"{doc_name}_p{page_num}_c{i}")
        
        if not all_chunks:
            print("No text extracted from PDF.")
            return 0
        
        self.collection.add(
            documents=all_chunks,
            metadatas=metadatas,
            ids=ids,
        )
        
        print(f"✓ Indexed {len(all_chunks)} chunks from {doc_name}")
        return len(all_chunks)
    
    def add_text(self, text: str, doc_name: str, subject: str = "general") -> int:
        """Index a plain text string (e.g., pasted notes)."""
        chunks = self.splitter.split_text(text)
        ids = [f"{doc_name}_c{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": doc_name, "doc_name": doc_name, "subject": subject, "page": 0}
            for _ in chunks
        ]
        
        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        print(f"✓ Indexed {len(chunks)} chunks from text doc '{doc_name}'")
        return len(chunks)
    
    def retrieve(self, query: str, k: int = 2, subject: Optional[str] = None,
                 source_filter: Optional[str] = None) -> List[RAGChunk]:
        """Retrieve top-k relevant chunks, with optional subject/source filter."""
        where = {}
        if subject:
            where["subject"] = subject
        if source_filter:
            where["source"] = {"$contains": source_filter}
        
        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where or None,
        )
        
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        
        chunks: List[RAGChunk] = []
        for d, m in zip(docs, metas):
            chunks.append(
                RAGChunk(
                    text=d,
                    source=m.get("source", ""),
                    page=m.get("page", 0),
                )
            )
        
        return chunks
    
    def build_context(self, query: str, k: int = 2,
                      subject: Optional[str] = None,
                      source_filter: Optional[str] = None,
                      max_chars_per_chunk: int = 500) -> str:
        """
        Return a formatted context string from top-k chunks.
        OPTIMIZED: Reduced k and truncated chunks to prevent timeouts.
        """
        chunks = self.retrieve(query, k=k, subject=subject, source_filter=source_filter)
        
        if not chunks:
            return "No relevant context found."
        
        lines = []
        for i, ch in enumerate(chunks, start=1):
            # Truncate long chunks to prevent timeout
            text = ch.text[:max_chars_per_chunk]
            if len(ch.text) > max_chars_per_chunk:
                text += "..."
            
            lines.append(
                f"[DOC {i}] Source: {Path(ch.source).name}, page {ch.page}\n{text}"
            )
        
        return "\n\n---\n\n".join(lines)

if __name__ == "__main__":
    rag = KnowledgeRAG()
    print("Place PDFs into ~/llm-council/textbooks or ~/llm-council/uploads and index them with add_pdf().")
