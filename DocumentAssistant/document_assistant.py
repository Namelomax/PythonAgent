from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import TextChunker
from .loaders import load_document
from .llm_openrouter import OpenRouterLLM


class DocumentAssistant:
    """
    DocumentAssistant implements a simple RAG pipeline:
    - document loading
    - text chunking
    - embedding indexing
    - semantic search
    - answer generation via LLM
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        top_k: int = 3,
    ):
        self.chunker = TextChunker(chunk_size, overlap)
        self.top_k = top_k

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.llm = OpenRouterLLM()

        # In-memory storage
        self.chunks: List[str] = []
        self.embeddings: np.ndarray | None = None

        # NOTE:
        # Here we keep everything in memory.
        # This can later be replaced with a vector database
        # (e.g. SurrealDB, FAISS, Pinecone).

    def index_documents(self, documents: List[str]) -> None:
        """
        Loads documents, splits them into chunks,
        and computes embeddings for each chunk.
        """
        all_chunks: List[str] = []

        for path in documents:
            text = load_document(path)
            chunks = self.chunker.split(text)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No text chunks were created")

        self.chunks = all_chunks
        self.embeddings = self.embedding_model.encode(
            all_chunks, convert_to_numpy=True
        )

    def answer_query(self, query: str) -> str:
        """
        Finds relevant document chunks and generates
        an answer using an LLM.
        """
        if self.embeddings is None:
            raise RuntimeError("Documents are not indexed")

        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True
        )

        similarities = cosine_similarity(
            query_embedding, self.embeddings
        )[0]

        top_indices = similarities.argsort()[-self.top_k:][::-1]
        retrieved_chunks = [self.chunks[i] for i in top_indices]

        prompt = self._build_prompt(query, retrieved_chunks)

        return self.llm.generate(prompt)

    @staticmethod
    def _build_prompt(query: str, chunks: List[str]) -> str:
        context = "\n\n".join(chunks)

        return (
            "Используй только следующие фрагменты документов для ответа:\n"
            f"{context}\n\n"
            f"Вопрос: {query}\n"
            "Ответ:"
        )
