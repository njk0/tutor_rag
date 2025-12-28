"""
Vector Store Module
Manages FAISS indices for subject-specific document storage and retrieval.
"""
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

try:
    import faiss
except ImportError:
    print("FAISS not installed. Please run: pip install faiss-cpu")
    raise

import ollama

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    EMBEDDING_MODEL, VECTOR_STORE_DIR, SUBJECTS, 
    TOP_K_RESULTS, OLLAMA_BASE_URL
)

# Maximum characters for embedding (mxbai-embed-large has ~512 token limit, ~1.5 chars/token for mixed content)
MAX_EMBEDDING_CHARS = 500


class VectorStore:
    """Manages FAISS indices for the RAG system."""
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedding_model = embedding_model
        self.indices: Dict[str, faiss.Index] = {}
        self.documents: Dict[str, List[Dict[str, Any]]] = {subject: [] for subject in SUBJECTS}
        self.embedding_dim: Optional[int] = None
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        
        # Ensure vector store directory exists
        VECTOR_STORE_DIR.mkdir(exist_ok=True)
    
    def _truncate_text(self, text: str, max_chars: int = MAX_EMBEDDING_CHARS) -> str:
        """Truncate text to fit within embedding model limits."""
        if len(text) <= max_chars:
            return text
        # Truncate at word boundary
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:  # Only use word boundary if reasonable
            truncated = truncated[:last_space]
        return truncated
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array
        """
        # Truncate text to fit within model limits
        text = self._truncate_text(text)
        
        try:
            response = self.ollama_client.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            embedding = np.array(response['embedding'], dtype=np.float32)
            
            # Set embedding dimension on first call
            if self.embedding_dim is None:
                self.embedding_dim = len(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding for text (length={len(text)}): {e}")
            # Return zero vector on error to continue processing
            if self.embedding_dim:
                return np.zeros(self.embedding_dim, dtype=np.float32)
            raise
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.get_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings, dtype=np.float32)
    
    def create_index(self, subject: str, documents: List[Dict[str, Any]]) -> None:
        """
        Create a FAISS index for a specific subject.
        
        Args:
            subject: Subject name
            documents: List of document chunks with text and metadata
        """
        if not documents:
            print(f"No documents to index for {subject}")
            return
        
        print(f"\nCreating index for {subject} with {len(documents)} documents...")
        
        # Extract texts for embedding
        texts = [doc["text"] for doc in documents]
        
        # Generate embeddings
        embeddings = self.get_embeddings_batch(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Store index and documents
        self.indices[subject] = index
        self.documents[subject] = documents
        
        print(f"  Created index with {index.ntotal} vectors")
    
    def add_documents(self, subject: str, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to an existing index.
        
        Args:
            subject: Subject name
            documents: List of document chunks to add
        """
        if subject not in self.indices:
            self.create_index(subject, documents)
            return
        
        print(f"Adding {len(documents)} documents to {subject} index...")
        
        # Extract texts and generate embeddings
        texts = [doc["text"] for doc in documents]
        embeddings = self.get_embeddings_batch(texts)
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.indices[subject].add(embeddings)
        self.documents[subject].extend(documents)
        
        print(f"  Index now has {self.indices[subject].ntotal} vectors")
    
    def search(
        self, 
        query: str, 
        subject: str, 
        top_k: int = TOP_K_RESULTS,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents in a subject index.
        
        Args:
            query: Search query
            subject: Subject to search in
            top_k: Number of results to return
            metadata_filter: Optional metadata filters
            
        Returns:
            List of relevant documents with scores
        """
        if subject not in self.indices or self.indices[subject].ntotal == 0:
            print(f"No index found for {subject}")
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index - get more results if filtering
        search_k = top_k * 3 if metadata_filter else top_k
        scores, indices = self.indices[subject].search(query_embedding, min(search_k, self.indices[subject].ntotal))
        
        # Collect results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for invalid indices
                continue
                
            doc = self.documents[subject][idx].copy()
            doc["score"] = float(score)
            
            # Apply metadata filter if provided
            if metadata_filter:
                match = True
                for key, value in metadata_filter.items():
                    if key in doc.get("metadata", {}):
                        if doc["metadata"][key] != value:
                            match = False
                            break
                if not match:
                    continue
            
            results.append(doc)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_all_subjects(
        self, 
        query: str, 
        top_k: int = TOP_K_RESULTS
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all subject indices.
        
        Args:
            query: Search query
            top_k: Number of results per subject
            
        Returns:
            Dictionary mapping subjects to results
        """
        results = {}
        for subject in SUBJECTS:
            if subject in self.indices and self.indices[subject].ntotal > 0:
                results[subject] = self.search(query, subject, top_k)
        return results
    
    def save(self, directory: Path = VECTOR_STORE_DIR) -> None:
        """
        Save all indices and documents to disk.
        
        Args:
            directory: Directory to save to
        """
        directory.mkdir(exist_ok=True)
        
        for subject in SUBJECTS:
            if subject in self.indices and self.indices[subject].ntotal > 0:
                # Save FAISS index
                index_path = directory / f"{subject.lower()}_index.faiss"
                faiss.write_index(self.indices[subject], str(index_path))
                
                # Save documents
                docs_path = directory / f"{subject.lower()}_docs.pkl"
                with open(docs_path, "wb") as f:
                    pickle.dump(self.documents[subject], f)
                
                print(f"Saved {subject} index: {self.indices[subject].ntotal} vectors")
    
    def load(self, directory: Path = VECTOR_STORE_DIR) -> None:
        """
        Load all indices and documents from disk.
        
        Args:
            directory: Directory to load from
        """
        for subject in SUBJECTS:
            index_path = directory / f"{subject.lower()}_index.faiss"
            docs_path = directory / f"{subject.lower()}_docs.pkl"
            
            if index_path.exists() and docs_path.exists():
                # Load FAISS index
                self.indices[subject] = faiss.read_index(str(index_path))
                
                # Load documents
                with open(docs_path, "rb") as f:
                    self.documents[subject] = pickle.load(f)
                
                print(f"Loaded {subject} index: {self.indices[subject].ntotal} vectors")
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about stored indices."""
        stats = {}
        for subject in SUBJECTS:
            if subject in self.indices:
                stats[subject] = self.indices[subject].ntotal
            else:
                stats[subject] = 0
        return stats


if __name__ == "__main__":
    # Test the vector store
    store = VectorStore()
    
    # Test embedding
    test_text = "This is a test sentence about science."
    embedding = store.get_embedding(test_text)
    print(f"Embedding dimension: {len(embedding)}")
    
    # Test index creation
    test_docs = [
        {"text": "Photosynthesis is the process by which plants make food.", "metadata": {"subject": "Science", "topic": "Biology"}},
        {"text": "The equation of a straight line is y = mx + c.", "metadata": {"subject": "Maths", "topic": "Algebra"}},
    ]
    
    store.create_index("Science", [test_docs[0]])
    results = store.search("How do plants make food?", "Science")
    print(f"\nSearch results: {results}")
