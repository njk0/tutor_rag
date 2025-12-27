"""
RAG Chain Module
Main RAG pipeline combining retrieval and generation.
"""
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

import ollama

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    LLM_MODEL, OLLAMA_BASE_URL, TOP_K_RESULTS,
    GENERAL_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT
)
from src.vector_store import VectorStore
from src.language_detector import LanguageDetector
from src.query_classifier import QueryClassifier
from src.output_formatter import OutputFormatter


class RAGChain:
    """Main RAG pipeline for the school tutor system."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the RAG chain.
        
        Args:
            vector_store: Optional pre-initialized vector store
        """
        self.vector_store = vector_store or VectorStore()
        self.language_detector = LanguageDetector()
        self.query_classifier = QueryClassifier()
        self.output_formatter = OutputFormatter()
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        self.llm_model = LLM_MODEL
    
    def query(
        self, 
        question: str,
        subject_override: Optional[str] = None,
        top_k: int = TOP_K_RESULTS
    ) -> Dict[str, Any]:
        """
        Process a query and return structured response.
        
        Args:
            question: User's question
            subject_override: Optional subject to force
            top_k: Number of documents to retrieve
            
        Returns:
            Structured JSON response
        """
        # Step 1: Detect language
        language = self.language_detector.detect(question)
        print(f"📝 Detected language: {language}")
        
        # Step 2: Classify subject
        if subject_override:
            subject = subject_override
            confidence = 1.0
        else:
            subject, confidence = self.query_classifier.classify_subject(question)
        
        # Step 3: Check if math problem
        is_math = self.query_classifier.is_math_problem(question)
        if is_math:
            subject = "Maths"
            confidence = 0.8
        print(f"📚 Classified subject: {subject} (confidence: {confidence:.2f})")
        print(f"🔢 Is math problem: {is_math}")
        
        # Step 4: Retrieve relevant documents
        results = []
        
        # If no confident subject classification, search ALL subjects and pick best results
        if subject is None or confidence < 0.3:
            print(f"🔍 Low confidence classification, searching ALL indices...")
            all_results = self.vector_store.search_all_subjects(question, top_k)
            
            # Find results with best scores across all subjects
            best_results = []
            for subj, res in all_results.items():
                if res:
                    for doc in res:
                        best_results.append((subj, doc))
            
            # Sort by score (higher is better) and pick top results
            if best_results:
                best_results.sort(key=lambda x: x[1].get('score', 0), reverse=True)
                subject = best_results[0][0]  # Use subject of best match
                results = [r[1] for r in best_results[:top_k]]
                print(f"📄 Best match found in {subject} index")
        else:
            # Search specific subject index
            print(f"🔍 Searching in {subject} index...")
            results = self.vector_store.search(
                query=question,
                subject=subject,
                top_k=top_k,
                metadata_filter=None
            )
            
            # If no results in primary subject, try other subjects
            if not results:
                print(f"⚠️ No results in {subject}, searching other indices...")
                all_results = self.vector_store.search_all_subjects(question, top_k)
                for subj, res in all_results.items():
                    if res:
                        results = res[:top_k]
                        subject = subj
                        break
        
        if not results:
            return self._create_no_results_response(question, language)
        
        print(f"📄 Retrieved {len(results)} documents")
        
        # Step 6: Build context from retrieved documents
        context = self._build_context(results)
        
        # Step 7: Generate response using LLM
        print(f"🤖 Generating response with {self.llm_model}...")
        response_text = self._generate_response(
            question=question,
            context=context,
            is_math=is_math,
            language=language
        )
        
        # Step 8: Format response
        if is_math:
            formatted = self.output_formatter.format_math_response(response_text, question)
        else:
            formatted = self.output_formatter.format_general_response(response_text, question, subject)
        
        # Add metadata to response
        formatted["_metadata"] = {
            "subject": subject,
            "language": language,
            "is_math_problem": is_math,
            "documents_retrieved": len(results),
            "confidence": confidence
        }
        
        return formatted
    
    def _build_context(self, documents: List[Dict[str, Any]], max_length: int = 4000) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            max_length: Maximum context length
            
        Returns:
            Concatenated context string
        """
        context_parts = []
        current_length = 0
        
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            # Add source info
            source_info = f"[Source: {metadata.get('source_file', 'Unknown')} - {metadata.get('topic', 'General')}]\n"
            doc_text = source_info + text + "\n\n"
            
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "".join(context_parts)
    
    def _generate_response(
        self,
        question: str,
        context: str,
        is_math: bool,
        language: str
    ) -> str:
        """
        Generate response using Ollama LLM.
        
        Args:
            question: User question
            context: Retrieved context
            is_math: Whether this is a math problem
            language: Query language
            
        Returns:
            Generated response text
        """
        # Build language instruction as a system-level requirement
        if language == "Tamil":
            lang_instruction = "CRITICAL LANGUAGE REQUIREMENT: You MUST respond ONLY in Tamil language. All text in your JSON response must be in Tamil."
        else:
            lang_instruction = "CRITICAL LANGUAGE REQUIREMENT: You MUST respond ONLY in English language. Do NOT use Tamil or any other language. All text in your JSON response must be in English only."
        
        # Select appropriate prompt template
        if is_math:
            base_prompt = MATH_SYSTEM_PROMPT.format(context=context, question=question)
        else:
            base_prompt = GENERAL_SYSTEM_PROMPT.format(context=context, question=question)
        
        # Combine with language instruction at the START (more prominent)
        prompt = f"{lang_instruction}\n\n{base_prompt}\n\nREMINDER: {lang_instruction}"
        
        try:
            response = self.ollama_client.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1000  # Reduced for faster responses
                }
            )
            return response["response"]
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return json.dumps({
                "summary": f"Error generating response: {str(e)}",
                "caption": "Error",
                "bullet_points": [],
                "table": []
            })
    
    def _create_no_results_response(self, question: str, language: str) -> Dict[str, Any]:
        """Create response when no documents are found."""
        if language == "Tamil":
            message = "மன்னிக்கவும், இந்த கேள்விக்கான தகவல்கள் கிடைக்கவில்லை."
        else:
            message = "Sorry, I couldn't find relevant information for your question."
        
        return {
            "summary": message,
            "caption": "No Results Found",
            "bullet_points": [],
            "table": [],
            "_metadata": {
                "subject": "Unknown",
                "language": language,
                "is_math_problem": False,
                "documents_retrieved": 0,
                "confidence": 0.0
            }
        }
    
    def load_vector_store(self, directory: Optional[Path] = None) -> None:
        """Load vector store from disk."""
        if directory is not None:
            self.vector_store.load(directory)
        else:
            self.vector_store.load()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        return {
            "vector_store_stats": self.vector_store.get_stats(),
            "llm_model": self.llm_model,
            "embedding_model": self.vector_store.embedding_model
        }


if __name__ == "__main__":
    # Test the RAG chain
    rag = RAGChain()
    
    # Try to load existing vector store
    try:
        rag.load_vector_store()
        print("Vector store loaded successfully")
        print(f"Stats: {rag.get_stats()}")
    except Exception as e:
        print(f"No existing vector store found: {e}")
        print("Please run ingest.py first to create the vector store")
