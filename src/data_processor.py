"""
Data Processor Module
Handles PDF text extraction, chunking, and preprocessing for the RAG system.
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pypdf import PdfReader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR, 
    SUBJECTS, CONTENT_TYPES
)
from src.metadata_extractor import MetadataExtractor


class DataProcessor:
    """Processes PDF documents for ingestion into the RAG system."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata_extractor = MetadataExtractor()
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF file with page-level metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing text and page metadata
        """
        pages_data = []
        
        try:
            reader = PdfReader(str(pdf_path))
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                
                if text and text.strip():
                    # Clean the extracted text
                    cleaned_text = self._clean_text(text)
                    
                    pages_data.append({
                        "text": cleaned_text,
                        "page_number": page_num,
                        "source_file": pdf_path.name
                    })
                    
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            
        return pages_data
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep Tamil and mathematical symbols
        # Keep: Tamil (0B80-0BFF), basic Latin, numbers, math symbols
        text = re.sub(r'[^\u0B80-\u0BFF\u0000-\u007F\u2200-\u22FF\s\.\,\;\:\!\?\-\(\)\[\]\{\}\+\=\*\/\%\^\<\>\°]', '', text)
        
        # Normalize multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        if len(text) <= self.chunk_size:
            chunks.append({
                "text": text,
                "metadata": metadata.copy()
            })
            return chunks
        
        # Split by sentences first for better context preservation
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk += sentence + " "
                current_length += sentence_length + 1
            else:
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": chunk_metadata
                    })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence + " "
                current_length = len(current_chunk)
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle both English and Tamil sentence endings
        sentence_endings = r'[.!?।॥]+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the last portion of text for overlap."""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]
    
    def detect_chapter_topic(self, text: str, page_num: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect chapter and topic from text content.
        
        Args:
            text: Text content
            page_num: Page number
            
        Returns:
            Tuple of (chapter, topic)
        """
        chapter = None
        topic = None
        
        # Common chapter patterns
        chapter_patterns = [
            r'Chapter\s*(\d+)\s*[:\-]?\s*(.+?)(?:\n|$)',
            r'CHAPTER\s*(\d+)\s*[:\-]?\s*(.+?)(?:\n|$)',
            r'Unit\s*(\d+)\s*[:\-]?\s*(.+?)(?:\n|$)',
            r'அலகு\s*(\d+)\s*[:\-]?\s*(.+?)(?:\n|$)',  # Tamil unit pattern
        ]
        
        for pattern in chapter_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                chapter = f"Chapter {match.group(1)}: {match.group(2).strip()}"
                break
        
        # Topic detection (section headers)
        topic_patterns = [
            r'^(\d+\.\d+)\s+(.+?)(?:\n|$)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$',
        ]
        
        for pattern in topic_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                topic = match.group(0).strip()[:100]  # Limit topic length
                break
        
        return chapter, topic
    
    def detect_content_type(self, text: str) -> str:
        """
        Detect the type of content (theory, example, exercise, etc.)
        
        Args:
            text: Text content
            
        Returns:
            Content type string
        """
        text_lower = text.lower()
        
        # Check for exercise/problem patterns
        if any(pattern in text_lower for pattern in [
            'solve', 'find', 'calculate', 'exercise', 'problem',
            'கணக்கிடுக', 'தீர்க்க'  # Tamil exercise keywords
        ]):
            return "exercise"
        
        # Check for example patterns
        if any(pattern in text_lower for pattern in [
            'example', 'for instance', 'such as', 'e.g.',
            'எடுத்துக்காட்டு', 'உதாரணம்'  # Tamil example keywords
        ]):
            return "example"
        
        # Check for definition patterns
        if any(pattern in text_lower for pattern in [
            'definition', 'is defined as', 'refers to',
            'வரையறை', 'என்பது'  # Tamil definition keywords
        ]):
            return "definition"
        
        # Check for formula patterns
        if any(pattern in text_lower for pattern in [
            'formula', '=', 'equation',
            'சூத்திரம்'  # Tamil formula keyword
        ]):
            return "formula"
        
        return "theory"
    
    def process_all_pdfs(self, data_dir: Path = DATA_DIR) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all PDFs in the data directory and organize by subject.
        
        Args:
            data_dir: Directory containing PDF files
            
        Returns:
            Dictionary mapping subjects to list of processed chunks
        """
        processed_data = {subject: [] for subject in SUBJECTS}
        
        pdf_files = list(data_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            # Extract metadata from filename
            file_metadata = self.metadata_extractor.extract_from_filename(pdf_path.name)
            subject = file_metadata.get("subject", "Science")
            
            # Extract text from PDF
            pages = self.extract_text_from_pdf(pdf_path)
            
            for page_data in pages:
                text = page_data["text"]
                page_num = page_data["page_number"]
                
                # Detect chapter and topic
                chapter, topic = self.detect_chapter_topic(text, page_num)
                
                # Detect content type
                content_type = self.detect_content_type(text)
                
                # Build complete metadata
                metadata = {
                    **file_metadata,
                    "chapter": chapter or "Unknown",
                    "topic": topic or "General",
                    "content_type": content_type,
                    "page_number": page_num,
                    "source_file": pdf_path.name
                }
                
                # Chunk the text
                chunks = self.chunk_text(text, metadata)
                
                # Add to appropriate subject
                if subject in processed_data:
                    processed_data[subject].extend(chunks)
                else:
                    processed_data["Science"].extend(chunks)
        
        # Print summary
        for subject, chunks in processed_data.items():
            print(f"  {subject}: {len(chunks)} chunks")
        
        return processed_data


if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor()
    data = processor.process_all_pdfs()
    
    for subject, chunks in data.items():
        if chunks:
            print(f"\n{subject} - Sample chunk:")
            print(f"  Text: {chunks[0]['text'][:200]}...")
            print(f"  Metadata: {chunks[0]['metadata']}")
