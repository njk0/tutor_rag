"""
Data Ingestion Script
Processes all PDF documents and creates subject-specific FAISS indices.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, VECTOR_STORE_DIR, SUBJECTS
from src.data_processor import DataProcessor
from src.vector_store import VectorStore
from src.metadata_extractor import MetadataExtractor


def main():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("📚 School Tutor RAG System - Data Ingestion")
    print("=" * 60)
    
    # Initialize components
    print("\n🔧 Initializing components...")
    processor = DataProcessor()
    vector_store = VectorStore()
    metadata_extractor = MetadataExtractor()
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("Please ensure PDF files are in the 'data' folder.")
        return
    
    # Count PDF files
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {DATA_DIR}")
        return
    
    print(f"\n📁 Found {len(pdf_files)} PDF files")
    
    # Process all PDFs
    print("\n🔄 Processing PDFs...")
    processed_data = processor.process_all_pdfs(DATA_DIR)
    
    # Create indices for each subject
    print("\n📊 Creating FAISS indices...")
    for subject in SUBJECTS:
        docs = processed_data.get(subject, [])
        if docs:
            print(f"\n  Creating index for {subject}...")
            
            # Enrich metadata with content-based detection
            enriched_docs = []
            for doc in docs:
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
                enriched_metadata = metadata_extractor.enrich_metadata(metadata, text)
                enriched_docs.append({
                    "text": text,
                    "metadata": enriched_metadata
                })
            
            vector_store.create_index(subject, enriched_docs)
        else:
            print(f"\n  ⚠️ No documents for {subject}")
    
    # Save all indices
    print("\n💾 Saving indices to disk...")
    vector_store.save(VECTOR_STORE_DIR)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Ingestion Complete!")
    print("=" * 60)
    
    stats = vector_store.get_stats()
    print("\n📈 Index Statistics:")
    total_docs = 0
    for subject, count in stats.items():
        print(f"  {subject}: {count} vectors")
        total_docs += count
    
    print(f"\n  Total: {total_docs} vectors")
    print(f"\n📂 Indices saved to: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
