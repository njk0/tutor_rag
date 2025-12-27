"""
School Tutor RAG System - Main Application
A multilingual tutoring system with subject-specific knowledge retrieval.
"""
import sys
import json
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import VECTOR_STORE_DIR
from src.rag_chain import RAGChain
from src.output_formatter import OutputFormatter


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║           🎓 School Tutor RAG System 🎓                      ║
║                                                              ║
║   Ask questions in English or Tamil (தமிழ்)                  ║
║   Subjects: Science, Maths, English, Social Science, Tamil   ║
║                                                              ║
║   Type 'quit' or 'exit' to end the session                   ║
║   Type 'stats' to see system statistics                      ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def format_response_display(response: dict) -> str:
    """Format response for console display."""
    output = []
    
    # Check if it's a math response
    is_math = "steps" in response
    
    if is_math:
        output.append("\n" + "=" * 60)
        output.append(f"📐 {response.get('caption', 'Math Solution')}")
        output.append("=" * 60)
        
        output.append(f"\n📝 Problem: {response.get('problem', '')}")
        
        output.append("\n📋 Solution Steps:")
        for step in response.get("steps", []):
            step_num = step.get("step_number", "")
            action = step.get("action", "")
            explanation = step.get("explanation", "")
            expression = step.get("expression", "")
            result = step.get("result", "")
            
            output.append(f"\n  Step {step_num}: {action}")
            if explanation:
                output.append(f"    💡 Why: {explanation}")
            if expression:
                output.append(f"    📝 {expression}")
            if result:
                output.append(f"    ➡️  Result: {result}")
        
        if response.get("final_answer"):
            output.append(f"\n✅ Final Answer: {response['final_answer']}")
        
        if response.get("concept_used"):
            output.append(f"\n📚 Concepts Used: {', '.join(response['concept_used'])}")
        
        if response.get("tips"):
            output.append("\n💡 Tips:")
            for tip in response["tips"]:
                output.append(f"   • {tip}")
    
    else:
        output.append("\n" + "=" * 60)
        output.append(f"📖 {response.get('caption', 'Response')}")
        output.append("=" * 60)
        
        if response.get("summary"):
            output.append(f"\n📝 Summary:\n{response['summary']}")
        
        if response.get("bullet_points"):
            output.append("\n📋 Key Points:")
            for point in response["bullet_points"]:
                point_text = point.get("point", str(point))
                output.append(f"   • {point_text}")
        
        if response.get("table"):
            for table in response["table"]:
                if isinstance(table, dict):
                    output.append(f"\n📊 {table.get('header', 'Table')}:")
                    for row in table.get("rows", []):
                        if isinstance(row, dict):
                            prop = row.get("property", "")
                            val = row.get("value", "")
                            output.append(f"   • {prop}: {val}")
    
    # Add metadata if available
    metadata = response.get("_metadata", {})
    if metadata:
        output.append("\n" + "-" * 40)
        output.append(f"📌 Subject: {metadata.get('subject', 'Unknown')}")
        output.append(f"🌐 Language: {metadata.get('language', 'Unknown')}")
        output.append(f"📄 Documents used: {metadata.get('documents_retrieved', 0)}")
    
    return "\n".join(output)


def main():
    """Main application loop."""
    print_banner()
    
    # Initialize RAG chain
    print("🔧 Initializing system...")
    
    try:
        rag = RAGChain()
        rag.load_vector_store()
        stats = rag.get_stats()
        
        total_vectors = sum(stats["vector_store_stats"].values())
        if total_vectors == 0:
            print("\n⚠️ Warning: Vector store is empty!")
            print("Please run the ingestion script first:")
            print("  python src/ingest.py")
            return
        
        print(f"✅ System ready! ({total_vectors} vectors loaded)")
        
    except Exception as e:
        print(f"\n❌ Error initializing system: {e}")
        print("\nPlease ensure:")
        print("  1. Ollama is running (ollama serve)")
        print("  2. Models are installed (ollama pull llama3.2 && ollama pull mxbai-embed-large)")
        print("  3. Data has been ingested (python src/ingest.py)")
        return
    
    formatter = OutputFormatter()
    
    # Main loop
    print("\n" + "-" * 60)
    
    while True:
        try:
            # Get user input
            print()
            user_input = input("🎯 Your question: ").strip()
            
            if not user_input:
                continue
            
            # Check for commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thank you for using School Tutor. Goodbye!")
                break
            
            if user_input.lower() == "stats":
                print("\n📊 System Statistics:")
                stats = rag.get_stats()
                print(f"  LLM Model: {stats['llm_model']}")
                print(f"  Embedding Model: {stats['embedding_model']}")
                print("  Vector Store:")
                for subject, count in stats["vector_store_stats"].items():
                    print(f"    {subject}: {count} vectors")
                continue
            
            if user_input.lower() == "json":
                print("\n📋 JSON output mode enabled for next query")
                json_mode = True
                continue
            
            # Process query
            print("\n⏳ Processing your question...")
            response = rag.query(user_input)
            
            # Display response
            print(format_response_display(response))
            
            # Also print raw JSON for debugging
            print("\n📦 Raw JSON Response:")
            print(formatter.to_json_string(response))
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
