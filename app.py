"""
School Tutor RAG System - Streamlit UI
A multilingual tutoring system with subject-specific knowledge retrieval.
"""
import streamlit as st
import sys
import json
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_chain import RAGChain
from src.output_formatter import OutputFormatter


# Page configuration
st.set_page_config(
    page_title="🎓 School Tutor RAG System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .response-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .summary-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .bullet-point {
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }
    .metadata-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .json-container {
        background: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        overflow-x: auto;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached)."""
    try:
        rag = RAGChain()
        rag.load_vector_store()
        return rag, None
    except Exception as e:
        return None, str(e)


def display_response(response: dict):
    """Display the formatted response."""
    is_math = "steps" in response
    metadata = response.get("_metadata", {})
    
    # Caption/Title
    caption = response.get("caption", "Response")
    st.markdown(f"### 📖 {caption}")
    
    # Summary Section
    st.markdown("#### 📝 Summary")
    summary = response.get("summary", "")
    if summary:
        st.info(summary)
    
    if is_math:
        # Math-specific display
        st.markdown("#### 📋 Solution Steps")
        for step in response.get("steps", []):
            step_num = step.get("step_number", "")
            action = step.get("action", "")
            explanation = step.get("explanation", "")
            expression = step.get("expression", "")
            result = step.get("result", "")
            
            with st.expander(f"**Step {step_num}:** {action}", expanded=True):
                if explanation:
                    st.markdown(f"💡 **Why:** {explanation}")
                if expression:
                    st.markdown(f"📝 **Expression:** `{expression}`")
                if result:
                    st.markdown(f"➡️ **Result:** `{result}`")
        
        # Final Answer
        if response.get("final_answer"):
            st.success(f"✅ **Final Answer:** {response['final_answer']}")
        
        # Concepts Used
        if response.get("concept_used"):
            st.markdown("#### 📚 Concepts Used")
            concepts = ", ".join(response["concept_used"])
            st.info(concepts)
        
        # Tips
        if response.get("tips"):
            st.markdown("#### 💡 Tips")
            for tip in response["tips"]:
                st.markdown(f"• {tip}")
    
    else:
        # General response display
        
        # Bullet Points
        if response.get("bullet_points"):
            st.markdown("#### 📋 Key Points")
            for point in response["bullet_points"]:
                point_text = point.get("point", str(point)) if isinstance(point, dict) else str(point)
                st.markdown(f"• {point_text}")
        
        # Table
        if response.get("table"):
            st.markdown("#### 📊 Data Table")
            for table in response["table"]:
                if isinstance(table, dict):
                    st.markdown(f"**{table.get('header', 'Information')}**")
                    rows = table.get("rows", [])
                    if rows:
                        table_data = []
                        for row in rows:
                            if isinstance(row, dict):
                                table_data.append({
                                    "Property": row.get("property", ""),
                                    "Value": row.get("value", "")
                                })
                        if table_data:
                            st.table(table_data)
    
    # Metadata Section
    st.markdown("---")
    st.markdown("#### 📌 Response Metadata")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Subject", metadata.get("subject", "Unknown"))
    with col2:
        st.metric("Language", metadata.get("language", "Unknown"))
    with col3:
        st.metric("Documents Used", metadata.get("documents_retrieved", 0))
    with col4:
        confidence = metadata.get("confidence", 0)
        st.metric("Confidence", f"{confidence:.0%}")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 School Tutor RAG System</h1>
        <p>Ask questions in English or Tamil (தமிழ்)</p>
        <p>Subjects: Science, Maths, English, Social Science, Tamil</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    with st.spinner("🔧 Initializing system..."):
        rag, error = initialize_rag_system()
    
    if error:
        st.error(f"❌ Error initializing system: {error}")
        st.markdown("""
        **Please ensure:**
        1. Ollama is running (`ollama serve`)
        2. Models are installed (`ollama pull llama3.2 && ollama pull mxbai-embed-large`)
        3. Data has been ingested (`python src/ingest.py`)
        """)
        return
    
    # Get stats
    stats = rag.get_stats()
    total_vectors = sum(stats["vector_store_stats"].values())
    
    if total_vectors == 0:
        st.warning("⚠️ Vector store is empty! Please run: `python src/ingest.py`")
        return
    
    # Sidebar with stats
    with st.sidebar:
        st.markdown("### 📊 System Statistics")
        st.markdown(f"**LLM Model:** `{stats['llm_model']}`")
        st.markdown(f"**Embedding Model:** `{stats['embedding_model']}`")
        st.markdown(f"**Total Vectors:** `{total_vectors:,}`")
        
        st.markdown("---")
        st.markdown("### 📚 Vector Store")
        for subject, count in stats["vector_store_stats"].items():
            if count > 0:
                st.markdown(f"• **{subject}:** {count:,} vectors")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This is a multilingual tutoring system that uses RAG (Retrieval Augmented Generation) 
        to answer questions from school textbooks in English and Tamil.
        """)
    
    # Main input area
    st.markdown("### 🎯 Ask Your Question")
    
    # Subject selection (optional)
    col1, col2 = st.columns([3, 1])
    with col1:
        user_question = st.text_area(
            "Enter your question:",
            placeholder="Type your question here... (e.g., 'What is photosynthesis?' or 'ஒளிச்சேர்க்கை என்றால் என்ன?')",
            height=100,
            key="question_input"
        )
    with col2:
        subject_override = st.selectbox(
            "Subject (optional):",
            ["Auto-detect", "Science", "Maths", "English", "Social_Science", "Tamil"],
            key="subject_select"
        )
        subject = None if subject_override == "Auto-detect" else subject_override
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        submit_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        if "last_response" in st.session_state:
            del st.session_state["last_response"]
        st.rerun()
    
    # Process query
    if submit_button and user_question.strip():
        with st.spinner("⏳ Processing your question..."):
            try:
                response = rag.query(user_question, subject_override=subject)
                st.session_state["last_response"] = response
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                return
    
    # Display response if available
    if "last_response" in st.session_state:
        response = st.session_state["last_response"]
        
        st.markdown("---")
        st.markdown("## 📄 Response")
        
        # Display formatted response
        display_response(response)
        
        # JSON Response Toggle
        st.markdown("---")
        show_json = st.checkbox("📦 Show JSON Response", key="show_json_checkbox")
        
        if show_json:
            st.markdown("### 📦 Raw JSON Response")
            formatter = OutputFormatter()
            json_str = formatter.to_json_string(response)
            st.code(json_str, language="json")
            
            # Download button for JSON
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name="response.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
