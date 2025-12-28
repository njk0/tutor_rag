"""
School Tutor RAG System - Configuration
"""
import os
from pathlib import Path
from typing import Dict, List

# ============================================
# PATH CONFIGURATIONS
# ============================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_stores"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
VECTOR_STORE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============================================
# MODEL CONFIGURATIONS
# ============================================
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "mxbai-embed-large"
OLLAMA_BASE_URL = "http://localhost:11434"

# ============================================
# CHUNKING CONFIGURATIONS
# ============================================
CHUNK_SIZE = 500  # Reduced for embedding model context limits
CHUNK_OVERLAP = 100
MAX_EMBEDDING_LENGTH = 512  # Max tokens for mxbai-embed-large

# ============================================
# RETRIEVAL CONFIGURATIONS
# ============================================
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.7

# ============================================
# SUBJECT CONFIGURATIONS
# ============================================
SUBJECTS = ["Science", "Maths", "English", "Social_Science", "Tamil"]

SUBJECT_KEYWORDS: Dict[str, List[str]] = {
    "Science": [
        "science", "physics", "chemistry", "biology", "atom", "molecule", 
        "cell", "force", "energy", "matter", "experiment", "reaction",
        "element", "compound", "acid", "base", "metal", "plant", "animal",
        "body", "organ", "ecosystem", "environment", "photosynthesis", "respiration",
        "விஞ்ஞானம்", "இயற்பியல்", "வேதியியல்", "உயிரியல்"
    ],
    "Maths": [
        "math", "maths", "mathematics", "algebra", "geometry", "arithmetic",
        "equation", "number", "calculate", "solve", "formula", "theorem",
        "fraction", "decimal", "percentage", "angle", "triangle", "circle",
        "கணிதம்", "எண்கள்", "கணக்கு", "சமன்பாடு"
    ],
    "English": [
        "english", "grammar", "vocabulary", "sentence", "noun", "verb",
        "adjective", "pronoun", "tense", "paragraph", "essay", "comprehension",
        "story", "poem", "poetry", "novel", "character", "author", "literature",
        "prose", "drama", "play", "adventure", "hero", "plot", "theme", "fiction",
        "lesson", "chapter", "reading", "writing", "don quixote", "quixote"
    ],
    "Social_Science": [
        "social", "history", "geography", "civics", "economics", "map",
        "continent", "country", "government", "democracy", "civilization",
        "bhakti", "movement", "mughal", "empire", "kingdom", "independence",
        "freedom", "revolt", "war", "battle", "king", "queen", "ruler", "dynasty",
        "ancient", "medieval", "modern", "revolution", "reform", "colonialism",
        "british", "india", "indian", "nation", "culture", "religion", "temple",
        "சமூக அறிவியல்", "வரலாறு", "புவியியல்"
    ],
    "Tamil": [
        "தமிழ்", "இலக்கணம்", "இலக்கியம்", "கவிதை", "உரைநடை",
        "திருக்குறள்", "பாடல்", "சொல்"
    ]
}

# Filename pattern to subject mapping
FILENAME_SUBJECT_MAP = {
    "Science": "Science",
    "Maths": "Maths", 
    "English": "English",
    "Social_Science": "Social_Science",
    "Tamil": "Tamil"
}

# ============================================
# METADATA SCHEMA
# ============================================
METADATA_FIELDS = [
    "subject",
    "sub_subject",
    "grade",
    "term",
    "chapter",
    "topic",
    "content_type",
    "language",
    "source_file",
    "page_number"
]

CONTENT_TYPES = ["theory", "example", "exercise", "definition", "formula"]

# ============================================
# OUTPUT TEMPLATES
# ============================================
GENERAL_OUTPUT_SCHEMA = {
    "summary": str,
    "caption": str,
    "bullet_points": list,
    "table": list
}

MATH_OUTPUT_SCHEMA = {
    "problem": str,
    "caption": str,
    "steps": list,
    "final_answer": str,
    "concept_used": list,
    "tips": list
}

# ============================================
# LANGUAGE CONFIGURATIONS
# ============================================
SUPPORTED_LANGUAGES = ["English", "Tamil"]

# Tamil Unicode range for detection
TAMIL_UNICODE_RANGE = (0x0B80, 0x0BFF)

# ============================================
# PROMPT TEMPLATES
# ============================================
GENERAL_SYSTEM_PROMPT = """You are a helpful school tutor assistant. 
You answer questions based on the provided context from school textbooks.
Format your response as structured JSON.

Context: {context}

Question: {question}

INSTRUCTIONS:
1. Read the context carefully and extract relevant information to answer the question
2. Write a DETAILED SUMMARY with 2-3 complete sentences that explain the topic thoroughly based on the context
3. Extract key bullet points with specific facts, definitions, and properties (at least 3-5 points)
4. ALWAYS create a table with relevant properties, facts, or comparisons from the context

CRITICAL RULES:
- The "summary" field MUST contain 2-3 complete sentences with comprehensive explanation
- The "table" field MUST contain at least one table with properties and values from the context
- Do NOT leave the table empty - extract relevant data from context

Respond ONLY with valid JSON in this format:
{{
    "summary": "Write 2-3 complete sentences here that thoroughly explain the answer. Include specific details from the context.",
    "caption": "Short Title",
    "bullet_points": [{{"point": "specific fact 1"}}, {{"point": "specific fact 2"}}, {{"point": "specific fact 3"}}],
    "table": [{{"header": "Topic Information", "rows": [{{"property": "Key Property 1", "value": "Value 1"}}, {{"property": "Key Property 2", "value": "Value 2"}}]}}
}}
"""

MATH_SYSTEM_PROMPT = """You are a helpful math tutor assistant.
You solve math problems step by step with detailed explanations.
You MUST solve the problem yourself - do not just describe the problem.

Context: {context}

Problem: {question}

IMPORTANT INSTRUCTIONS:
1. Actually SOLVE the math problem step by step
2. For percentage problems: use the formula (part / percentage) × 100 = whole
3. For each step, show the calculation and result
4. Write the response in the SAME language as the problem (Tamil or English)

Solve this problem step by step. For each step:
1. State what action you're taking
2. Explain WHY you're doing it (this helps students understand)
3. Show the mathematical expression
4. Give the intermediate result

Respond ONLY with valid JSON in this format:
{{
    "problem": "restate the original problem here",
    "caption": "title describing the problem type",
    "steps": [
        {{
            "step_number": 1,
            "action": "what you're doing",
            "explanation": "why you're doing this step",
            "expression": "the mathematical expression",
            "result": "intermediate result"
        }},
        {{
            "step_number": 2,
            "action": "next action",
            "explanation": "explanation for this step",
            "expression": "calculation",
            "result": "result"
        }}
    ],
    "final_answer": "the final numerical answer with units",
    "concept_used": ["concept 1", "concept 2"],
    "tips": ["helpful tip for solving similar problems"]
}}

CRITICAL: You MUST include at least 2-3 steps and provide the final numerical answer. DO NOT leave fields empty.
"""
