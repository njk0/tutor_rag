"""
Query Classifier Module
Classifies user queries by subject and builds metadata filters.
"""
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import SUBJECTS, SUBJECT_KEYWORDS


class QueryClassifier:
    """Classifies queries to determine subject and build metadata filters."""
    
    def __init__(self):
        self.subject_keywords = SUBJECT_KEYWORDS
    
    def classify_subject(self, query: str) -> Tuple[str, float]:
        """
        Classify the query into a subject category.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (subject, confidence_score)
        """
        query_lower = query.lower()
        
        # Count keyword matches for each subject
        scores = {}
        for subject, keywords in self.subject_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 1
            scores[subject] = score
        
        # Find the subject with highest score
        if max(scores.values()) == 0:
            # No keywords matched, use heuristics
            return self._classify_by_heuristics(query)
        
        best_subject = max(scores, key=scores.get)
        total_matches = sum(scores.values())
        confidence = scores[best_subject] / total_matches if total_matches > 0 else 0
        
        return best_subject, confidence
    
    def _classify_by_heuristics(self, query: str) -> Tuple[str, float]:
        """
        Classify query using heuristics when no keywords match.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (subject, confidence_score)
            Returns (None, 0) if no confident classification can be made
        """
        query_lower = query.lower()
        
        # Math detection - look for numbers, equations, mathematical terms
        math_patterns = [
            r'\d+\s*[\+\-\*\/\=]\s*\d+',  # Basic operations
            r'solve',
            r'calculate',
            r'equation',
            r'how many',
            r'find the value',
            r'x\s*[\+\-\*\/\=]',  # Variable equations
        ]
        if any(re.search(pattern, query_lower) for pattern in math_patterns):
            return "Maths", 0.7
        
        # Science detection - ONLY specific science patterns, not generic ones
        science_patterns = [
            r'photosynthesis',
            r'chemical',
            r'physical properties',
            r'reaction',
            r'experiment',
            r'organism',
            r'cell structure',
        ]
        if any(re.search(pattern, query_lower) for pattern in science_patterns):
            return "Science", 0.6
        
        # Return None to indicate no confident classification - will search all subjects
        return None, 0.0
    
    def is_math_problem(self, query: str) -> bool:
        """
        Determine if the query is a math problem requiring step-by-step solution.
        
        Args:
            query: User query
            
        Returns:
            True if it's a math problem
        """
        query_lower = query.lower()
        
        # Patterns indicating a math problem
        problem_patterns = [
            # English patterns
            r'solve',
            r'calculate',
            r'find the value',
            r'evaluate',
            r'simplify',
            r'factorize',
            r'prove that',
            r'\d+\s*[\+\-\*\/\=]',  # Equations with numbers
            r'x\s*[\+\-\*\/\=]',  # Equations with variables
            r'area of',
            r'perimeter of',
            r'sum of',
            r'product of',
            r'how many',
            r'how much',
            r'what is the',
            r'total',
            r'ratio',
            r'proportion',
            
            # Percentage patterns (works for any language)
            r'\d+\s*%',  # Numbers with percentage symbol
            r'\d+\s*percent',
            r'\d+\s*சதவீதம்',  # Tamil percentage
            
            # Unit patterns (quantity problems)
            r'\d+\s*grams?',
            r'\d+\s*kg',
            r'\d+\s*cm',
            r'\d+\s*meters?',
            r'\d+\s*கிராம்',  # Tamil grams
            r'\d+\s*கிலோ',  # Tamil kilo
            r'\d+\s*மீட்டர்',  # Tamil meter
            r'\d+\s*செமீ',  # Tamil cm
            
            # Tamil math problem indicators
            r'கணக்கிடு',  # Calculate
            r'தீர்க்க',  # Solve
            r'கண்டுபிடி',  # Find
            r'எத்தனை',  # How many/much
            r'எவ்வளவு',  # How much
            r'மொத்தம்',  # Total
            r'தேவைப்படும்',  # Required/needed
            r'அளவு',  # Quantity/amount
            r'கலவை',  # Mixture
            r'விகிதம்',  # Ratio
            r'சதவீதம்',  # Percentage
            r'பெற',  # To get/obtain
        ]
        
        return any(re.search(pattern, query_lower) for pattern in problem_patterns)
    
    def build_metadata_filter(
        self, 
        subject: str, 
        language: str,
        sub_subject: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build a metadata filter for retrieval.
        
        Args:
            subject: Main subject
            language: Query language
            sub_subject: Optional sub-subject filter
            content_type: Optional content type filter
            
        Returns:
            Metadata filter dictionary
        """
        filter_dict = {
            "subject": subject,
            "language": language
        }
        
        if sub_subject:
            filter_dict["sub_subject"] = sub_subject
        
        if content_type:
            filter_dict["content_type"] = content_type
        
        return filter_dict
    
    def extract_topic_hints(self, query: str) -> List[str]:
        """
        Extract potential topic hints from the query.
        
        Args:
            query: User query
            
        Returns:
            List of potential topic keywords
        """
        # Remove common words
        stop_words = {
            'what', 'is', 'are', 'the', 'of', 'in', 'a', 'an', 'how', 'why',
            'explain', 'describe', 'tell', 'me', 'about', 'give', 'list',
            'என்ன', 'எப்படி', 'ஏன்', 'பற்றி'  # Tamil stop words
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        topic_hints = [w for w in words if w not in stop_words and len(w) > 2]
        
        return topic_hints


if __name__ == "__main__":
    # Test the query classifier
    classifier = QueryClassifier()
    
    test_queries = [
        "What are the properties of alcohol?",
        "Solve the equation 2x + 5 = 15",
        "What is photosynthesis?",
        "Find the area of a circle with radius 5cm",
        "மது பற்றிய பண்புகள் என்ன?",
        "2 + 2 = ?",
        "Explain the Indian independence movement"
    ]
    
    for query in test_queries:
        subject, confidence = classifier.classify_subject(query)
        is_math = classifier.is_math_problem(query)
        print(f"\nQuery: {query}")
        print(f"  Subject: {subject} (confidence: {confidence:.2f})")
        print(f"  Is math problem: {is_math}")
