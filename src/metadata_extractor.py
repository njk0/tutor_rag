"""
Metadata Extractor Module
Extracts and generates metadata from PDF filenames and content.
"""
import re
from typing import Dict, Any, Optional
from pathlib import Path


class MetadataExtractor:
    """Extracts metadata from filenames and content."""
    
    # Filename patterns for subject extraction
    SUBJECT_PATTERNS = {
        "Science": [r"Science", r"science"],
        "Maths": [r"Maths", r"Math", r"maths", r"math"],
        "English": [r"English", r"english"],
        "Social_Science": [r"Social[_\s]?Science", r"social[_\s]?science"],
        "Tamil": [r"Tamil", r"tamil"]
    }
    
    # Language detection from filename
    LANGUAGE_PATTERNS = {
        "English": [r"_EM\.pdf$", r"_EM_", r"English"],
        "Tamil": [r"_TM\.pdf$", r"_TM_", r"Tamil"]
    }
    
    def extract_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF filename.
        
        Example filename: 7th_Science_Term_II_EM.pdf
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            "subject": "Unknown",
            "sub_subject": "General",
            "grade": "7",
            "term": "Unknown",
            "language": "English"
        }
        
        # Extract grade
        grade_match = re.search(r'(\d+)(?:th|st|nd|rd)?[_\s]', filename, re.IGNORECASE)
        if grade_match:
            metadata["grade"] = grade_match.group(1)
        
        # Extract subject
        for subject, patterns in self.SUBJECT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    metadata["subject"] = subject
                    break
            if metadata["subject"] != "Unknown":
                break
        
        # Extract term
        term_match = re.search(r'Term[_\s]*(I{1,3}|\d+)', filename, re.IGNORECASE)
        if term_match:
            term = term_match.group(1)
            # Convert Roman numerals to Arabic
            roman_to_arabic = {"I": "1", "II": "2", "III": "3"}
            metadata["term"] = roman_to_arabic.get(term.upper(), term)
        
        # Extract language
        for language, patterns in self.LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, filename):
                    metadata["language"] = language
                    break
            if metadata["language"] != "English":
                break
        
        # Detect sub-subject based on content keywords (will be refined during processing)
        metadata["sub_subject"] = self._infer_sub_subject(metadata["subject"])
        
        return metadata
    
    def _infer_sub_subject(self, subject: str) -> str:
        """
        Infer initial sub-subject based on main subject.
        Will be refined during content processing.
        """
        sub_subject_defaults = {
            "Science": "General Science",
            "Maths": "General Mathematics",
            "English": "English Language",
            "Social_Science": "General Social Science",
            "Tamil": "Tamil Language"
        }
        return sub_subject_defaults.get(subject, "General")
    
    def detect_sub_subject_from_content(self, text: str, subject: str) -> str:
        """
        Detect more specific sub-subject from content.
        
        Args:
            text: Text content
            subject: Main subject
            
        Returns:
            Detected sub-subject
        """
        text_lower = text.lower()
        
        if subject == "Science":
            # Physics keywords
            physics_keywords = [
                "force", "motion", "energy", "gravity", "velocity", 
                "acceleration", "momentum", "wave", "light", "sound",
                "விசை", "இயக்கம்", "ஆற்றல்"
            ]
            # Chemistry keywords
            chemistry_keywords = [
                "atom", "molecule", "element", "compound", "reaction",
                "acid", "base", "chemical", "periodic", "bond",
                "அணு", "மூலக்கூறு", "தனிமம்"
            ]
            # Biology keywords
            biology_keywords = [
                "cell", "organism", "plant", "animal", "photosynthesis",
                "respiration", "digestion", "circulation", "nervous",
                "செல்", "உயிரினம்", "தாவரம்"
            ]
            
            physics_count = sum(1 for kw in physics_keywords if kw in text_lower)
            chemistry_count = sum(1 for kw in chemistry_keywords if kw in text_lower)
            biology_count = sum(1 for kw in biology_keywords if kw in text_lower)
            
            max_count = max(physics_count, chemistry_count, biology_count)
            if max_count > 0:
                if physics_count == max_count:
                    return "Physics"
                elif chemistry_count == max_count:
                    return "Chemistry"
                else:
                    return "Biology"
        
        elif subject == "Maths":
            # Algebra keywords
            algebra_keywords = [
                "equation", "variable", "polynomial", "linear", "quadratic",
                "சமன்பாடு", "மாறி"
            ]
            # Geometry keywords
            geometry_keywords = [
                "triangle", "circle", "angle", "area", "perimeter",
                "முக்கோணம்", "வட்டம்", "பரப்பு"
            ]
            # Arithmetic keywords
            arithmetic_keywords = [
                "number", "fraction", "decimal", "percentage", "ratio",
                "எண்", "பின்னம்", "விகிதம்"
            ]
            
            algebra_count = sum(1 for kw in algebra_keywords if kw in text_lower)
            geometry_count = sum(1 for kw in geometry_keywords if kw in text_lower)
            arithmetic_count = sum(1 for kw in arithmetic_keywords if kw in text_lower)
            
            max_count = max(algebra_count, geometry_count, arithmetic_count)
            if max_count > 0:
                if algebra_count == max_count:
                    return "Algebra"
                elif geometry_count == max_count:
                    return "Geometry"
                else:
                    return "Arithmetic"
        
        elif subject == "Social_Science":
            # History keywords
            history_keywords = [
                "history", "ancient", "medieval", "modern", "civilization",
                "kingdom", "empire", "war", "independence",
                "வரலாறு", "நாகரிகம்", "பேரரசு"
            ]
            # Geography keywords
            geography_keywords = [
                "geography", "continent", "country", "river", "mountain",
                "climate", "map", "ocean", "region",
                "புவியியல்", "கண்டம்", "நாடு"
            ]
            # Civics keywords
            civics_keywords = [
                "government", "democracy", "constitution", "rights",
                "citizen", "parliament", "election", "law",
                "அரசாங்கம்", "ஜனநாயகம்"
            ]
            
            history_count = sum(1 for kw in history_keywords if kw in text_lower)
            geography_count = sum(1 for kw in geography_keywords if kw in text_lower)
            civics_count = sum(1 for kw in civics_keywords if kw in text_lower)
            
            max_count = max(history_count, geography_count, civics_count)
            if max_count > 0:
                if history_count == max_count:
                    return "History"
                elif geography_count == max_count:
                    return "Geography"
                else:
                    return "Civics"
        
        return self._infer_sub_subject(subject)
    
    def enrich_metadata(self, metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Enrich metadata with content-based detection.
        
        Args:
            metadata: Existing metadata
            text: Text content
            
        Returns:
            Enriched metadata
        """
        enriched = metadata.copy()
        
        # Detect sub-subject from content
        subject = metadata.get("subject", "Unknown")
        enriched["sub_subject"] = self.detect_sub_subject_from_content(text, subject)
        
        return enriched


if __name__ == "__main__":
    # Test the metadata extractor
    extractor = MetadataExtractor()
    
    test_files = [
        "7th_Science_Term_II_EM.pdf",
        "7th_Maths_Term-III_TM.pdf",
        "7th_Social_Science_Term_III_EM.pdf",
        "7th_Tamil_Term-III.pdf"
    ]
    
    for filename in test_files:
        metadata = extractor.extract_from_filename(filename)
        print(f"\nFile: {filename}")
        print(f"Metadata: {metadata}")
