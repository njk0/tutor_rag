"""
Language Detector Module
Detects query language (English/Tamil) for appropriate response generation.
"""
import re
from typing import Tuple


class LanguageDetector:
    """Detects language of text, specifically English and Tamil."""
    
    # Tamil Unicode range
    TAMIL_RANGE_START = 0x0B80
    TAMIL_RANGE_END = 0x0BFF
    
    def __init__(self):
        pass
    
    def detect(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            'Tamil' or 'English'
        """
        tamil_chars = 0
        english_chars = 0
        
        for char in text:
            code_point = ord(char)
            
            # Check if Tamil character
            if self.TAMIL_RANGE_START <= code_point <= self.TAMIL_RANGE_END:
                tamil_chars += 1
            # Check if English letter
            elif char.isalpha() and code_point < 128:
                english_chars += 1
        
        # If more than 20% Tamil characters, consider it Tamil
        total_chars = tamil_chars + english_chars
        if total_chars == 0:
            return "English"
        
        tamil_ratio = tamil_chars / total_chars
        if tamil_ratio > 0.2:
            return "Tamil"
        
        return "English"
    
    def contains_tamil(self, text: str) -> bool:
        """
        Check if text contains any Tamil characters.
        
        Args:
            text: Text to check
            
        Returns:
            True if Tamil characters are found
        """
        for char in text:
            code_point = ord(char)
            if self.TAMIL_RANGE_START <= code_point <= self.TAMIL_RANGE_END:
                return True
        return False
    
    def get_language_info(self, text: str) -> Tuple[str, float, float]:
        """
        Get detailed language information.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (detected_language, tamil_ratio, english_ratio)
        """
        tamil_chars = 0
        english_chars = 0
        
        for char in text:
            code_point = ord(char)
            
            if self.TAMIL_RANGE_START <= code_point <= self.TAMIL_RANGE_END:
                tamil_chars += 1
            elif char.isalpha() and code_point < 128:
                english_chars += 1
        
        total_chars = tamil_chars + english_chars
        if total_chars == 0:
            return "English", 0.0, 0.0
        
        tamil_ratio = tamil_chars / total_chars
        english_ratio = english_chars / total_chars
        
        detected = "Tamil" if tamil_ratio > 0.2 else "English"
        
        return detected, tamil_ratio, english_ratio


if __name__ == "__main__":
    # Test the language detector
    detector = LanguageDetector()
    
    test_texts = [
        "What are the properties of alcohol?",
        "மது பற்றிய பண்புகள் என்ன?",
        "Hello, இது mixed text",
        "Solve the equation 2x + 5 = 15"
    ]
    
    for text in test_texts:
        lang, tamil_r, english_r = detector.get_language_info(text)
        print(f"\nText: {text}")
        print(f"  Detected: {lang}")
        print(f"  Tamil ratio: {tamil_r:.2%}")
        print(f"  English ratio: {english_r:.2%}")
