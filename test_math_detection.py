# -*- coding: utf-8 -*-
"""Test math problem detection for Tamil queries"""
import sys
sys.path.append('c:/tutor_rag')

from src.query_classifier import QueryClassifier

qc = QueryClassifier()

# Test queries
test_queries = [
    "ஒரு உலோகக் கலவையில் 26% தாமிரம் உள்ளது. 260 கிராம் தாமிரத்தைப் பெற எத்தனை அளவு உலோகக் கலவை தேவைப்படும்?",
    "Solve: 2x + 5 = 15",
    "What is photosynthesis?",
    "50% of 200 is how much?",
    "கணக்கிடு: 5 + 3 = ?",
]

print("Testing math problem detection:")
print("=" * 50)
for query in test_queries:
    is_math = qc.is_math_problem(query)
    subject, conf = qc.classify_subject(query)
    print(f"Query: {query[:60]}...")
    print(f"  Is Math Problem: {is_math}")
    print(f"  Subject: {subject} (confidence: {conf:.2f})")
    print()
