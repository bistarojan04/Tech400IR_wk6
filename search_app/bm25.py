# search_app/bm25.py

import os
import math
from collections import Counter
from django.conf import settings  # Import to access static files

# BM25 parameters
k1 = 1.5
b = 0.75

# Get the path to the static directory
dataset_path = os.path.join(settings.BASE_DIR, 'search_app', 'static', 'search_app', 'dataset')

def load_documents(folder_path):
    documents = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(f.read())
    return documents


def tokenize(text):
    return text.lower().split()

def calculate_tf(tokenized_doc):
    return Counter(tokenized_doc)

def calculate_idf(docs):
    N = len(docs)
    df = Counter()
    for doc in docs:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] += 1
    
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log(1 + (N - freq + 0.5) / (freq + 0.5))
    
    return idf

def bm25_score(query, doc_tokens, idf, avg_doc_len, doc_tf):
    score = 0
    doc_len = len(doc_tokens)
    
    for term in query:
        if term in doc_tf:
            term_freq = doc_tf[term]
            numerator = term_freq * (k1 + 1)
            denominator = term_freq + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf.get(term, 0) * (numerator / denominator)
    
    return score

documents = load_documents(dataset_path)
tokenized_docs = [tokenize(doc) for doc in documents]
doc_tfs = [calculate_tf(doc) for doc in tokenized_docs]
doc_lengths = [len(doc) for doc in tokenized_docs]
avg_doc_len = sum(doc_lengths) / len(doc_lengths)
idf = calculate_idf(tokenized_docs)

def search_bm25(query, top_n=5):
    tokenized_query = tokenize(query)
    scores = []
    
    for idx, doc_tokens in enumerate(tokenized_docs):
        score = bm25_score(tokenized_query, doc_tokens, idf, avg_doc_len, doc_tfs[idx])
        scores.append((idx, score))
    
    ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    results = []
    for idx, _ in ranked_docs:
        results.append({
            "document_id": idx + 1,
            "document": documents[idx]
        })
    
    return results
