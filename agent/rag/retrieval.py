import os
from typing import List, Dict
from rank_bm25 import BM25Okapi
import re

class SimpleRetriever:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.chunks: List[Dict] = []
        self.bm25 = None
        self._load_and_chunk_docs()
        self._build_index()

    def _load_and_chunk_docs(self):
        """Loads markdown files and splits them into chunks."""
        for filename in os.listdir(self.docs_dir):
            if not filename.endswith(".md"):
                continue
            
            filepath = os.path.join(self.docs_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                      
            doc_id_base = filename.replace(".md", "")
            
            # Split by headers
            sections = re.split(r'(^#+\s.*$)', content, flags=re.MULTILINE)
            
            current_header = "Intro"
            chunk_counter = 0
            
            # sections[0] might be empty or intro text
            if sections[0].strip():
                 self.chunks.append({
                    "id": f"{doc_id_base}::chunk{chunk_counter}",
                    "content": sections[0].strip(),
                    "source": filename,
                    "header": current_header
                })
                 chunk_counter += 1

            for i in range(1, len(sections), 2):
                header = sections[i].strip()
                text = sections[i+1].strip() if i+1 < len(sections) else ""
                
                if text:
                    self.chunks.append({
                        "id": f"{doc_id_base}::chunk{chunk_counter}",
                        "content": f"{header}\n{text}",
                        "source": filename,
                        "header": header
                    })
                    chunk_counter += 1

    def _build_index(self):
        if not self.chunks:
            return
            
        tokenized_corpus = [self._tokenize(chunk["content"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        if not self.chunks or not self.bm25:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zip scores with chunks and sort
        scored_chunks = []
        for i, score in enumerate(scores):
            # Lower threshold or remove > 0 check to ensure we get something if k is large
            # But > 0 is good for BM25. Let's keep it but ensure tokenization is robust.
            if score > -1: 
                chunk = self.chunks[i].copy()
                chunk["score"] = score
                scored_chunks.append(chunk)
        
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:k]

