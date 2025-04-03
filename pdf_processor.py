
# rag_core/pdf_processor.py
import fitz
import pdfplumber
import os
import tiktoken
from typing import List, Dict

class PDFProcessor:
    def extract_text_chunks(self, pdf_path: str, max_tokens=2000, overlap_tokens=50) -> List[Dict]:
        chunks = []
        enc = tiktoken.encoding_for_model("gpt-4")

        def paragraph_chunks(text: str) -> List[str]:
            paragraphs = text.split("\n\n")
            token_buffer = []
            current_tokens = 0
            result = []

            for para in paragraphs:
                para_tokens = enc.encode(para)
                if current_tokens + len(para_tokens) > max_tokens:
                    result.append(enc.decode(token_buffer))
                    token_buffer = token_buffer[-overlap_tokens:] + para_tokens
                    current_tokens = len(token_buffer)
                else:
                    token_buffer += para_tokens
                    current_tokens += len(para_tokens)

            if token_buffer:
                result.append(enc.decode(token_buffer))
            return result

        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    metadata = {"source": os.path.basename(pdf_path), "page": page_num + 1}
                    for chunk in paragraph_chunks(text):
                        chunks.append({"content": chunk, "metadata": metadata})
        except Exception as e:
            print(f"Fehler bei Text in {pdf_path}: {e}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table in tables:
                        table_text = "\n".join([
                            " | ".join([str(cell) if cell is not None else "" for cell in row])
                            for row in table if row
                        ])
                        metadata = {"source": os.path.basename(pdf_path), "page": i + 1}
                        for chunk in paragraph_chunks(table_text):
                            chunks.append({"content": chunk, "metadata": metadata})
        except Exception as e:
            print(f"Fehler bei Tabellen in {pdf_path}: {e}")

        return chunks
