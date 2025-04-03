import os
import fitz  # PyMuPDF
import pdfplumber
import tiktoken

class PDFProcessor:
    def extract_text_chunks(self, pdf_path, max_tokens=2000, overlap_tokens=50):
        chunks = []
        enc = tiktoken.encoding_for_model("gpt-4o-mini")

        def paragraph_chunks(text):
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

        # Text aus Seiten
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text()
                metadata = {"source": os.path.basename(pdf_path), "page": page_num + 1}
                for chunk in paragraph_chunks(text):
                    chunks.append({"content": chunk, "metadata": metadata})

        # Tabellen mit pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    table_text = "\n".join([
                        " | ".join([str(cell) if cell else "" for cell in row])
                        for row in table
                    ])
                    metadata = {"source": os.path.basename(pdf_path), "page": i + 1}
                    for chunk in paragraph_chunks(table_text):
                        chunks.append({"content": chunk, "metadata": metadata})

        return chunks
