"""
File Parsing Helper for Resume Evaluator.
Extracts clean, normalized text from Plain Text, PDF, and DOCX files.
"""

import io
from typing import Tuple


def extract_text_from_file(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Extracts text content from uploaded file bytes based on filename extension.
    Returns (extracted_text, status_message).
    """
    lower_name = filename.lower()

    if lower_name.endswith(".txt"):
        try:
            return file_bytes.decode("utf-8").strip(), "Success"
        except UnicodeDecodeError:
            try:
                return file_bytes.decode("latin-1").strip(), "Success (latin-1)"
            except Exception as e:
                return "", f"Failed to decode text file: {e}"

    elif lower_name.endswith(".pdf"):
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            extracted = "\n\n".join(pages_text).strip()
            if not extracted:
                return "", "PDF contained no extractable text (may be a scanned image)."
            return extracted, f"Extracted {len(pages_text)} pages"
        except Exception as e:
            return "", f"PDF parsing error: {e}"

    elif lower_name.endswith(".docx"):
        try:
            import docx

            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            extracted = "\n".join(paragraphs).strip()
            if not extracted:
                return "", "DOCX contained no text."
            return extracted, "Success"
        except Exception as e:
            return "", f"DOCX parsing error: {e}"

    else:
        # Fallback attempt text decode
        try:
            return file_bytes.decode("utf-8").strip(), "Success (generic decode)"
        except Exception:
            return "", f"Unsupported file type '{filename}'. Please upload PDF, DOCX, or TXT."
