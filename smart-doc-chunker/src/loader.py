import os
import re


def load_document(file_path: str) -> str:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    print(f"Loading file: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        print("Detected type: pdf")
        text = _load_pdf(file_path)
    elif ext == ".txt":
        print("Detected type: txt")
        text = _load_txt(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: .pdf, .txt"
        )

    text = _normalize(text)
    print(f"Extraction complete. Characters: {len(text)}")
    return text


def _load_pdf(file_path: str) -> str:
    try:
        import PyPDF2
    except ImportError:
        raise ImportError(
            "PyPDF2 is required for PDF extraction. "
            "Install it with: pip install PyPDF2"
        )

    pages = []
    with open(file_path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page in reader.pages:
            content = page.extract_text()
            if content is None:
                continue
            content = content.strip()
            if content:
                pages.append(content)

    return "\n\n".join(pages)


def _load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read().strip()


def _normalize(text: str) -> str:
    # Collapse runs of spaces/tabs to a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse more than two consecutive newlines to exactly two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
