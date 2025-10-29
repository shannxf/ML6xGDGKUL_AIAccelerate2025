import os
import json
from typing import List, Optional, Tuple, Dict, Any


def _read_text_file(path: str, encoding: Optional[str] = "utf-8") -> str:
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader  # lightweight, pure-Python
    except Exception as e:
        raise ImportError("pypdf is required for PDF extraction. Add 'pypdf' to dependencies.") from e
    reader = PdfReader(path)
    chunks = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            chunks.append(txt)
    return "\n\n".join(chunks)


def _read_docx(path: str) -> str:
    try:
        from docx import Document  # python-docx
    except Exception as e:
        raise ImportError("python-docx is required for DOCX extraction. Add 'python-docx' to dependencies.") from e
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs]
    return "\n".join([p for p in paras if p])


def _read_html(path: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except Exception as e:
        raise ImportError("beautifulsoup4 is required for HTML extraction. Add 'beautifulsoup4' to dependencies.") from e
    html = _read_text_file(path)
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _read_csv(path: str) -> str:
    import csv
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(", ".join(row))
    return "\n".join(rows)


def _read_json(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        # Fall back to raw text
        return _read_text_file(path)


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _extract_one(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"ok": False, "file": path, "type": "missing", "text": "", "error": "file_not_found"}

    ext = _ext(path)
    try:
        if ext in {".txt", ".md", ".rst", ".log"}:
            text = _read_text_file(path)
            ftype = "text"
        elif ext == ".pdf":
            text = _read_pdf(path)
            ftype = "pdf"
        elif ext == ".docx":
            text = _read_docx(path)
            ftype = "docx"
        elif ext in {".html", ".htm"}:
            text = _read_html(path)
            ftype = "html"
        elif ext == ".csv":
            text = _read_csv(path)
            ftype = "csv"
        elif ext == ".json":
            text = _read_json(path)
            ftype = "json"
        else:
            # Unknown: try text fallback
            text = _read_text_file(path)
            ftype = "unknown-text"
        return {"ok": True, "file": path, "type": ftype, "text": text, "chars": len(text)}
    except Exception as e:
        return {"ok": False, "file": path, "type": ext or "unknown", "text": "", "error": str(e)}


def extract_text(path: str = "", paths: Optional[List[str]] = None, max_chars: int = 50000, join: bool = True) -> Dict[str, Any]:
    """Extract plain text from common file types. Safely truncates to max_chars.

    Args:
        path: single file path
        paths: list of file paths (overrides 'path' if provided)
        max_chars: truncate joined text to this size
        join: when multiple files, whether to join into a single 'text' field

    Returns:
        { ok, items: [ {file,type,text,chars,ok,error?} ], text?, total_chars }
    """
    file_list = paths or ([path] if path else [])
    if not file_list:
        return {"ok": False, "items": [], "text": "", "total_chars": 0, "error": "no_paths_provided"}

    items: List[Dict[str, Any]] = []
    total = 0
    for p in file_list:
        item = _extract_one(p)
        items.append(item)
        total += len(item.get("text", ""))

    if join:
        joined = []
        for it in items:
            if it.get("ok") and it.get("text"):
                joined.append(f"===== {os.path.basename(it['file'])} ({it['type']}) =====\n{it['text']}")
        big = "\n\n".join(joined)
        if len(big) > max_chars:
            big = big[:max_chars]
        return {"ok": any(it.get("ok") for it in items), "items": items, "text": big, "total_chars": total}

    return {"ok": any(it.get("ok") for it in items), "items": items, "total_chars": total}


# ADK tool descriptor for robust auto-calling
tool = {
    "name": "extract_text",
    "description": "Extract plain text from files (txt, md, pdf, docx, html, csv, json).",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Single file path"},
            "paths": {"type": "array", "items": {"type": "string"}},
            "max_chars": {"type": "integer"},
            "join": {"type": "boolean"}
        }
    },
    "fn": extract_text,
}
