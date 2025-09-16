import sys, csv
from pathlib import Path
from typing import Optional


def _opt_import(mod, pip_name=None):
    try:
        return __import__(mod)
    except ImportError as e:
        raise ImportError(
            f"Missing optional dependency '{mod}'. Install with: pip install {pip_name or mod}"
        ) from e


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_md(path: Path) -> str:
    return read_txt(path)


def read_pdf(path: Path) -> str:
    pdfplumber = _opt_import("pdfplumber", "pdfplumber")
    text = []
    with pdfplumber.open(str(path)) as pdf:
        for p in pdf.pages:
            text.append(p.extract_text() or "")
    return "\n".join(text).strip()


def read_docx(path: Path) -> str:
    docx = _opt_import("docx", "python-docx")
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def read_csv_file(path: Path) -> str:
    buf = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            buf.append(" ".join(cell.strip() for cell in row if cell.strip()))
    return "\n".join(buf)


def read_html_file(path: Path) -> str:
    bs4 = _opt_import("bs4", "beautifulsoup4")
    soup = bs4.BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for bad in soup(["script", "style", "noscript"]):
        bad.extract()
    return soup.get_text(separator="\n")


def read_url(url: str) -> str:
    requests = _opt_import("requests", "requests")
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()

    # Try trafilatura first (better boilerplate removal), fallback to bs4
    try:
        trafilatura = _opt_import("trafilatura", "trafilatura")
        extracted = trafilatura.extract(resp.text, url=url, include_tables=False)
        if extracted and extracted.strip():
            return extracted.strip()
    except ImportError:
        pass

    bs4 = _opt_import("bs4", "beautifulsoup4")
    soup = bs4.BeautifulSoup(resp.text, "lxml")
    for bad in soup(["script", "style", "noscript"]):
        bad.extract()
    return soup.get_text(separator="\n")


def load_input(
    text_arg: Optional[str], file_arg: Optional[str], url_arg: Optional[str]
) -> str:
    if url_arg:
        return read_url(url_arg)

    if file_arg:
        p = Path(file_arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        ext = p.suffix.lower()
        if ext == ".txt":
            return read_txt(p)
        if ext == ".md":
            return read_md(p)
        if ext == ".pdf":
            return read_pdf(p)
        if ext == ".docx":
            return read_docx(p)
        if ext == ".csv":
            return read_csv_file(p)
        if ext in (".html", ".htm"):
            return read_html_file(p)
        return read_txt(p)  # default

    if text_arg:
        return text_arg

    data = sys.stdin.read()
    if data.strip():
        return data
    raise SystemExit(
        "No input provided. Use --file FILE, --url URL, a text argument, or pipe via stdin."
    )
