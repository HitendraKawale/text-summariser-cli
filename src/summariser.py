#!/usr/bin/env python3
import sys, argparse, csv, re
from pathlib import Path
from typing import Optional, List


# --- Optional imports are gated so the script still runs without them ---
def _opt_import(mod, pip_name=None):
    try:
        return __import__(mod)
    except ImportError as e:
        raise ImportError(
            f"Missing optional dependency '{mod}'. "
            f"Install with: pip install {pip_name or mod}"
        ) from e


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_md(path: Path) -> str:
    # Markdown is just text for our purposes
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
    # Combine all text-like cells; good enough for quick summarisation
    buf = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            buf.append(" ".join(cell.strip() for cell in row if cell.strip()))
    return "\n".join(buf)


def read_html_file(path: Path) -> str:
    bs4 = _opt_import("bs4", "beautifulsoup4")
    soup = bs4.BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    # Remove scripts/styles
    for bad in soup(["script", "style", "noscript"]):
        bad.extract()
    return soup.get_text(separator="\n")


def read_url(url: str) -> str:
    # Try robust extraction via trafilatura; fallback to BeautifulSoup
    requests = _opt_import("requests", "requests")
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()

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
        if ext in (".txt",):
            return read_txt(p)
        if ext in (".md",):
            return read_md(p)
        if ext in (".pdf",):
            return read_pdf(p)
        if ext in (".docx",):
            return read_docx(p)
        if ext in (".csv",):
            return read_csv_file(p)
        if ext in (".html", ".htm"):
            return read_html_file(p)
        # default: treat as text
        return read_txt(p)

    if text_arg:
        return text_arg

    data = sys.stdin.read()
    if data.strip():
        return data
    raise SystemExit(
        "No input provided. Use --file FILE, --url URL, a text argument, or pipe via stdin."
    )


# --- Summarisation ---
def auto_lengths(text: str, short_cut=False):
    n = len(re.findall(r"\w+", text))
    if short_cut or n < 300:
        max_len = max(32, min(160, int(n * 0.6 + 30)))
        min_len = max(16, min(max_len - 8, int(n * 0.3 + 12)))
    else:
        max_len, min_len = 180, 60
    if min_len >= max_len:
        min_len = max(16, max_len // 2)
    return max_len, min_len


def chunk_text(text: str, target_words=400) -> List[str]:
    # Simple word-based chunker, preserves paragraphs where possible
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= target_words:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def summarise_text(
    text: str, model_id: str, beams: int, no_repeat: int, short_flag: bool
):
    from transformers import pipeline, AutoTokenizer

    # use device_map="auto" so M1 uses MPS; CPU elsewhere
    summarizer = pipeline("summarization", model=model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # If text is long, summarise in chunks -> then summarise the summaries
    max_len, min_len = auto_lengths(text, short_flag)
    max_tokens = getattr(tokenizer.model_max_length, "__int__", lambda: 1024)()

    # Rough heuristic: if text looks too long, chunk it
    if len(tokenizer.encode(text)) > max_tokens:
        parts = chunk_text(text, target_words=350)
        partials = []
        for part in parts:
            ml, mn = auto_lengths(part, short_flag)
            out = summarizer(
                part,
                max_length=ml,
                min_length=mn,
                do_sample=False,
                num_beams=beams,
                no_repeat_ngram_size=no_repeat,
                early_stopping=True,
            )[0]["summary_text"]
            partials.append(out)
        combined = " ".join(partials)
        # final pass
        ml, mn = auto_lengths(combined, True)
        final = summarizer(
            combined,
            max_length=ml,
            min_length=mn,
            do_sample=False,
            num_beams=max(4, beams),
            no_repeat_ngram_size=max(no_repeat, 3),
            early_stopping=True,
        )[0]["summary_text"]
        return final

    out = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        num_beams=beams,
        no_repeat_ngram_size=no_repeat,
        early_stopping=True,
    )[0]["summary_text"]
    return out


def main():
    ap = argparse.ArgumentParser(description="Summarise text, files, or webpages.")
    ap.add_argument(
        "text",
        nargs="?",
        help="Text to summarise (optional if --file/--url/stdin is used)",
    )
    ap.add_argument(
        "-f", "--file", help="Path to input file (.txt, .md, .pdf, .docx, .csv, .html)"
    )
    ap.add_argument("-u", "--url", help="Webpage URL to summarise")
    ap.add_argument(
        "--model", default="sshleifer/distilbart-cnn-12-6", help="HF model id"
    )
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--no-repeat", type=int, default=3, dest="no_repeat")
    ap.add_argument(
        "--short", action="store_true", help="Bias towards shorter summaries"
    )
    ap.add_argument("-o", "--out", help="Write summary to file")
    args = ap.parse_args()

    text = load_input(args.text, args.file, args.url).strip()
    if not text:
        raise SystemExit("Input is empty after parsing.")

    summary = summarise_text(text, args.model, args.beams, args.no_repeat, args.short)

    if args.out:
        Path(args.out).write_text(summary, encoding="utf-8")
        print(f"✅ Summary written to {args.out}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
