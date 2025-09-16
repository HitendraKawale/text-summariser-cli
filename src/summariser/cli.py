#!/usr/bin/env python3
import argparse
from pathlib import Path
from .loaders import load_input
from .summariser import summarise_text


def build_parser():
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
    return ap


def main():
    args = build_parser().parse_args()
    text = load_input(args.text, args.file, args.url).strip()
    if not text:
        raise SystemExit("Input is empty after parsing.")
    summary = summarise_text(text, args.model, args.beams, args.no_repeat, args.short)

    if args.out:
        Path(args.out).write_text(summary, encoding="utf-8")
        print(f"✅ Summary written to {args.out}")
    else:
        print(summary)
