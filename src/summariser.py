import sys
import argparse
from transformers import pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "text", nargs="?", help="Text to summarize. If omitted, reads from stdin."
    )
    ap.add_argument(
        "--model", default="sshleifer/distilbart-cnn-12-6", help="HF model id"
    )
    ap.add_argument("--max_len", type=int, default=90)
    ap.add_argument("--min_len", type=int, default=25)
    args = ap.parse_args()

    text = args.text or sys.stdin.read().strip()
    if not text:
        print('Usage: python src/summarizer.py "Your text..."')
        sys.exit(1)

    summarizer = pipeline(
        "summarization", model=args.model, device_map="auto"
    )  # will use MPS on M1
    out = summarizer(
        text, max_length=args.max_len, min_length=args.min_len, do_sample=False
    )
    print(out[0]["summary_text"])


if __name__ == "__main__":
    main()
