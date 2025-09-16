import re
from typing import List


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
