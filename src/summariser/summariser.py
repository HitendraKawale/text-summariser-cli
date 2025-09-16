from typing import List
from transformers import pipeline, AutoTokenizer
from .utils import auto_lengths, chunk_text


def summarise_text(
    text: str, model_id: str, beams: int, no_repeat: int, short_flag: bool
) -> str:
    summarizer = pipeline("summarization", model=model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    max_len, min_len = auto_lengths(text, short_flag)

    # If too long for model context, chunk -> partial summaries -> final pass
    max_tokens = int(getattr(tokenizer, "model_max_length", 1024))
    if len(tokenizer.encode(text)) > max_tokens:
        parts: List[str] = chunk_text(text, target_words=350)
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
