from summariser.utils import auto_lengths, chunk_text


def test_auto_lengths_basic():
    max_len, min_len = auto_lengths("hello world " * 50)
    assert max_len > min_len >= 16


def test_chunk_text():
    text = " ".join(["word"] * 1000)
    parts = chunk_text(text, target_words=200)
    assert len(parts) >= 4
