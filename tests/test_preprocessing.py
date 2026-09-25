from cyberbully.preprocessing import clean_text


def test_clean_text_basic():
    raw = "Check out https://example.com/bad and @john_doe you are terrible"
    cleaned = clean_text(raw)
    assert "[url]" in cleaned
    assert "[user]" in cleaned
    assert "https://example.com/bad" not in cleaned
    assert "@john_doe" not in cleaned


def test_clean_text_preserves_leetspeak():
    raw = "you are a b@st@rd and an @$$h0le"
    cleaned = clean_text(raw)
    assert "b@st@rd" in cleaned
    assert "@$$h0le" in cleaned


def test_clean_text_hashtags():
    raw = "Stop being so rude #cyberbullying #toxic"
    cleaned = clean_text(raw)
    assert "cyberbullying" in cleaned
    assert "toxic" in cleaned
    assert "#" not in cleaned


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""
