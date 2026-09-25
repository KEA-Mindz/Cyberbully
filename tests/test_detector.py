import pytest
from cyberbully import CyberbullyingDetector, is_cyberbullying, predict, explain


@pytest.fixture(scope="module")
def detector():
    return CyberbullyingDetector(auto_download=False)


def test_detector_inference(detector):
    res_pos = detector.predict("you are pathetic and useless")
    assert res_pos.is_cyberbullying is True
    assert res_pos.score > res_pos.threshold
    assert res_pos.label == "cyberbullying"

    res_neg = detector.predict("Have a fantastic day!")
    assert res_neg.is_cyberbullying is False
    assert res_neg.score < res_neg.threshold
    assert res_neg.label == "not_cyberbullying"


def test_batch_prediction(detector):
    texts = [
        "Have a wonderful and blessed day!",
        "Nobody likes you, you are ugly and pathetic",
    ]
    results = detector.predict(texts)
    assert len(results) == 2
    assert results[0].is_cyberbullying is False
    assert results[1].is_cyberbullying is True


def test_convenience_functions():
    assert is_cyberbullying("You are terrible and disgusting") is True
    assert is_cyberbullying("Thank you for helping me out today") is False

    exp = explain("You are so annoying")
    assert "cyberbullying_score" in exp
    assert "primary_intent_category" in exp
