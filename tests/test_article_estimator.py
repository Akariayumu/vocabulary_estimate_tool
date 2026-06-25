from fastapi.testclient import TestClient

from server.main import app
from vocab_estimator.article_estimator import estimate_article, tokenize_article


def test_tokenize_article_removes_stopwords():
    tokens = tokenize_article("This is a simple article about climate change and education.")

    assert "this" not in tokens
    assert "is" not in tokens
    assert "article" in tokens
    assert "climate" in tokens


def test_estimate_article_uses_stage_vocab():
    result = estimate_article(
        "Students analyze evidence, compare information, and develop vocabulary "
        "through reading, writing, science, technology, culture, and education."
    )

    assert 0 <= result["difficulty_median"] <= 1
    assert result["estimated_vocab"] > 0
    assert result["level"]
    assert result["coverage"]["stage_vocab"] > 0
    assert result["coverage"]["difficulty_distribution"]["median"] == result["difficulty_median"]


def test_estimate_article_v2_endpoint():
    client = TestClient(app)
    response = client.post(
        "/api/v2/estimate/article",
        json={
            "article": (
                "Students analyze evidence, compare information, and develop vocabulary "
                "through reading, writing, science, technology, culture, and education."
            )
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert set(["difficulty_median", "estimated_vocab", "level", "coverage"]).issubset(data)
    assert data["coverage"]["stage_vocab"] > 0
