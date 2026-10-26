from fastapi.testclient import TestClient

from semsearch import api


def _client(offline_settings):
    api.build_state(offline_settings, system="all")
    return TestClient(api.app)


def test_health_reports_the_loaded_index(offline_settings):
    response = _client(offline_settings).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents"] == 12
    assert "hybrid" in body["systems"]


def test_search_returns_ranked_hits_with_snippets(offline_settings):
    response = _client(offline_settings).get(
        "/search", params={"q": "reciprocal rank fusion", "system": "bm25", "top_k": 5}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["system"] == "bm25"
    assert body["took_ms"] >= 0
    assert body["hits"][0]["doc_id"] == "d1"
    assert "<mark>" in body["hits"][0]["snippet"]
    assert [hit["rank"] for hit in body["hits"]] == list(range(1, len(body["hits"]) + 1))


def test_search_without_highlighting(offline_settings):
    response = _client(offline_settings).get(
        "/search", params={"q": "inverted index", "system": "bm25", "highlight": False}
    )
    assert "<mark>" not in response.json()["hits"][0]["snippet"]


def test_unknown_system_is_a_400(offline_settings):
    response = _client(offline_settings).get("/search", params={"q": "x", "system": "magic"})
    assert response.status_code == 400
    assert "Unknown system" in response.json()["detail"]


def test_empty_query_is_rejected_by_validation(offline_settings):
    assert _client(offline_settings).get("/search", params={"q": ""}).status_code == 422


def test_all_four_systems_answer_the_same_query(offline_settings):
    client = _client(offline_settings)
    for system in ["bm25", "dense", "hybrid", "hybrid+rerank"]:
        response = client.get("/search", params={"q": "what does nDCG measure", "system": system})
        assert response.status_code == 200, system
        assert response.json()["hits"], system
