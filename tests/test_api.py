from fastapi.testclient import TestClient

from api.main import app


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_pages_and_encounter_endpoints():
    c = TestClient(app)
    assert c.get("/").status_code == 200
    assert c.get("/app").status_code == 200
    assert c.get("/api/encounters/..%2Fetc").status_code == 404
    assert c.get("/api/encounters/day9_consultation99").status_code == 404
    listing = c.get("/api/encounters").json()
    if listing:  # verdict files are gitignored, so only check when present
        d = c.get(f"/api/encounters/{listing[0]['id']}").json()
        assert d["turns"] and d["claims"] and "reference_note" not in d
