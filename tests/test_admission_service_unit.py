"""
Tests unitaires pour le service BentoML d'admission - version TestClient
Aucune dépendance réseau : le service est instancié directement en mémoire.

Lancer avec : pytest test_admission_service_unit.py -v
"""

import time
import jwt
import pytest
from starlette.testclient import TestClient

from src.service import ModelProbaAdmission, JWT_SECRET_KEY, JWT_ALGORITHM

# -- Constantes JWT - DOIVENT correspondre à celles du JWTAuthMiddleware ----
SECRET_KEY = JWT_SECRET_KEY
ALGORITHM = JWT_ALGORITHM


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    asgi_app = ModelProbaAdmission.to_asgi()
    with TestClient(asgi_app) as c:
        yield c


VALID_CREDENTIALS = { "credentials":{"username": "user123", "password": "password123"} }
INVALID_CREDENTIALS = { "credentials":{"username": "user456", "password": "wrongpassword"} }

VALID_PAYLOAD = {
  "input_data": {
    "GRE Score": 320,
    "TOEFL Score": 110,
    "University Rating": 3,
    "SOP": 3.5,
    "LOR": 3.5,
    "CGPA": 8.5,
    "Research": 1,
    }
}


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def get_valid_token(client) -> str:
    """Récupère un token JWT via /login - token au format émis par le service."""
    resp = client.post("/login", json=VALID_CREDENTIALS)
    assert resp.status_code == 200, f"Login échoué : {resp.text}"
    return resp.json()["token"]


def make_expired_token() -> str:
    """Token signé avec la même clé, mais expiré."""
    payload = {"sub": "userA", "exp": int(time.time()) - 3600}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def make_invalid_signature_token() -> str:
    """Token bien formé mais signé avec une mauvaise clé."""
    payload = {"sub": "userA", "exp": int(time.time()) + 3600}
    return jwt.encode(payload, "wrong_secret_key", algorithm=ALGORITHM)


# -------------------------------------------------------------------------
# 1. Authentification JWT
# -------------------------------------------------------------------------

class TestJWTAuthentication:

    def test_missing_token_fails(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 401

    def test_invalid_token_format_fails(self, client):
        headers = {"Authorization": "Token abcdef123456"}  # pas "Bearer"
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=headers)
        assert resp.status_code == 401

    def test_garbage_token_fails(self, client):
        headers = {"Authorization": "Bearer not.a.valid.jwt"}
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=headers)
        assert resp.status_code == 401

    def test_invalid_signature_token_fails(self, client):
        bad_token = make_invalid_signature_token()
        headers = {"Authorization": f"Bearer {bad_token}"}
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=headers)
        assert resp.status_code == 401

    def test_expired_token_fails(self, client):
        expired_token = make_expired_token()
        headers = {"Authorization": f"Bearer {expired_token}"}
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=headers)
        assert resp.status_code == 401

    def test_valid_token_succeeds(self, client):
        token = get_valid_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=headers)
        assert resp.status_code == 200


# -------------------------------------------------------------------------
# 2. API /login
# -------------------------------------------------------------------------

class TestLoginAPI:

    def test_login_with_valid_credentials_returns_token(self, client):
        resp = client.post("/login", json=VALID_CREDENTIALS)
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert isinstance(data["token"], str)
        assert len(data["token"]) > 0
        # Format JWT minimal : header.payload.signature
        assert data["token"].count(".") == 2

    def test_login_with_invalid_credentials_returns_401(self, client):
        resp = client.post("/login", json=INVALID_CREDENTIALS)
        assert resp.status_code == 401

    def test_login_with_unknown_user_returns_401(self, client):
        resp = client.post("/login", json={ "credentials": {"username": "unknown_user", "password": "x"}})
        assert resp.status_code == 401

    def test_login_with_missing_fields_returns_error(self, client):
        resp = client.post("/login", json={ "credentials": {"username": "userA"}})
        assert resp.status_code in (400, 422)


# -------------------------------------------------------------------------
# 3. API /predict
# -------------------------------------------------------------------------

class TestPredictAPI:

    @pytest.fixture(scope="class")
    def auth_headers(self, client):
        token = get_valid_token(client)
        return {"Authorization": f"Bearer {token}"}

    def test_predict_without_token_returns_401(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 401

    def test_predict_with_invalid_token_returns_401(self, client):
        headers = {"Authorization": "Bearer invalid.token.here"}
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=headers)
        assert resp.status_code == 401

    def test_predict_with_valid_data_returns_prediction(self, client, auth_headers):
        resp = client.post("/predict", json=VALID_PAYLOAD, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert isinstance(data["prediction"][0], float)
        assert 0.0 <= data["prediction"][0] <= 1.0

    if False:
        @pytest.mark.parametrize("missing_field", [
            "GRE Score", "TOEFL Score", "University Rating",
            "SOP", "LOR", "CGPA", "Research"
        ])
        def test_predict_with_missing_field_returns_error(self, client, auth_headers, missing_field):
            payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing_field}
            resp = client.post("/predict", json=payload, headers=auth_headers)
            assert resp.status_code in (400, 422)

        @pytest.mark.parametrize("field,bad_value", [
            ("GRE Score", 999),
            ("TOEFL Score", -5),
            ("University Rating", 10),
            ("CGPA", 15.0),
            ("Research", 2),
            ("GRE Score", "not_a_number"),
        ])
        def test_predict_with_out_of_range_values_returns_error(self, client, auth_headers, field, bad_value):
            payload = {**VALID_PAYLOAD, field: bad_value}
            resp = client.post("/predict", json=payload, headers=auth_headers)
            assert resp.status_code == 422

    def test_predict_with_empty_body_returns_error(self, client, auth_headers):
        resp = client.post("/predict", json={}, headers=auth_headers)
        assert resp.status_code == 400