"""Real image/JWT/OIDC compatibility; provider HTTP and database IO are faked.

Run in the isolated candidate environment with the cloud dependencies installed.
No Google sign-in, production credentials, database writes or GPU work occur here.
"""
import asyncio
import base64
import hashlib
import importlib.metadata
import io
import json
import time
import uuid
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlparse

import pytest

pytest.importorskip("authlib", reason="cloud dependency compatibility needs Authlib")
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from PIL import Image
from starlette.requests import Request

import thumbnail
from cloud import auth, oauth


@pytest.mark.parametrize("package,expected", [
    ("Pillow", "13.0.0"), ("Authlib", "1.9.5"),
    ("PyJWT", "2.15.1"), ("joserfc", "1.8.0"),
])
def test_candidate_installs_reviewed_dependency_versions(package, expected):
    # Installed metadata, not the requirement line: old environments cannot
    # supply evidence for the reviewed pins. Behavioral checks follow below.
    assert importlib.metadata.version(package) == expected


@pytest.mark.parametrize("format,mode", [("PNG", "RGBA"), ("JPEG", "RGB")])
def test_uploaded_image_reencodes_and_finalizes_with_marks(tmp_path, format, mode):
    source = tmp_path / ("source.png" if format == "PNG" else "source.jpg")
    Image.new(mode, (1600, 1000), "blue").save(source, format)
    part = thumbnail._image_part(str(source))
    assert part.inline_data.mime_type == "image/jpeg"
    with Image.open(io.BytesIO(part.inline_data.data)) as decoded:
        decoded.load()
        assert decoded.mode == "RGB" and decoded.size == (1600, 1000)
        assert decoded.getpixel((800, 500))[2] > 240
        output = tmp_path / "final.jpg"
        thumbnail.finalize_thumbnail(decoded, str(output))
    with Image.open(output) as final:
        final.load()
        assert final.format == "JPEG" and final.size == (1280, 720)
        assert final.getpixel((640, 360))[2] > 240
        assert "OpenShorts" in final.getexif()[0x0131]
        assert b"trainedAlgorithmicMedia" in final.info["xmp"]
    assert 0 < output.stat().st_size <= 2 * 1024 * 1024


@pytest.fixture
def session_secret(monkeypatch):
    secret = "synthetic-test-session-secret-not-for-production-or-real-identities"
    monkeypatch.setattr(auth, "settings", SimpleNamespace(jwt_secret=secret))
    return secret


def test_application_session_jwt_roundtrip(session_secret):
    before = int(time.time())
    token = auth.issue_jwt("fixture-user", "fixture@example.invalid")
    claims = auth._decode_jwt(token)
    assert claims["sub"] == "fixture-user"
    assert claims["email"] == "fixture@example.invalid"
    assert before <= claims["iat"] <= int(time.time())
    assert claims["exp"] - claims["iat"] == 30 * 24 * 60 * 60


@pytest.mark.parametrize("fault", ["expired", "bad_signature", "wrong_algorithm", "altered_payload"])
def test_application_session_jwt_rejects_invalid_tokens(session_secret, fault):
    claims = {"sub": "fixture-user", "iat": int(time.time()), "exp": int(time.time()) + 600}
    if fault == "expired":
        claims["exp"] = int(time.time()) - 600
    key = "different-synthetic-secret-at-least-32-bytes" if fault == "bad_signature" else session_secret
    token = jwt.encode(claims, key, algorithm="HS384" if fault == "wrong_algorithm" else "HS256")
    if fault == "altered_payload":
        header, _, signature = token.split(".")
        payload = base64.urlsafe_b64encode(json.dumps({**claims, "sub": "attacker"}).encode()).rstrip(b"=").decode()
        token = ".".join((header, payload, signature))
    assert auth._decode_jwt(token) is None


@pytest.fixture
def oidc_flow(monkeypatch):
    """Real login/callback and Authlib; fake only provider HTTP and database IO."""
    monkeypatch.setattr(oauth, "settings", SimpleNamespace(
        google_auth_enabled=True, google_client_id="fixture-client",
        google_client_secret="synthetic-client-secret", frontend_url="https://fixture-app.invalid"))
    monkeypatch.setattr(oauth, "oauth", None)
    monkeypatch.setenv("OAUTH_REDIRECT_URI", "https://fixture-app.invalid/callback")
    oauth.register()
    client = oauth.oauth.google
    # Authlib 1.3 uses httpx; 1.9 uses the httpx2 already present in both locks.
    # Use its actual transport implementation, without replacing its OAuth logic.
    from authlib.integrations.httpx_client import oauth2_client
    # The installed 1.9 integration renamed this module binding to httpx2;
    # the baseline uses httpx. This changes only the fake HTTP boundary.
    http = getattr(oauth2_client, "httpx2", None) or oauth2_client.httpx
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key(), as_dict=True)
    jwk.update(kid="fixture-key", use="sig", alg="RS256")
    calls, state, database_calls = [], {}, []
    existing_user = SimpleNamespace(id=uuid.UUID(int=1), email="fixture@example.invalid",
                                    google_sub="fixture-user", last_login_at=None)

    class DatabaseSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def begin(self):
            return self
        async def execute(self, statement):
            database_calls.append("read_user")
            return SimpleNamespace(scalar_one_or_none=lambda: existing_user)
        def add(self, record):
            database_calls.append(record)

    monkeypatch.setattr(oauth.database, "session", DatabaseSession)

    def provider(request):
        calls.append((request.method, str(request.url)))
        if str(request.url) == "https://accounts.google.com/.well-known/openid-configuration":
            return http.Response(200, json={
                "issuer": "https://accounts.google.com",
                "authorization_endpoint": "https://fixture-idp.invalid/authorize",
                "token_endpoint": "https://fixture-idp.invalid/token",
                "jwks_uri": "https://fixture-idp.invalid/jwks",
                "id_token_signing_alg_values_supported": ["RS256"],
            })
        if str(request.url) == "https://fixture-idp.invalid/token":
            assert request.method == "POST"
            assert b"code=fixture-code" in request.content
            return http.Response(200, json={"access_token": "fixture-access-token", "token_type": "Bearer",
                                            "id_token": state["id_token"]})
        if str(request.url) == "https://fixture-idp.invalid/jwks":
            return http.Response(200, json={"keys": [jwk]})
        pytest.fail(f"Unexpected fixture provider URL: {request.url}")

    client.client_kwargs["transport"] = http.MockTransport(provider)

    async def run(fault=None):
        session = {}
        def request(query=""):
            return Request({"type": "http", "method": "GET", "scheme": "https",
                "path": "/callback", "query_string": query.encode(), "headers": [],
                "server": ("fixture-app.invalid", 443), "session": session})
        response = await oauth.google_login(request(urlencode({"cc": "a" * 43})))
        params = parse_qs(urlparse(response.headers["location"]).query)
        claims = {"iss": "https://accounts.google.com", "sub": "fixture-user", "aud": "fixture-client",
                  "iat": int(time.time()), "exp": int(time.time()) + 600, "nonce": params["nonce"][0],
                  "email": "fixture@example.invalid", "email_verified": True}
        if fault in ("audience", "audience_with_azp"):
            claims["aud"] = "different-client"
        if fault == "audience_list_with_azp":
            claims["aud"] = ["different-client"]
        if fault == "valid_audience_list":
            # Only the configured client is trusted; an extra audience must
            # not be accepted just because the configured client is present.
            claims["aud"] = ["fixture-client"]
        if fault == "extra_audience":
            claims["aud"] = ["fixture-client", "second-client"]
        if fault in ("audience_with_azp", "audience_list_with_azp", "valid_audience_list", "extra_audience"):
            claims["azp"] = "fixture-client"
        if fault == "wrong_azp":
            claims["azp"] = "different-client"
        if fault == "missing_audience":
            claims.pop("aud")
        if fault == "malformed_audience":
            claims["aud"] = {"client": "fixture-client"}
        if fault == "issuer":
            claims["iss"] = "https://different-issuer.invalid"
        if fault == "missing_issuer":
            claims.pop("iss")
        if fault == "malformed_issuer":
            claims["iss"] = ["https://accounts.google.com"]
        if fault == "expired":
            claims["exp"] = int(time.time()) - 600
        if fault in ("missing_nonce", "missing_nonce_bypass"):
            claims.pop("nonce")
        if fault in ("mismatched_nonce", "mismatched_nonce_bypass"):
            claims["nonce"] = "different-nonce"
        if fault in ("missing_nonce_bypass", "mismatched_nonce_bypass"):
            # Authlib's documented security change: an ID-token claim must not
            # disable the nonce that this client actually saved at redirect.
            claims["nonce_supported"] = False
        # Sign raw fixture claims so malformed issuer values reach the consumer;
        # jwt.encode rejects them while constructing the fixture in newer PyJWT.
        state["id_token"] = jwt.api_jws.encode(json.dumps(claims).encode(),
            other if fault == "signature" else private, algorithm="RS256", headers={"kid": "fixture-key"})
        query = {"code": "fixture-code", "state": params["state"][0]}
        if fault == "missing_state":
            query.pop("state")
        if fault == "mismatched_state":
            query["state"] = "different-state"
        return await oauth.google_callback(request(urlencode(query)))
    return run, calls, database_calls


@pytest.mark.parametrize("fault", [None, "valid_audience_list"])
def test_actual_oidc_callback_accepts_signed_identity(oidc_flow, fault):
    run, calls, database_calls = oidc_flow
    response = asyncio.run(run(fault))
    assert response.status_code == 307
    location = response.headers["location"]
    assert location.startswith("https://fixture-app.invalid/#/auth/callback?code=")
    code = parse_qs(location.split("?", 1)[1])["code"][0]
    saved = [row for row in database_calls if isinstance(row, auth.LoginCode)]
    assert len(saved) == 1
    assert saved[0].code_hash == hashlib.sha256(code.encode()).hexdigest()
    assert saved[0].user_id == uuid.UUID(int=1) and saved[0].challenge == "a" * 43
    assert "read_user" in database_calls
    assert [method for method, _ in calls] == ["GET", "POST", "GET"]


@pytest.mark.parametrize("fault", [
    "signature", "audience", "audience_with_azp", "audience_list_with_azp", "extra_audience", "wrong_azp",
    "missing_audience", "malformed_audience", "issuer", "missing_issuer", "malformed_issuer",
    "expired", "missing_nonce", "mismatched_nonce", "missing_nonce_bypass", "mismatched_nonce_bypass",
    "missing_state", "mismatched_state",
])
def test_actual_oidc_callback_rejects_invalid_identity(oidc_flow, fault):
    # Assert the real callback decision, not only Authlib's exception class:
    # a claims_options fix must be wired here and retain issuer validation.
    run, calls, database_calls = oidc_flow
    response = asyncio.run(run(fault))
    assert response.status_code == 307
    assert response.headers["location"] == "https://fixture-app.invalid/#/auth/callback?error=oauth"
    assert database_calls == []
    if fault in ("missing_state", "mismatched_state"):
        assert [method for method, _ in calls] == ["GET"]  # no token exchange without state
