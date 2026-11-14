"""Opt-in PostgreSQL/session acceptance. Only provider HTTP is faked.

Use a dedicated local PostgreSQL database named openshorts_auth_test_<suffix>,
owned by auth_test:

  PYTHON_DOTENV_DISABLED=1 OPENSHORTS_AUTH_TEST_DATABASE_URL=<local test URL> \
      python -m pytest tests/test_postgres_oauth_acceptance.py -q -p no:cacheprovider

The URL must use postgresql+asyncpg and a literal loopback IP. No database,
table, or evidence row is dropped/deleted: unique fixture identities remain for
inspection. Run without xdist against a dedicated test database. This mounts
the actual auth routers with production's SessionMiddleware configuration; it
does not start the unrelated billing/sweeper services or exercise a real browser.
"""
import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from urllib.parse import parse_qs, urlparse

import pytest


def _local_test_database_url(value):
    parsed = urlparse(value)
    if not (
        parsed.scheme == "postgresql+asyncpg"
        and parsed.hostname in ("127.0.0.1", "::1")
        and parsed.username == "auth_test"
        and re.fullmatch(r"/openshorts_auth_test_[a-z0-9_]+", parsed.path)
        and not parsed.query and not parsed.fragment
    ):
        raise ValueError("Use a dedicated auth_test PostgreSQL database on a literal loopback IP")
    return value


@pytest.mark.parametrize("url", [
    "postgresql+asyncpg://auth_test:synthetic@database.invalid/openshorts_auth_test_one",
    "postgresql+asyncpg://auth_test:synthetic@localhost/openshorts_auth_test_one",
    "postgresql+asyncpg://auth_test:synthetic@127.0.0.1/production",
    "postgresql+asyncpg://postgres:synthetic@127.0.0.1/openshorts_auth_test_one",
    "sqlite:///openshorts_auth_test_one",
    "postgresql+asyncpg://auth_test:synthetic@127.0.0.1/openshorts_auth_test_one?host=other",
    "postgresql+asyncpg://auth_test:synthetic@127.0.0.1/openshorts_auth_test_one#other",
])
def test_rejects_nonlocal_or_nontest_database_before_connecting(url):
    with pytest.raises(ValueError, match="dedicated auth_test"):
        _local_test_database_url(url)


@pytest.fixture
def real_auth_flow(monkeypatch):
    raw_url = os.environ.get("OPENSHORTS_AUTH_TEST_DATABASE_URL")
    if not raw_url:
        pytest.skip("Opt-in local PostgreSQL URL missing; no persistence acceptance claim")
    url = _local_test_database_url(raw_url)  # before application imports or writes
    pytest.importorskip("asyncpg")
    pytest.importorskip("authlib")
    pytest.importorskip("boto3")  # actual signup denial lookup imports cloud.account
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "fixture-client")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "synthetic-client-secret")
    monkeypatch.setenv("JWT_SECRET", "synthetic-session-secret-never-used-for-real-identities")
    monkeypatch.setenv("FRONTEND_URL", "https://fixture-app.invalid")
    monkeypatch.setenv("OAUTH_REDIRECT_URI", "https://fixture-app.invalid/api/auth/google/callback")
    monkeypatch.setenv("LEGACY_TOKEN_REDIRECT_UNTIL", "2000-01-01T00:00:00+00:00")

    import httpx
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi import FastAPI
    from sqlalchemy import func, select, text
    from starlette.middleware.sessions import SessionMiddleware
    from authlib.integrations.httpx_client import oauth2_client
    from cloud import auth, database, evidence, oauth
    from cloud.models import EvidenceEvent, LoginCode, User

    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_sessionmaker", None)
    monkeypatch.setattr(oauth, "oauth", None)
    oauth.register()
    provider_http = getattr(oauth2_client, "httpx2", None) or oauth2_client.httpx
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    wrong_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key(), as_dict=True)
    jwk.update(kid="fixture-key", use="sig", alg="RS256")
    provider_state, provider_calls = {}, []

    def provider(request):
        provider_calls.append(str(request.url))
        if str(request.url) == "https://accounts.google.com/.well-known/openid-configuration":
            return provider_http.Response(200, json={
                "issuer": "https://accounts.google.com",
                "authorization_endpoint": "https://fixture-idp.invalid/authorize",
                "token_endpoint": "https://fixture-idp.invalid/token",
                "jwks_uri": "https://fixture-idp.invalid/jwks",
                "id_token_signing_alg_values_supported": ["RS256"],
            })
        if str(request.url) == "https://fixture-idp.invalid/token":
            assert request.method == "POST" and b"code=fixture-code" in request.content
            return provider_http.Response(200, json={
                "access_token": "synthetic-provider-access", "token_type": "Bearer",
                "id_token": provider_state["token"],
            })
        if str(request.url) == "https://fixture-idp.invalid/jwks":
            return provider_http.Response(200, json={"keys": [jwk]})
        pytest.fail("Unexpected provider HTTP request")

    oauth.oauth.google.client_kwargs["transport"] = provider_http.MockTransport(provider)
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key=auth.settings.jwt_secret)
    app.include_router(oauth.router)
    app.include_router(auth.router)

    async def counts():
        async with database.session() as session:
            return tuple([await session.scalar(select(func.count()).select_from(model))
                          for model in (User, LoginCode, EvidenceEvent)])

    async def run(fault=None):
        identity = uuid.uuid4().hex
        email = f"acceptance-{identity}@example.invalid"
        verifier = "v" * 64
        await database.init_engine()  # actual PostgreSQL citext, models and additive DDL
        try:
            async with database.session() as session:
                assert await session.scalar(text(
                    "SELECT udt_name FROM information_schema.columns "
                    "WHERE table_schema='public' AND table_name='users' AND column_name='email'"
                )) == "citext"
            before = await counts()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="https://fixture-app.invalid") as browser:
                login = await browser.get("/api/auth/google", params={"cc": auth.challenge_for(verifier)})
                # Authlib's authorize_redirect uses 302; our application's
                # callback uses Starlette's default 307. Assert each boundary.
                assert login.status_code == 302
                cookie = login.headers["set-cookie"]
                assert "httponly" in cookie.lower() and "samesite=lax" in cookie.lower()
                assert browser.cookies.get("session")
                params = parse_qs(urlparse(login.headers["location"]).query)
                claims = {
                    "iss": "https://accounts.google.com", "sub": identity,
                    "aud": "fixture-client", "iat": int(time.time()),
                    "exp": int(time.time()) + 600, "nonce": params["nonce"][0],
                    "email": email, "email_verified": True,
                }
                if fault == "issuer":
                    claims["iss"] = "https://other-issuer.invalid"
                if fault == "audience":
                    claims.update(aud="other-client", azp="fixture-client")
                if fault == "extra_audience":
                    claims.update(aud=["fixture-client", "other-client"], azp="fixture-client")
                if fault == "nonce":
                    claims["nonce"] = "wrong-nonce"
                if fault == "missing_nonce":
                    claims.pop("nonce")
                if fault == "expired":
                    claims["exp"] = int(time.time()) - 600
                provider_state["token"] = jwt.api_jws.encode(
                    json.dumps(claims).encode(), wrong_private if fault == "signature" else private,
                    algorithm="RS256", headers={"kid": "fixture-key"})
                query = {"code": "fixture-code", "state": params["state"][0]}
                if fault == "state":
                    query["state"] = "wrong-state"
                if fault == "missing_state":
                    query.pop("state")
                if fault in ("missing_cookie", "tampered_cookie"):
                    old_cookie = browser.cookies.get("session")
                    browser.cookies.clear()
                    if fault == "tampered_cookie":
                        browser.cookies.set("session", old_cookie + "tampered")
                callback = await browser.get("/api/auth/google/callback", params=query)
                assert callback.status_code == 307
                if evidence._pending:
                    await asyncio.gather(*list(evidence._pending))
                if fault:
                    assert callback.headers["location"] == "https://fixture-app.invalid/#/auth/callback?error=oauth"
                    assert await counts() == before
                    assert (await browser.get("/api/me")).status_code == 401
                    if fault in ("state", "missing_state", "missing_cookie", "tampered_cookie"):
                        assert provider_calls == ["https://accounts.google.com/.well-known/openid-configuration"]
                    return

                code = parse_qs(callback.headers["location"].split("?", 1)[1])["code"][0]
                code_hash = hashlib.sha256(code.encode()).hexdigest()
                async with database.session() as session:
                    user = (await session.execute(select(User).where(User.email == email.upper()))).scalar_one()
                    row = await session.get(LoginCode, code_hash)
                    assert user.email == email and user.google_sub == identity
                    assert row.user_id == user.id and row.challenge == auth.challenge_for(verifier)
                    assert row.used_at is None
                    user_id = user.id
                    event = (await session.execute(select(EvidenceEvent).where(
                        EvidenceEvent.user_id == user_id))).scalar_one()
                    assert event.kind == "login" and event.method == "google"
                assert await counts() == tuple(n + 1 for n in before)
                wrong = await browser.post("/api/auth/exchange", json={"code": code, "verifier": "x" * 64})
                assert wrong.status_code == 400 and "token" not in wrong.json()
                async with database.session() as session:
                    assert (await session.get(LoginCode, code_hash)).used_at is None

                # Actual concurrent requests exercise PostgreSQL's row lock, not
                # a fake session that serializes the exchange on our behalf.
                attempts = await asyncio.gather(*(
                    browser.post("/api/auth/exchange", json={"code": code, "verifier": verifier})
                    for _ in range(2)))
                assert sorted(response.status_code for response in attempts) == [200, 400]
                signed_in = next(response.json() for response in attempts if response.status_code == 200)
                assert signed_in["user"]["id"] == str(user_id)
                assert signed_in["user"]["email"] == email
                async with database.session() as session:
                    assert (await session.get(LoginCode, code_hash)).used_at is not None
                replay = await browser.post("/api/auth/exchange", json={"code": code, "verifier": verifier})
                assert replay.status_code == 400 and "token" not in replay.json()
                me = await browser.get("/api/me", headers={"Authorization": "Bearer " + signed_in["token"]})
                assert me.status_code == 200
                assert me.json()["user"] == {"id": str(user_id), "email": email}
                assert (await browser.get("/api/me")).status_code == 401
                broken_token = signed_in["token"][:-8] + "abcdefgh"
                assert (await browser.get("/api/me", headers={"Authorization": "Bearer " + broken_token})).status_code == 401
                assert await counts() == tuple(n + 1 for n in before)
        finally:
            if evidence._pending:
                await asyncio.gather(*list(evidence._pending))
            await database._engine.dispose()
    return run


def test_real_postgres_callback_exchange_and_session_identity(real_auth_flow):
    asyncio.run(real_auth_flow())


@pytest.mark.parametrize("fault", [
    "issuer", "audience", "extra_audience", "nonce", "missing_nonce", "expired", "signature",
    "state", "missing_state", "missing_cookie", "tampered_cookie",
])
def test_rejected_callback_has_no_persistent_auth_effects(real_auth_flow, fault):
    asyncio.run(real_auth_flow(fault))
