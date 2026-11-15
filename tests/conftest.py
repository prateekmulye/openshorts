import os
import sys

# Make the repo root importable so tests can import the app modules directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests always run the app in self-host (BYOK) mode. app.py freezes
# BILLING_ENABLED at import time and load_dotenv never overrides an existing
# variable, so this must be set here — before any test module imports app — or
# the suite's behavior would depend on the developer's personal .env.
os.environ["BILLING_ENABLED"] = "0"

# The shared YouTube-account state (yt_account) lives in files: give the suite
# its own directory and no pacing delay, so a test never waits on, or leaves a
# cooldown for, another test or the developer's own output/ directory.
import tempfile  # noqa: E402

os.environ["YT_ACCOUNT_STATE_DIR"] = tempfile.mkdtemp(prefix="yt-account-tests-")
os.environ.setdefault("YT_ACCOUNT_MIN_INTERVAL", "0")


# Opt-in only: these tests already fake their video/HTTP probes, so external
# DNS must be a fixture too. Keep the production SSRF guard and every other
# hostname's resolution unchanged; no result or guard verdict is mocked.
import pytest  # noqa: E402


@pytest.fixture
def public_youtube_dns(monkeypatch):
    import socket
    resolve = socket.getaddrinfo

    def fixture_dns(host, port, *args, **kwargs):
        if host == 'www.youtube.com':
            # Same public-address fixture as test_ssrf_guard, not a live DNS claim.
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP,
                     '', ('142.250.184.14', port or 0))]
        return resolve(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, 'getaddrinfo', fixture_dns)
