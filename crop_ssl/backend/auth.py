"""
CropSSL Authentication Module.

Provides JWT-like token authentication for the API and frontend.
Default credentials for development are provided.

Security notes:
- Tokens are signed with CROPSSL_SECRET (env). If it is not set, token
  creation/verification FAILS CLOSED — the server refuses to issue or
  accept tokens rather than silently using a known dev secret.
- Passwords are stored salted (PBKDF2-SHA256, 200k iterations). Legacy
  unsalted-SHA256 entries are verified and transparently upgraded on the
  next successful login.
- Set CROPSSL_ALLOW_ANONYMOUS=1 ONLY for local development to bypass
  auth on protected routes when no Authorization header is sent.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Dict, Optional


# ============================================================
# Configuration
# ============================================================
TOKEN_EXPIRY = 86400  # 24 hours
PBKDF2_ITERATIONS = 200_000

# No default secret: an unset CROPSSL_SECRET disables token auth entirely
# (fail closed) instead of making every deployment forgeable.
JWT_SECRET = os.environ.get("CROPSSL_SECRET", "")

# Dev-only bypass: requests with NO Authorization header are treated as admin.
ANONYMOUS_MODE = os.environ.get("CROPSSL_ALLOW_ANONYMOUS") == "1"

# Static operator credential for machine clients (CI jobs, scripts).
# Accepts 'Authorization: ApiKey <key>' or 'X-API-Key: <key>'. Empty = disabled.
API_KEY = os.environ.get("CROPSSL_API_KEY", "")

USERS_FILE = Path(__file__).parent.parent / ".users.json"


# ============================================================
# Password hashing (salted PBKDF2; legacy sha256 readable)
# ============================================================
def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Return 'pbkdf2$<iterations>$<salt_hex>$<hash_hex>' for storage."""
    salt = secrets.token_bytes(16) if salt is None else salt
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, PBKDF2_ITERATIONS
    )
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """Verify a stored hash; supports legacy unsalted sha256 entries."""
    if stored.startswith("pbkdf2$"):
        try:
            _, iters, salt_hex, hash_hex = stored.split("$")
            digest = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), bytes.fromhex(salt_hex), int(iters)
            )
            return hmac.compare_digest(digest.hex(), hash_hex)
        except (ValueError, TypeError):
            return False
    # Legacy: unsalted sha256 (transparently upgraded on next login)
    legacy = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(legacy, stored)


def _is_legacy_hash(stored: str) -> bool:
    return not stored.startswith("pbkdf2$")


# ============================================================
# Default Credentials (documented demo accounts; salted per-install)
# ============================================================
DEFAULT_USERS = {
    "admin": {
        "password_hash": _hash_password("admin123"),
        "display_name": "Administrator",
        "role": "admin",
        "created_at": "2026-01-01",
    },
    "researcher": {
        "password_hash": _hash_password("research2026"),
        "display_name": "Researcher",
        "role": "researcher",
        "created_at": "2026-01-01",
    },
    "demo": {
        "password_hash": _hash_password("demo123"),
        "display_name": "Demo User",
        "role": "viewer",
        "created_at": "2026-01-01",
    },
}


# ============================================================
# User Management
# ============================================================
def _load_users() -> Dict:
    """Load users from file or return defaults."""
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return json.loads(json.dumps(DEFAULT_USERS))  # deep copy


def _save_users(users: Dict):
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def init_users():
    """Initialize default users if no users file exists."""
    if not USERS_FILE.exists():
        _save_users(DEFAULT_USERS)
        print(f"✅ Default users created in {USERS_FILE}")
        print("   admin / admin123 (Admin) — CHANGE THESE in production:")
        print("   delete the file or call change_password(), then restart.")


def get_user(username: str) -> Optional[Dict]:
    users = _load_users()
    return users.get(username)


def list_users() -> list:
    """List all usernames (display name + role only, never hashes)."""
    users = _load_users()
    return [
        {"username": u, "display_name": v.get("display_name", u), "role": v.get("role", "viewer")}
        for u, v in users.items()
    ]


def create_user(username: str, password: str, display_name: str = "", role: str = "viewer") -> bool:
    """Create a new user with a salted PBKDF2 password hash."""
    users = _load_users()
    if username in users:
        return False
    users[username] = {
        "password_hash": _hash_password(password),
        "display_name": display_name or username,
        "role": role,
        "created_at": time.strftime("%Y-%m-%d"),
    }
    _save_users(users)
    return True


def change_password(username: str, old_password: str, new_password: str) -> bool:
    """Change a user's password (re-hashed with a fresh random salt)."""
    users = _load_users()
    if username not in users:
        return False
    if not _verify_password(old_password, users[username]["password_hash"]):
        return False
    users[username]["password_hash"] = _hash_password(new_password)
    _save_users(users)
    return True


# ============================================================
# Authentication
# ============================================================
def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user; upgrades legacy unsalted hashes on success."""
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    stored = user["password_hash"]
    if not _verify_password(password, stored):
        return None
    if _is_legacy_hash(stored):
        users[username]["password_hash"] = _hash_password(password)
        _save_users(users)
    return {
        "username": username,
        "display_name": user.get("display_name", username),
        "role": user.get("role", "viewer"),
    }


def verify_api_key(key: str) -> bool:
    """Constant-time check of a caller-supplied API key. Empty stored key = disabled."""
    if not API_KEY or not key:
        return False
    return hmac.compare_digest(key, API_KEY)


def _require_secret() -> str:
    if not JWT_SECRET:
        raise RuntimeError(
            "CROPSSL_SECRET is not set — refusing to issue/verify tokens. "
            "Start the server with: CROPSSL_SECRET=<random-secret> python -m crop_ssl.backend.api "
            "(dev only: CROPSSL_ALLOW_ANONYMOUS=1 bypasses auth entirely)."
        )
    return JWT_SECRET


def create_token(username: str, role: str = "viewer") -> str:
    """Create a signed token: base64(payload).sha256(payload+secret)."""
    secret = _require_secret()
    payload = {
        "username": username,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_EXPIRY,
    }
    payload_json = json.dumps(payload, sort_keys=True)
    signature = hashlib.sha256((payload_json + secret).encode()).hexdigest()
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode()
    return f"{payload_b64}.{signature}"


def verify_token(token: str) -> Optional[Dict]:
    """Verify a token's signature and expiry. Returns payload or None."""
    try:
        secret = _require_secret()
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, signature = parts
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload_json = base64.urlsafe_b64decode(payload_b64).decode()
        expected_sig = hashlib.sha256((payload_json + secret).encode()).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None
        payload = json.loads(payload_json)
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except RuntimeError:
        raise  # missing secret must surface, not masquerade as a bad token
    except Exception:
        return None


def get_user_from_token(token: str) -> Optional[str]:
    """Extract username from token. Returns username or None."""
    payload = verify_token(token)
    if payload:
        return payload.get("username")
    return None
