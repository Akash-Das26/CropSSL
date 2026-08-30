"""
CropSSL Authentication Module.

Provides JWT-based authentication for the API and frontend.
Default credentials for development are provided.

Usage:
    from crop_ssl.backend.auth import authenticate_user, create_token
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional


# ============================================================
# Configuration
# ============================================================
TOKEN_EXPIRY = 86400  # 24 hours
JWT_SECRET = os.environ.get("CROPSSL_SECRET", "cropssl-dev-secret-key-change-in-production")
USERS_FILE = Path(__file__).parent.parent / ".users.json"


# ============================================================
# Default Credentials
# ============================================================
DEFAULT_USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "display_name": "Administrator",
        "role": "admin",
        "created_at": "2026-01-01",
    },
    "researcher": {
        "password_hash": hashlib.sha256("research2026".encode()).hexdigest(),
        "display_name": "Researcher",
        "role": "researcher",
        "created_at": "2026-01-01",
    },
    "demo": {
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
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
    return DEFAULT_USERS.copy()


def _save_users(users: Dict):
    """Save users to file."""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def init_users():
    """Initialize default users if no users file exists."""
    if not USERS_FILE.exists():
        _save_users(DEFAULT_USERS)
        print("✅ Default users created:")
        print("   admin / admin123 (Admin)")
        print("   researcher / research2026 (Researcher)")
        print("   demo / demo123 (Viewer)")


def get_user(username: str) -> Optional[Dict]:
    """Get user by username."""
    users = _load_users()
    return users.get(username)


def list_users() -> list:
    """List all usernames."""
    users = _load_users()
    return [
        {"username": u, "display_name": v.get("display_name", u), "role": v.get("role", "viewer")}
        for u, v in users.items()
    ]


def create_user(username: str, password: str, display_name: str = "", role: str = "viewer") -> bool:
    """Create a new user."""
    users = _load_users()
    if username in users:
        return False
    users[username] = {
        "password_hash": hashlib.sha256(password.encode()).hexdigest(),
        "display_name": display_name or username,
        "role": role,
        "created_at": time.strftime("%Y-%m-%d"),
    }
    _save_users(users)
    return True


def change_password(username: str, old_password: str, new_password: str) -> bool:
    """Change user password."""
    users = _load_users()
    if username not in users:
        return False
    old_hash = hashlib.sha256(old_password.encode()).hexdigest()
    if users[username]["password_hash"] != old_hash:
        return False
    users[username]["password_hash"] = hashlib.sha256(new_password.encode()).hexdigest()
    _save_users(users)
    return True


# ============================================================
# Authentication
# ============================================================
def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user with username and password.

    Returns user dict if valid, None if invalid.
    """
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    if user["password_hash"] != password_hash:
        return None
    return {
        "username": username,
        "display_name": user.get("display_name", username),
        "role": user.get("role", "viewer"),
    }


def create_token(username: str, role: str = "viewer") -> str:
    """Create a simple JWT-like token."""
    payload = {
        "username": username,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_EXPIRY,
    }
    payload_json = json.dumps(payload, sort_keys=True)
    signature = hashlib.sha256((payload_json + JWT_SECRET).encode()).hexdigest()
    import base64
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode()
    return f"{payload_b64}.{signature}"


def verify_token(token: str) -> Optional[Dict]:
    """Verify a JWT-like token. Returns payload dict or None."""
    try:
        import base64
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, signature = parts
        # Add padding
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload_json = base64.urlsafe_b64decode(payload_b64).decode()
        expected_sig = hashlib.sha256((payload_json + JWT_SECRET).encode()).hexdigest()
        if signature != expected_sig:
            return None
        payload = json.loads(payload_json)
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def get_user_from_token(token: str) -> Optional[str]:
    """Extract username from token. Returns username or None."""
    payload = verify_token(token)
    if payload:
        return payload.get("username")
    return None
