#!/usr/bin/env bash
# CropSSL one-line installer.
#
# Performs exactly the manual install steps from README.md
# ("Installation & Quick Start → Install"):
#   git clone → python3 -m venv venv → pip install -r requirements.txt
#   → pip install -e .
# and finishes with an import smoke test.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/officialarghya29/CropSSL/main/install.sh | bash
#   bash install.sh [TARGET_DIR]     # fresh install (or re-run) at TARGET_DIR
#   bash install.sh                  # inside an existing checkout: install in place
#
# Re-running is safe: an existing clone is fast-forwarded, the venv is reused.
set -euo pipefail

REPO_URL="https://github.com/officialarghya29/CropSSL.git"
TARGET_DIR="${1:-CropSSL}"

echo "==> CropSSL installer"

# 1. Locate or create the source tree
if [ "$#" -eq 0 ] && [ -f pyproject.toml ] && [ -d crop_ssl ]; then
    echo "==> Existing checkout found in $PWD — installing in place"
    TARGET_DIR="."
elif [ -d "$TARGET_DIR/.git" ]; then
    echo "==> Existing clone found at $TARGET_DIR — updating (ff-only)"
    git -C "$TARGET_DIR" pull --ff-only
else
    echo "==> Cloning $REPO_URL into $TARGET_DIR"
    git clone "$REPO_URL" "$TARGET_DIR"
fi
cd "$TARGET_DIR"

# 2. Virtualenv (mirrors: python3 -m venv venv && source venv/bin/activate)
if [ ! -x venv/bin/python ]; then
    echo "==> Creating virtualenv ./venv"
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

# 3. Dependencies + editable install (mirrors README exactly)
echo "==> Installing dependencies (several minutes on first run; torch is large)"
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Verify: package imports and torch loads
echo "==> Verifying installation"
python -c "import crop_ssl, torch; print(f'CropSSL {crop_ssl.__version__} installed OK (torch {torch.__version__})')"

echo
echo "✅ Done. Get started with:"
echo "   cd \"$PWD\" && source venv/bin/activate"
echo "   # Required for API auth (login-protected routes):"
echo "   export CROPSSL_SECRET=\"\$(python3 -c 'import secrets; print(secrets.token_hex(32))')\""
echo "   python3 -m crop_ssl.scripts.run_pipeline --epochs 1 --device cpu"
echo
echo "NOTE: the venv cannot stay activated after 'curl | bash' —"
echo "run the 'source venv/bin/activate' line above in your shell."
