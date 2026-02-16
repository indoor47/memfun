#!/usr/bin/env bash
# Memfun — one-line install
# Usage: curl -fsSL https://raw.githubusercontent.com/indoor47/memfun/main/install.sh | bash
set -euo pipefail

REPO="https://github.com/indoor47/memfun.git"
INSTALL_DIR="${MEMFUN_HOME:-$HOME/.memfun-install}"
BIN_DIR="/usr/local/bin"

echo ""
echo "  ███╗   ███╗███████╗███╗   ███╗███████╗██╗   ██╗███╗   ██╗"
echo "  ████╗ ████║██╔════╝████╗ ████║██╔════╝██║   ██║████╗  ██║"
echo "  ██╔████╔██║█████╗  ██╔████╔██║█████╗  ██║   ██║██╔██╗ ██║"
echo "  ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══╝  ██║   ██║██║╚██╗██║"
echo "  ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║██║     ╚██████╔╝██║ ╚████║"
echo "  ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═══╝"
echo ""

# ── 1. Install uv if missing ──────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ── 2. Clone or update repo ───────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "Updating memfun..."
    git -C "$INSTALL_DIR" pull --ff-only origin main 2>/dev/null || true
else
    echo "Cloning memfun..."
    git clone --depth 1 "$REPO" "$INSTALL_DIR"
fi

# ── 3. Install dependencies ───────────────────────────────
echo "Installing dependencies..."
cd "$INSTALL_DIR"
uv sync --quiet

# ── 4. Create global 'memfun' command ─────────────────────
WRAPPER="$BIN_DIR/memfun"
if [ -w "$BIN_DIR" ] || [ "$(id -u)" = "0" ]; then
    cat > "$WRAPPER" << WRAP
#!/usr/bin/env bash
exec "$INSTALL_DIR/.venv/bin/memfun" "\$@"
WRAP
    chmod +x "$WRAPPER"
else
    sudo tee "$WRAPPER" > /dev/null << WRAP
#!/usr/bin/env bash
exec "$INSTALL_DIR/.venv/bin/memfun" "\$@"
WRAP
    sudo chmod +x "$WRAPPER"
fi

echo ""
echo "Installed! Run:"
echo ""
echo "  memfun init    # setup wizard (API key, backend, etc.)"
echo "  memfun         # start chatting"
echo ""
