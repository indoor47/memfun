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

# ── 0. Check system prerequisites ────────────────────────
for cmd in git curl; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: '$cmd' is required but not installed."
        echo "Install it with your package manager (e.g., apt install $cmd)"
        exit 1
    fi
done

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
uv sync --all-packages --quiet

# Verify the entrypoint was created
if [ ! -f "$INSTALL_DIR/.venv/bin/memfun" ]; then
    echo "Error: 'memfun' binary was not created in the virtual environment."
    echo "Try running: cd $INSTALL_DIR && uv sync --all-packages"
    exit 1
fi

# ── 4. Create global 'memfun' command ─────────────────────
WRAPPER="$BIN_DIR/memfun"
if [ -w "$BIN_DIR" ] || [ "$(id -u)" = "0" ]; then
    cat > "$WRAPPER" << WRAP
#!/usr/bin/env bash
exec "$INSTALL_DIR/.venv/bin/memfun" "\$@"
WRAP
    chmod +x "$WRAPPER"
else
    # Try sudo; fall back to ~/.local/bin if sudo fails
    if sudo -n true 2>/dev/null; then
        sudo tee "$WRAPPER" > /dev/null << WRAP
#!/usr/bin/env bash
exec "$INSTALL_DIR/.venv/bin/memfun" "\$@"
WRAP
        sudo chmod +x "$WRAPPER"
    else
        BIN_DIR="$HOME/.local/bin"
        mkdir -p "$BIN_DIR"
        WRAPPER="$BIN_DIR/memfun"
        cat > "$WRAPPER" << WRAP
#!/usr/bin/env bash
exec "$INSTALL_DIR/.venv/bin/memfun" "\$@"
WRAP
        chmod +x "$WRAPPER"
        echo "Note: Installed to $BIN_DIR/memfun (add to PATH if needed)"
    fi
fi

echo ""
echo "Installed! Run:"
echo ""
echo "  memfun init    # setup wizard (API key, backend, etc.)"
echo "  memfun         # start chatting"
echo ""
