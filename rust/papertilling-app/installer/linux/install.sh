#!/usr/bin/env sh
# User-local installer for Escher Desktop Explorer on Linux (no root required).
# Usage: ./install.sh          (install to ~/.local)
#        ./install.sh --remove (uninstall)
set -eu

PREFIX="${PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"
APP_DIR="$PREFIX/share/applications"
ICON_DIR="$PREFIX/share/icons/hicolor/256x256/apps"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if [ "${1:-}" = "--remove" ] || [ "${1:-}" = "--uninstall" ]; then
    rm -f "$BIN_DIR/papertilling-app" "$APP_DIR/papertilling.desktop" "$ICON_DIR/papertilling.png"
    echo "Removed Escher from $PREFIX"
    exit 0
fi

mkdir -p "$BIN_DIR" "$APP_DIR" "$ICON_DIR"
install -m 0755 "$SCRIPT_DIR/escher" "$BIN_DIR/papertilling-app"
install -m 0644 "$SCRIPT_DIR/icon.png" "$ICON_DIR/papertilling.png"
sed "s|^Exec=papertilling-app$|Exec=$BIN_DIR/papertilling-app|" \
    "$SCRIPT_DIR/escher.desktop" > "$APP_DIR/papertilling.desktop"
chmod 0644 "$APP_DIR/papertilling.desktop"

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APP_DIR" >/dev/null 2>&1 || true
fi

echo "Escher is installed. Launch it from your applications menu."
