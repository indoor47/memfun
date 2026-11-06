#!/usr/bin/env python3
"""SSH-forward 127.0.0.1:<local> (default 18000, never 8434) to the pod's SGLang port.

Usage: python3 scripts/runpod/tunnel.py --pod-id <id> [--local-port 18000] [--run]
"""

from __future__ import annotations

import sys

from runpod_kit import install_exit_signals, main

if __name__ == "__main__":
    install_exit_signals()
    sys.exit(main(["tunnel", *sys.argv[1:]]))
