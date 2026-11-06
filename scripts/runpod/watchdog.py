#!/usr/bin/env python3
"""Local watchdog: terminate a pod at a deadline, on Ctrl-C, or on error.

Usage: python3 scripts/runpod/watchdog.py --pod-id <id> --max-minutes 120
"""

from __future__ import annotations

import sys

from runpod_kit import install_exit_signals, main

if __name__ == "__main__":
    install_exit_signals()
    sys.exit(main(["watchdog", *sys.argv[1:]]))
