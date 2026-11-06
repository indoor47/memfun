#!/usr/bin/env python3
"""Plan (default, read-only) or rent a RunPod GPU pod.  See docs/runpod-launch.md.

``launch_pod.py [launch flags]`` plans/launches; ``launch_pod.py <subcommand> ...`` runs any
other kit subcommand (spend, watch, tunnel, watchdog, gpus, status).
"""

from __future__ import annotations

import sys

from runpod_kit import install_exit_signals, main

SUBCOMMANDS = {"launch", "spend", "watch", "tunnel", "watchdog", "gpus", "status"}

if __name__ == "__main__":
    install_exit_signals()
    argv = sys.argv[1:]
    if not argv or argv[0] not in SUBCOMMANDS:
        argv = ["launch", *argv]
    sys.exit(main(argv))
