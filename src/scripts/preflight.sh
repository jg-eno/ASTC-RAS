#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then echo "usage: $0 /env/bin/python config.yaml" >&2; exit 2; fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" "$1" -m eventvad_repro.cli preflight --config "$root/$2"
