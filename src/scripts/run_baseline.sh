#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 ]]; then echo "usage: $0 /score-env/bin/python config.yaml run-dir manifest.jsonl [--annotations labels.jsonl]" >&2; exit 2; fi
interpreter="$1"; config="$2"; run="$3"; manifest="$4"; shift 4
if [[ ! -x "$interpreter" ]]; then echo "configured interpreter is not executable: $interpreter" >&2; exit 2; fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" "$interpreter" -m eventvad_repro.cli run-baseline --config "$config" --run "$run" --manifest "$manifest" "$@"
