#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 /persistent/asset/root" >&2; exit 2; fi
asset_root="$1"; mkdir -p "$asset_root"
echo "Download datasets and model weights into: $asset_root"
echo "This script deliberately does not fetch gated or license-controlled assets without explicit URLs/credentials."
