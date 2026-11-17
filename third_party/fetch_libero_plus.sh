#!/bin/bash
# Download the pinned LIBERO-plus corruption source (B family) and verify its sha256. Not redistributed in this repo.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)/LIBERO-plus"; mkdir -p "$D"
curl -sSfL -o "$D/env_wrapper.py" \
  https://raw.githubusercontent.com/sylvestf/LIBERO-plus/4976dc30028e805ff8094b55501d532c48fec182/libero/libero/envs/env_wrapper.py
echo "e91d7b7b35cc3ad2b073606c99860def6b3ac43b66eba00ef0ddb6bfd8f39c3c  $D/env_wrapper.py" | sha256sum -c -
echo "export LIBERO_PLUS_ENV_WRAPPER=$D/env_wrapper.py"
