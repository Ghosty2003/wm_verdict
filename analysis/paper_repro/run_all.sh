#!/usr/bin/env bash
# Rebuild Table 3, Fig. 3a/3b and Fig. 4 of the VerdictBench draft, check them against the draft, and list the inputs.
#   bash run_all.sh                 # outputs in out/, inputs.md next to this script
# Environment:
#   PY        python 3 with matplotlib (vector PDF) or Pillow (raster fallback); default python3
#   WMV_ROOT  project root holding libero/ (default: the original project location set in repro_common.py)
#   REPRO_OUT output directory (default: out/ next to this script)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
OUTDIR="${REPRO_OUT:-out}"
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/_manifest_*.json
echo "python: $PY"
"$PY" -c "import plotlib; print('plot backend:', plotlib.backend_note())"
"$PY" make_table3.py
"$PY" make_fig3.py
"$PY" make_fig4.py
set +e
"$PY" check_against_paper.py
rc=$?
set -e
"$PY" make_inputs.py
echo "check_against_paper exit code: $rc (0 = every Table 3 cell matches in the 113-level variant)"
exit $rc
