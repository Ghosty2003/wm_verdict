#!/bin/bash
# two throwaway venvs for the B-family function check (production envs untouched):
#  pinned  = LIBERO-plus commit 4976dc3 requirements: numpy==1.22.4 (+ scikit-image, unpinned upstream -> 0.21.0, the last
#            release supporting numpy 1.22), python 3.10
#  current = numpy 2.2.6 as in our lerobot_libero / openwam envs, latest scikit-image
set -e
W=/mnt/qnap/chenmg93/wm_verdict/libero/out/b_family/healthcheck
PY=~/miniforge3/envs/openwam/bin/python      # python 3.10
V=~/bcheck_venvs; mkdir -p $V   # node-local on gix-30292620 (the SMB mount refuses venv symlinks)
$PY -m venv $V/venv_pinned; $V/venv_pinned/bin/pip -q install "numpy==1.22.4" "scipy==1.10.1" "scikit-image==0.21.0" "pillow<11" "matplotlib==3.7.5"
$PY -m venv $V/venv_current; $V/venv_current/bin/pip -q install "numpy==2.2.6" scipy scikit-image pillow
for v in pinned current; do $V/venv_$v/bin/python -c "import numpy, scipy, skimage; print('$v', numpy.__version__, scipy.__version__, skimage.__version__)"; done
