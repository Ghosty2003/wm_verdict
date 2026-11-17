#!/bin/bash
# OpenWAM-alpha precheck, step 1: environment + assets on $NODE_D (2026-10-23). Repo pinned at 1e103e1 (screened commit).
set -euo pipefail
REPO=$WMV_ROOT/libero/openwam/third_party/OpenWAM
source $CONDA_ROOT/etc/profile.d/conda.sh
if ! conda env list | grep -q "^openwam "; then conda create -y -n openwam python=3.10; fi
conda activate openwam
python -c "import torch" 2>/dev/null || pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
cd $REPO
pip install -e . 2>&1 | tail -5
python -c "import torch, openwam; print('torch', torch.__version__, torch.version.cuda, 'openwam ok')"
python scripts/download_assets/download_video_backbone.py --name Wan2.2-TI2V-5B --source huggingface --yes
python scripts/download_assets/download_openwam_checkpoints.py --family alpha --name OpenWAM-Alpha-Sim-LIBERO --yes
du -sh assets/* 2>/dev/null
echo SETUP_DONE
