# third_party/

Nothing is vendored here. Clone / download at the pinned versions (see the table in the top-level README):

- `lerobot/`  — `git clone https://github.com/huggingface/lerobot && git -C lerobot checkout e624f3f7f8411ec3a02635d06e79373341e5ef35`
- `OpenWAM/`  — `git clone https://github.com/OpenWAM-Official/OpenWAM && git -C OpenWAM checkout 1e103e1c61905c230dab6b0f99980370cf2f9e3c`,
  then apply `../patches/openwam-1e103e1_wan22_model_path.patch`
- `LIBERO-plus/env_wrapper.py` — `bash fetch_libero_plus.sh` (commit 4976dc30, sha256 checked)
- WorldVLA (appendix only) — `git clone -b worldvla https://github.com/alibaba-damo-academy/WorldVLA && git -C WorldVLA checkout ebdc36dc5cfd`
