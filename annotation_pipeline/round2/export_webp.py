"""把 manifest_real.json 里的卡片 PNG 转成 WebP（质量 85）放进网页仓库 cards/r<round>/（用户 2026-11-12 选定：网页用 WebP，NAS 保留 PNG 原图）。
已存在且比 PNG 新的 webp 跳过；可重复运行（补卡后再跑一次即可追加）。
  python export_webp.py <round_id> [<shard> <nshard>]
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json
import os
import sys
import zlib

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = _wmvp.P("$WMV_WEB_ROOT")


def main():
    rid = int(sys.argv[1])
    shard, nsh = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    out = f"{WEB}/cards/r{rid}"
    os.makedirs(out, exist_ok=True)
    n = skip = 0
    for m in json.load(open(f"{HERE}/manifest_real.json")):
        if zlib.crc32(m["unit_id"].encode()) % nsh != shard:
            continue
        dst = f"{out}/{m['unit_id']}.webp"
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(m["png"]):
            skip += 1; continue
        tmp = dst + ".tmp"
        Image.open(m["png"]).convert("RGB").save(tmp, "WEBP", quality=85, method=6)
        os.replace(tmp, dst); n += 1
    print("WEBP_DONE shard", shard, "written", n, "skipped", skip)


if __name__ == "__main__":
    main()
