"""标注网页真实帧卡的成败一致性核对（2026-11-15 用户：“确认所有数据标注没有回放与记录不一致的问题，写脚本判断，不要读图看”）。
只读，不看图片内容，只比对数字：
  A. 回放日志全表：runs/**/replay_frames*.jsonl、runs/lingbot_t*/rerun_closed/*.jsonl 里每个单元的 matches_recorded_success；
  B. 同卷页（wm_verdict_web/data/samecase_pages.json）上每张 REAL 卡、以及 round 2 入库清单（round2_real/manifest_real.json）的每张卡：
     我们两个模型 → 直接读该卡所用帧文件（npz）的逐步 success 数组，取前 400 步得到“画面里的成败 / 首次成功步”，
     与主表记录（inventory.jsonl 的 success / first_success_step，按 H = 400 截断）比对；
     Cosmos / UVA → 真实视频与成败出自同一次执行的同一个结果文件，标为“同源”，不做比对。
输出：real_consistency_report.json；有不一致时退出码 1，并列出单元。
  python check_real_consistency.py      （需要 numpy：在节点上用 lerobot_libero 跑）"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
RUNS = f"{ROOT}/libero/out/runs"
H = 400


def log_mismatches():
    last = {}
    pats = glob.glob(f"{RUNS}/*/replay_frames*.jsonl") + glob.glob(f"{RUNS}/*/*/replay_frames*.jsonl") + glob.glob(f"{RUNS}/lingbot_t*/rerun_closed/*.jsonl")
    for f in pats:
        for l in open(f):
            if l.strip():
                r = json.loads(l); last[(os.path.dirname(f), r["unit_id"])] = r
    bad = sorted({r["unit_id"] for r in last.values() if r.get("matches_recorded_success") is False})
    return len(last), bad


def frames_outcome(path):
    z = np.load(path)
    s = np.asarray(z["success"]).astype(bool).reshape(-1)[:H]
    return bool(s.any()), (int(np.argmax(s)) + 1 if s.any() else None)


def main():
    n_logged, log_bad = log_mismatches()
    inv = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{HERE}/inventory.jsonl")}
    pages = json.load(open(_wmvp.P("$WMV_WEB_ROOT/data/samecase_pages.json")))["pages"]
    cards = {}
    for p in pages:
        for r in p["rows"]:
            if r.get("real"):
                cards[r["real"]] = "samecase"
    for m in json.load(open(f"{ROOT}/libero/out/annotation/round2_real/manifest_real.json")):
        cards.setdefault(m["unit_id"], "round2")
    out, bad = [], []
    for uid, where in sorted(cards.items()):
        u = inv.get(uid)
        if u is None:
            out.append(dict(unit_id=uid, where=where, status="not_in_inventory")); continue
        if u["model"] in ("cosmos", "uva"):
            out.append(dict(unit_id=uid, where=where, status="same_source")); continue
        path = u["real_path"]
        if u["real"] == "replay":
            hits = glob.glob(f"{RUNS}/*/replay_frames/{uid}.npz") + glob.glob(f"{RUNS}/*/*/replay_frames/{uid}.npz")
            path = hits[0] if hits else None
        if not path or not path.endswith(".npz") or not os.path.exists(path):
            out.append(dict(unit_id=uid, where=where, status="no_frames_file")); continue
        ok_f, fs_f = frames_outcome(path)
        fs_r = u.get("first_success_step")
        ok_r = bool(u["success"]) and (fs_r is None or fs_r <= H)
        st = "ok" if ok_f == ok_r else "MISMATCH"
        row = dict(unit_id=uid, where=where, status=st, frames_success=ok_f, frames_first_step=fs_f, recorded_success=ok_r, recorded_first_step=fs_r,
                   step_equal=(fs_f == fs_r) if ok_f and ok_r else None, frames_file=path)
        out.append(row)
        if st == "MISMATCH":
            bad.append(row)
    from collections import Counter
    summ = dict(replay_logs_units=n_logged, replay_logs_mismatch=log_bad, cards_checked=len(out),
                status={f"{k[0]}|{k[1]}": v for k, v in Counter((o["where"], o["status"]) for o in out).items()}, mismatched_cards=bad,
                log_mismatch_in_cards=sorted(set(log_bad) & set(cards)),
                step_differs_same_outcome=sum(1 for o in out if o.get("step_equal") is False))
    json.dump(dict(summary=summ, rows=out), open(f"{HERE}/real_consistency_report.json", "w"), ensure_ascii=False, indent=0)
    print(json.dumps(summ, ensure_ascii=False, indent=1)[:3000])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
