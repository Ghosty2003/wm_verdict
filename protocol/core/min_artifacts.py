"""每单元存档最低要求（protocol.yaml → min_artifacts，用户规矩 2026-11-13，只管今后）。三处共用：
  runner      assert_unit(npz_path, row, kind)：单元 npz 写盘后、records 行写入前调用；缺任何一项抛 MinArtifactsError，
              作业立刻停，不写这一行。
  verify_run  check 10 调 verify_rows()：抽样重开已完成单元的 npz，核对键与长度。
  命令行      python min_artifacts.py verify --name <名> --records '<glob>' [--sample 5] [--kind closed|open]
              → out/artifact_verify/<名>.json（逐格 模型×任务×档：通过 / 不通过 + 问题清单），make_snapshot 据此判"完成"。
规格只在 protocol.yaml 里维护一份；本模块读它，不另抄。
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob
import json
import os
import random
import sys
import zipfile

ROOT = _wmvp.P("$WMV_ROOT")
PROTOCOL = f"{ROOT}/protocol.yaml"


class MinArtifactsError(RuntimeError):
    pass


def spec():
    import yaml
    return yaml.safe_load(open(PROTOCOL))["min_artifacts"]


def _get(row, dotted):
    cur = row
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return False, None
        cur = cur[p]
    return True, cur


def check_row(row, S):
    probs = []
    for item in S["record_fields"]:
        alts = item.split("|")
        found = [a for a in alts if _get(row, a)[0]]
        if not found:
            probs.append(f"record field missing: {item}")
        elif item.split("|")[0] in ("machine", "gpu") and not _get(row, found[0])[1]:
            probs.append(f"record field empty: {item}")
    return probs


def _keys(npz_path):
    with zipfile.ZipFile(npz_path) as z:              # 只读目录即可知道键；数组内容另行打开
        return {i.filename[:-4] for i in z.infolist()}


def check_npz(npz_path, row, kind, S):
    import numpy as np
    probs = []
    try:
        keys = _keys(npz_path)
        z = np.load(npz_path)
    except Exception as e:  # noqa: BLE001
        return [f"npz unreadable: {type(e).__name__}: {e}"]
    req = S["closed_loop"] if kind == "closed" else S["open_loop"]
    passes = [""] + (["run2_"] if any(k.startswith("run2_") for k in keys) or (kind == "closed" and "rerun" in (row.get("closed_loop") or {})) else [])
    for p in passes:
        for item, alts in req.items():
            if item == "checks":
                continue
            if not any(f"{p}{a}" in keys for a in alts):
                probs.append(f"{p or 'pass1 '}missing {item} (any of {alts})")
        if kind != "closed" or probs:
            continue
        try:
            c, e, s = z[f"{p}commanded"], z[f"{p}executed"], np.asarray(z[f"{p}success"]).astype(bool).ravel()
        except Exception as ex:  # noqa: BLE001
            probs.append(f"{p or 'pass1 '}array unreadable: {ex}"); continue
        cl = row.get("closed_loop") or {}
        ref = cl if p == "" else (cl.get("rerun") or {})
        if not (len(c) == len(e) == len(s)):
            probs.append(f"{p or 'pass1 '}length mismatch commanded {len(c)} executed {len(e)} success {len(s)}")
        if ref.get("steps") is not None and len(e) != int(ref["steps"]):
            probs.append(f"{p or 'pass1 '}executed length {len(e)} != recorded steps {ref['steps']}")
        if "success" in ref and bool(s.any()) != bool(ref["success"]):
            probs.append(f"{p or 'pass1 '}any(success) {bool(s.any())} != recorded success {ref['success']}")
        if ref.get("first_success_step") is not None and s.any() and int(np.argmax(s)) + 1 != int(ref["first_success_step"]):
            probs.append(f"{p or 'pass1 '}first success index {int(np.argmax(s)) + 1} != recorded {ref['first_success_step']}")
    return probs


def npz_of(row):
    rel = row.get("closed_loop_file") or row.get("trajectory_file")
    return None if not rel else (rel if rel.startswith("/") else f"{ROOT}/{rel}")


def assert_unit(npz_path, row, kind="closed"):
    """runner 调用：通过返回 None，不通过抛 MinArtifactsError（不要捕获它）。"""
    S = spec()
    probs = check_row(row, S) + (check_npz(npz_path, row, kind, S) if os.path.exists(npz_path) else [f"npz not on disk: {npz_path}"])
    if probs:
        raise MinArtifactsError(f"MIN_ARTIFACTS v{S['version']} FAIL {row.get('unit_id')}: " + "; ".join(probs))


def verify_rows(rows, sample=5, seed=20261113, kind="closed"):
    """按 (task, setting) 分格，每格按固定种子抽 sample 个已完成单元（sample<=0 = 全查）。返回 {cell: dict(status, checked, problems)}。"""
    S = spec()
    cells = {}
    for r in rows:
        cells.setdefault((r.get("paper_task") or r.get("task"), r.get("setting_id")), []).append(r)
    out = {}
    for key, rs in sorted(cells.items(), key=lambda x: str(x[0])):
        rs = sorted(rs, key=lambda r: r["unit_id"])
        pick = rs if sample <= 0 else random.Random(f"{seed}/{key}").sample(rs, min(sample, len(rs)))
        probs = []
        for r in pick:
            p = npz_of(r)
            pr = check_row(r, S) + (check_npz(p, r, kind, S) if p and os.path.exists(p) else [f"npz not on disk: {p}"])
            probs += [f"{r['unit_id']}: {x}" for x in pr]
        out[f"{key[0]}|{key[1]}"] = dict(status="pass" if not probs else "fail", n_rows=len(rs), checked=len(pick), problems=probs[:20])
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--records", required=True, help="glob of records jsonl")
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--kind", default="closed", choices=["closed", "open"])
    a = ap.parse_args()
    rows, seen = [], set()
    for f in sorted(glob.glob(a.records)):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r["unit_id"] not in seen:
                    seen.add(r["unit_id"]); rows.append(r)
    res = verify_rows(rows, a.sample, kind=a.kind)
    os.makedirs(f"{ROOT}/libero/out/artifact_verify", exist_ok=True)
    out = f"{ROOT}/libero/out/artifact_verify/{a.name}.json"
    json.dump(dict(name=a.name, records=a.records, sample_per_cell=a.sample, min_artifacts_version=spec()["version"], cells=res),
              open(out, "w"), ensure_ascii=False, indent=1)
    bad = {k: v for k, v in res.items() if v["status"] != "pass"}
    print(f"{a.name}: {len(res)} cells, {len(bad)} fail -> {out}")
    for k, v in list(bad.items())[:10]:
        print("  FAIL", k, v["problems"][:3])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
