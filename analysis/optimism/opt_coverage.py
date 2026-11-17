"""终局想象乐观率 · 第 1 步：清点所有存了【闭环】想象帧的单元（只读现有数据，不补采）。
输出 out/optimism/units.json（逐单元：模型、任务、族、档、来源、帧文件、帧的键、现实成败）与 coverage.tsv（模型 × 任务 × 族：单元数、失败数、有无判定器）。
现实成败 = 想象帧所属那一遍闭环的模拟器判定（同一遍，不混用两遍）。
来源（2026-11-12 清点）：
  ow_backfill  runs/openwam_{base,famA,famP}/closed_frames/<uid>.npz      键 imagined（第一遍）  标签 closed_frames_check.*.jsonl success[0]
  ow_phys      runs/openwam_t{4,6}/closed/<uid>.npz                       键 imagined            标签 phys/closed.*.jsonl closed_loop.success
  ow_famB      runs/openwam_t{4,6}/famB/closed_imagined/<uid>.npz         键 imagined            标签 famB/records*.jsonl closed_loop.success
  ow_rerun     runs/openwam_t{2,4,6}/rerun_closed/phys/<uid>.npz          键 imagined            标签 rerun_closed/phys.*.jsonl rerun.success
  lb_famB_t2   runs/lingbot_t2/famB/closed_frames/<uid>.npz               键 latents_NN（需解码）标签 famB/records.*.jsonl closed_loop.success
  lb_rerun     runs/lingbot_t{2,4,6}/rerun_closed/<src>/<uid>.npz         键 imag_latents_NN     标签 rerun_closed/<src>.*.jsonl rerun.success
  lb_famF      runs/lingbot_famF/closed/<uid>.npz                         键 imag_latents_NN     标签 famF/closed_t*.jsonl closed_loop.success（2026-11-15 加入）
  ow_famF      runs/openwam_famF/closed/<uid>.npz                         键 imagined            标签 famF/closed_t*.jsonl closed_loop.success（2026-11-15 加入）
判定器（protocol.yaml / experiment_parameters.md 已部署版）：LingBot t5 = v3、LingBot t3 = v4 联合、OpenWAM t3 = OpenWAM 视角 v4 联合；
OpenWAM t5 无（三次未过门槛）；t2/t4/t6 两模型都无。
  python3 opt_coverage.py
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import collections
import csv
import glob
import json
import os

RUNS = _wmvp.P("$WMV_ROOT/libero/out/runs")
ROOT = _wmvp.P("$WMV_ROOT")
HERE = os.path.dirname(os.path.abspath(__file__))
TN = {5: "T0", 3: "T1", 2: "T2", 4: "T3", 6: "T4"}
JUDGE = {("lingbot", "T0"): "LingBot t5 v3", ("lingbot", "T1"): "LingBot t3 v4 联合", ("openwam", "T1"): "OpenWAM t3 v4 联合"}


def rows(pat):
    out = []
    for f in sorted(glob.glob(f"{RUNS}/{pat}")):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def tid_of(uid):
    return int(uid.split("_libero10_t")[1].split("_")[0])


def level_of(s):
    s = s or "none"
    return s.split("__")[-1] if "__" in s else s


def fam(lv):
    if lv == "none":
        return "none"
    for p in ("NLw", "NL", "NR", "N1", "Em", "Ef"):
        if lv.startswith(p):
            return p
    return "B" if lv[0] == "B" else lv[0]


def add(U, seen, **u):
    if not os.path.exists(u["frames"]) or (u["model"], u["unit_id"]) in seen:
        return
    seen.add((u["model"], u["unit_id"]))
    u["family"] = fam(u["level"]); u["judge"] = JUDGE.get((u["model"], u["T"]), "")
    U.append(u)


def main():
    U, seen = [], set()
    for run in ("openwam_base", "openwam_famA", "openwam_famP"):
        for r in rows(f"{run}/closed_frames_check.*.jsonl"):
            lv = "none" if run == "openwam_base" else r["unit_id"].split("__")[-1]
            add(U, seen, model="openwam", T=TN[tid_of(r["unit_id"])], level=lv, source="ow_backfill", unit_id=r["unit_id"],
                frames=f"{ROOT}/{r['frames_file']}", key="imagined", success=bool(r["success"][0]))
    for tid in (4, 6):
        for r in rows(f"openwam_t{tid}/phys/closed.*.jsonl"):
            add(U, seen, model="openwam", T=TN[tid], level=level_of(r["setting_id"]), source="ow_phys", unit_id=r["unit_id"],
                frames=f"{RUNS}/openwam_t{tid}/closed/{r['unit_id']}.npz", key="imagined", success=bool(r["closed_loop"]["success"]))
        for r in rows(f"openwam_t{tid}/famB/records*.jsonl"):
            add(U, seen, model="openwam", T=TN[tid], level=level_of(r["setting_id"]), source="ow_famB", unit_id=r["unit_id"],
                frames=f"{RUNS}/openwam_t{tid}/famB/closed_imagined/{r['unit_id']}.npz", key="imagined",
                success=bool(r["closed_loop"]["success"]))
    for tid in (2, 4, 6):
        for r in rows(f"openwam_t{tid}/rerun_closed/phys.*.jsonl"):
            add(U, seen, model="openwam", T=TN[tid], level=r["level"], source="ow_rerun", unit_id=r["unit_id"],
                frames=f"{RUNS}/openwam_t{tid}/rerun_closed/phys/{r['unit_id']}.npz", key="imagined", success=bool(r["rerun"]["success"]))
    for r in rows("lingbot_t2/famB/records.*.jsonl"):
        add(U, seen, model="lingbot", T="T2", level=level_of(r["setting_id"]), source="lb_famB_t2", unit_id=r["unit_id"],
            frames=f"{RUNS}/lingbot_t2/famB/closed_frames/{r['unit_id']}.npz", key="latents", success=bool(r["closed_loop"]["success"]))
    for tid in (2, 4, 6):
        for r in rows(f"lingbot_t{tid}/rerun_closed/*.jsonl"):
            add(U, seen, model="lingbot", T=TN[tid], level=r["level"], source="lb_rerun", unit_id=r["unit_id"],
                frames=f"{ROOT}/{r['frames_file']}", key="imag_latents", success=bool(r["rerun"]["success"]))
    for mk in ("lingbot", "openwam"):                 # F 族（11-13 起按存档规矩跑，想象帧齐全），第一遍的成败
        for r in rows(f"{mk}_famF/closed_t*.jsonl"):
            add(U, seen, model=mk, T=TN[tid_of(r["unit_id"])], level=r["setting_id"], source={"lingbot": "lb_famF", "openwam": "ow_famF"}[mk],
                unit_id=r["unit_id"], frames=f"{ROOT}/{r['closed_loop_file']}",
                key="imag_latents" if mk == "lingbot" else "imagined", success=bool(r["closed_loop"]["success"]))
    json.dump(U, open(f"{HERE}/units.json", "w"), ensure_ascii=False, indent=0)
    c = collections.defaultdict(lambda: [0, 0])
    for u in U:
        c[(u["model"], u["T"], u["family"])][0] += 1; c[(u["model"], u["T"], u["family"])][1] += (not u["success"])
    order = ["none", "A", "P", "O", "Em", "Ef", "G", "K", "F", "N1", "NR", "NL", "NLw", "B"]
    with open(f"{HERE}/coverage.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["model", "task", "family", "n_units", "n_real_fail", "judge", "sources"])
        for (m, T, fa), (n, nf) in sorted(c.items(), key=lambda x: (x[0][0], x[0][1], order.index(x[0][2]))):
            src = sorted({u["source"] for u in U if (u["model"], u["T"], u["family"]) == (m, T, fa)})
            w.writerow([m, T, fa, n, nf, JUDGE.get((m, T), "无"), " ".join(src)])
    tot = collections.Counter((u["model"], u["T"]) for u in U)
    print("units", len(U), sorted(tot.items()))
    print("with judge", sum(1 for u in U if u["judge"]), "fail with judge", sum(1 for u in U if u["judge"] and not u["success"]))


if __name__ == "__main__":
    main()
