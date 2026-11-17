"""数据核对 2026-11-15 · 第 1 步：从队友源文件（只读）独立算出 Cosmos Policy / VPP2 每格核验值。
只用标准库。不调用 ext_models.py / cat_summary.py / make_page.py。输出 verified.json + verify_sources.log。
  python3 verify_sources.py
源：
  Cosmos T0/T1：optimism_update_20261114/unit_index.csv（cohort=current_retained_5x10, repeat_index=1）
                逐单元再读 canonical_source 的 complete.json 的 closed_loop.success / first_success_step 复核；
                再与 web_audit_20261115/cosmos_T0_T1_all_current.csv 的 current_n / current_successes / current_median 对照。
  Cosmos T2–T4（非 F）：cosmos_policy/analysis/closed_loop_audit_20261111/unit_outcomes.csv 第一遍（primary）。
  Cosmos F：f_family_20261114/results/T*/F*/units/*/complete.json repeats[0]（原始）
            ↔ f_family_20261114/terminal_manifest.json primary_for_statistics ↔ f_upload_20261115/condition_results.json。
  VPP2 none：vpp2/results/T*_main_v1/units/*/complete.json repeats[0]（原始） ↔ web_audit_20261115/vpp2_raw_unit_counts.json。
  VPP2 正式扰动：vpp2/results/perturbations_20261114/T*/<cond>/units/*/complete.json 计数。
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import collections
import csv
import glob
import json
import math
import os

SA = _wmvp.P("$COSMOS_DATA_ROOT")
OUT = os.path.dirname(os.path.abspath(__file__))
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); LOG.append(s); print(s)


# ---------------- 档位名映射（与网页 DATA 的 level 命名一致；零剂量对照档不进表）
NR = {0.05: 1, 0.1: 2, 0.2: 3, 0.3: 4, 0.4: 5}


def level_of(c):
    if c == "baseline":
        return "none"
    if c == "N1_step50":
        return "N1_1"
    if c.startswith("NR_"):
        return f"NR{NR[round(float(c[3:]), 2)]}"
    if c.startswith("NL_"):
        i = int(c[3:]); return f"NL{i}" if i else None
    if c.startswith("NLw"):
        return None                      # NLw 不纳入展示
    if c in ("O0", "Em0", "Ef0", "G0", "K0", "N_sham", "F0"):
        return None
    return c


def lin_q(xs, p):
    """numpy.percentile 默认（linear）"""
    xs = sorted(xs); n = len(xs)
    h = (n - 1) * p; lo = math.floor(h); hi = min(lo + 1, n - 1)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def wilson(k, n):
    p = k / n; z = 1.96; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


def stats(n, k, steps):
    lo, hi = wilson(k, n)
    d = dict(n=n, k=k, rate=round(k / n, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), n_steps=len(steps))
    if steps:
        d.update(median=lin_q(steps, .5), q1=lin_q(steps, .25), q3=lin_q(steps, .75))
    else:
        d.update(median=None, q1=None, q3=None)
    return d


CELLS = {}          # (model, task, level) -> dict(stats..., steps, source, batch, denom, acq)


def put(model, task, level, units, source, batch, extra=None):
    n = len(units); k = sum(1 for u in units if u["ok"]); steps = [u["step"] for u in units if u["ok"]]
    assert all(u["step"] is not None for u in units if u["ok"]), (model, task, level, "成功单元缺步数")
    d = stats(n, k, steps); d.update(steps=sorted(steps), source=source, batch=batch); d.update(extra or {})
    CELLS[(model, task, level)] = d


# ======================= Cosmos T0/T1 当前保留集合
UI = f"{SA}/optimism_update_20261114/unit_index.csv"
rows = [r for r in csv.DictReader(open(UI, encoding="utf-8-sig"))
        if r["task"] in ("T0", "T1") and r["cohort"] == "current_retained_5x10"]
log("unit_index current_retained_5x10 行数", len(rows), collections.Counter((r["task"], r["acquisition"], r["repeat_index"]) for r in rows))
assert all(r["repeat_index"] == "1" for r in rows)
uid = collections.Counter((r["task"], r["unit_id"]) for r in rows)
assert max(uid.values()) == 1, "unit_id 重复"
raw_mismatch = []; pairing = collections.Counter()
grp = collections.defaultdict(list)
for r in rows:
    d = json.load(open(r["canonical_source"]))
    cl = d["closed_loop"]
    ok_raw = bool(cl["success"]); st_raw = cl.get("first_success_step") if ok_raw else None
    ok_idx = r["actual_success"] == "True"; st_idx = int(float(r["first_success_step"])) if ok_idx and r["first_success_step"] else None
    if ok_raw != ok_idx or (ok_raw and int(st_raw) != st_idx):
        raw_mismatch.append((r["unit_id"], ok_raw, st_raw, ok_idx, st_idx))
    # 原始记录里的身份字段与索引一致性（任务、条件、policy_seed）
    u = d.get("unit") or {}
    if u:
        assert u.get("task") == r["task"] and str(u.get("policy_seed")) == r["policy_seed"], r["unit_id"]
        cond = u.get("condition") or (u.get("setting_spec") or {}).get("setting")
        assert cond in (r["condition"], None), (r["unit_id"], cond)
        pairing[(r["task"], u.get("pairing_status"))] += 1
    else:
        assert d.get("task") == r["task"], r["unit_id"]
        pairing[(r["task"], "original_record(" + r["acquisition"] + ")")] += 1
    lv = level_of(r["condition"])
    if lv is None:
        continue
    grp[(r["task"], lv)].append(dict(ok=ok_raw, step=int(st_raw) if ok_raw else None, acq=r["acquisition"]))
log("T0/T1 逐单元：索引 vs 原始 complete.json 不一致", len(raw_mismatch), raw_mismatch[:5])
log("原始记录里的配对状态", dict(pairing))
for (T, lv), us in grp.items():
    acq = collections.Counter(u["acq"] for u in us)
    put("Cosmos Policy", T, lv, us, "optimism_update_20261114/unit_index.csv → canonical_source complete.json closed_loop",
        "T0=11-13 全量重跑（current_retained_5x10）" if T == "T0" else "T1 当前保留集合（完整重跑+原窗口恢复）",
        dict(acq=dict(acq)))

# 与 all_current.csv 对照
AC = list(csv.DictReader(open(f"{SA}/web_audit_20261115/cosmos_T0_T1_all_current.csv", encoding="utf-8-sig")))
ac_bad = []
for r in AC:
    lv = level_of(r["condition"])
    if lv is None:
        continue
    c = CELLS[("Cosmos Policy", r["task"], lv)]
    med = None if r["current_median"] in ("", "None", "nan") else float(r["current_median"])
    if int(r["current_n"]) != c["n"] or int(r["current_successes"]) != c["k"] or med != c["median"]:
        ac_bad.append((r["task"], r["condition"], r["current_n"], r["current_successes"], r["current_median"], c["n"], c["k"], c["median"]))
log("all_current.csv 行", len(AC), "（不含零剂量后）对照不一致", len(ac_bad), ac_bad[:5])

# 配对表（rule 4）：按 任务+条件+实际锚点+policy_seed+环境 seed 计数
NT = list(csv.DictReader(open(f"{SA}/web_audit_20261115/pairing_audit/Cosmos_T0_T1_new_to_old_6400.csv", encoding="utf-8-sig")))
pc = collections.Counter((r["task"], r["match_status"]) for r in NT)
acq_c = collections.Counter((r["task"], r["acquisition"]) for r in NT)
key_dup = collections.Counter((r["task"], r["condition"], r["anchor_id"], r["policy_seed"], r["environment_seed"]) for r in NT)
seed_mis = sum(1 for r in NT if r["old_policy_seed"] not in ("", r["policy_seed"]))
log("配对表 6400 行：match_status", dict(pc), "acquisition", dict(acq_c), "键重复", sum(v > 1 for v in key_dup.values()),
    "旧/新 policy_seed 不同", seed_mis)
PAIR = dict(match_status={f"{a}|{b}": v for (a, b), v in pc.items()}, acquisition={f"{a}|{b}": v for (a, b), v in acq_c.items()})

# ======================= Cosmos T2–T4（非 F）：11-11 审计批次
UO = f"{SA}/cosmos_policy/analysis/closed_loop_audit_20261111/unit_outcomes.csv"
g2 = collections.defaultdict(list)
for r in csv.DictReader(open(UO, encoding="utf-8-sig")):
    if r["task"] not in ("T2", "T3", "T4"):
        continue
    lv = level_of(r["condition"])
    if lv is None:
        continue
    ok = r["primary_success"] == "True"
    g2[(r["task"], lv)].append(dict(ok=ok, step=int(float(r["primary_first_success_step"])) if ok else None))
for (T, lv), us in g2.items():
    put("Cosmos Policy", T, lv, us, "cosmos_policy/analysis/closed_loop_audit_20261111/unit_outcomes.csv（primary）", "11-11 闭环审计批次")

# ======================= Cosmos F
CR = {(c["task"], c["condition"]): c for c in json.load(open(f"{SA}/f_upload_20261115/condition_results.json"))}
CR2 = {(c["task"], c["condition"]): c for c in json.load(open(f"{SA}/f_family_20261114/condition_results.json"))}
assert CR == CR2
TM = collections.defaultdict(list)
for e in json.load(open(f"{SA}/f_family_20261114/terminal_manifest.json")):
    if e.get("primary_for_statistics"):
        T, cond = e["unit_id"].split("/")[1], e["condition"]
        TM[(T, cond)].append(e)
f_bad = []
for (T, cond), c in sorted(CR.items()):
    units = []; trig = early = 0
    files = sorted(glob.glob(f"{SA}/f_family_20261114/results/{T}/{cond}/units/*/complete.json"))
    for f in files:
        d = json.load(open(f)); r0 = d["repeats"][0]
        assert d.get("first_repeat_primary", True)
        ok = bool(r0["success"]); units.append(dict(ok=ok, step=int(r0["first_success_step"]) if ok else None))
        op = r0.get("observation_perturbation") or {}
        trig += bool(op.get("blackout_applied")); early += (not op.get("blackout_applied"))
    tm = TM[(T, cond)]
    tm_k = sum(bool(e["actual_success"]) for e in tm); tm_steps = sorted(int(e["first_success_step"]) for e in tm if e["actual_success"])
    put("Cosmos Policy", T, cond, units, "f_family_20261114/results/*/units/*/complete.json repeats[0]", "F 族 11-14（current 5×10 起点，两遍取第一遍）",
        dict(trigger=trig, early=early, episodes=2 * len(files)))
    x = CELLS[("Cosmos Policy", T, cond)]
    chk = [("raw_n", x["n"], c["completed_units"]), ("raw_k", x["k"], c["primary_successes"]),
           ("median", x["median"], c["success_step_median"]), ("q1", x["q1"], c["success_step_q1"]), ("q3", x["q3"], c["success_step_q3"]),
           ("trigger", trig, c["primary_blackout_applied"]), ("early", early, c["primary_ended_before_trigger"]),
           ("manifest_n", len(tm), x["n"]), ("manifest_k", tm_k, x["k"]), ("manifest_steps", tm_steps, x["steps"]),
           ("episodes", 2 * len(files), c["completed_episodes"])]
    for name, a, b in chk:
        if (a if not isinstance(a, float) else round(a, 4)) != (b if not isinstance(b, float) else round(b, 4)):
            f_bad.append((T, cond, name, a, b))
log("F：原始 complete.json vs condition_results.json vs terminal_manifest primary 不一致", len(f_bad), f_bad[:6])
F_INFO = {f"{T}|{c}": dict(trigger=v["primary_blackout_applied"], early=v["primary_ended_before_trigger"], actual_step=v["actual_trigger_grid_step"],
                           imagined=v["imagined_success"], status=v["status"]) for (T, c), v in CR.items()}

# ======================= VPP2
VR = {u["source"]: u for u in json.load(open(f"{SA}/web_audit_20261115/vpp2_raw_unit_counts.json"))}
v_bad = []
for T in ("T0", "T1", "T2", "T3", "T4"):
    us = []
    for f in sorted(glob.glob(f"{SA}/vpp2/results/{T}_main_v1/units/*/complete.json")):
        d = json.load(open(f)); r0 = d["repeats"][0]
        ok = bool(r0["success"]); st = int(r0["first_success_step"]) if ok else None
        us.append(dict(ok=ok, step=st))
        v = VR.get(f)
        if v is None or bool(v["success"]) != ok or (ok and int(v["first_success_step"]) != st):
            v_bad.append((T, f, ok, st, v))
    put("VPP2", T, "none", us, f"vpp2/results/{T}_main_v1/units/*/complete.json repeats[0]", "VPP2 无扰动主批次（5 锚 × 10 seed）")
log("VPP2 原始 vs vpp2_raw_unit_counts.json 不一致", len(v_bad), v_bad[:3])
pert = collections.Counter(); pert_conds = 0
for cd in sorted(glob.glob(f"{SA}/vpp2/results/perturbations_20261114/T*/*/")):
    if not os.path.exists(f"{cd}/manifest.json"):
        continue
    pert_conds += 1
    pert["unit_dirs"] += len(glob.glob(f"{cd}/units/*/"))
    pert["complete_json"] += len(glob.glob(f"{cd}/units/*/complete.json"))
log("VPP2 正式扰动条件目录", pert_conds, dict(pert))

# ======================= 六类汇总（P→环境·一次性，O→信息·一次性，不含 NLw；基线/对照 none/A0/P0/B0 不进类）
CATS = [("base", ["none"]),
        ("env_once", [f"A{i}" for i in range(1, 5)] + [f"P{i}" for i in range(1, 5)] + [f"K{i}" for i in range(1, 10)]),
        ("env_always", [f"Em{i}" for i in range(1, 5)] + [f"Ef{i}" for i in range(1, 5)]),
        ("phys_once", ["N1_1"]),
        ("phys_always", [f"G{i}" for i in range(1, 5)] + [f"NR{i}" for i in range(1, 6)] + [f"NL{i}" for i in range(1, 6)]),
        ("info_once", [f"O{i}" for i in range(1, 6)] + ["F60", "F160", "F260"]),
        ("info_always", [f"B{t}{s}" for t in "gfz" for s in (2, 5, 9)])]
CATV = {}
for m in ("Cosmos Policy", "VPP2"):
    for T in ("T0", "T1", "T2", "T3", "T4"):
        for key, lvs in CATS:
            got = [l for l in lvs if (m, T, l) in CELLS]; miss = [l for l in lvs if (m, T, l) not in CELLS]
            n = sum(CELLS[(m, T, l)]["n"] for l in got); k = sum(CELLS[(m, T, l)]["k"] for l in got)
            d = dict(n=n, k=k, levels_used=" ".join(l.replace("N1_1", "N1") for l in got), levels_missing=" ".join(l.replace("N1_1", "N1") for l in miss))
            if n:
                lo, hi = wilson(k, n); d.update(rate=round(k / n, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4))
            CATV[f"{m}|{T}|{key}"] = d

json.dump(dict(cells={"|".join(k): v for k, v in CELLS.items()}, cats=CATV, f_info=F_INFO, pairing=PAIR,
               vpp2_perturb=dict(conditions=pert_conds, **pert),
               checks=dict(t01_raw_vs_index=len(raw_mismatch), all_current_vs_raw=len(ac_bad), f_three_way=len(f_bad), vpp2_raw_vs_index=len(v_bad))),
          open(f"{OUT}/verified.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
open(f"{OUT}/verify_sources.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
log("cells", len(CELLS), "cats", len(CATV))
