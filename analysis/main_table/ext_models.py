"""队友模型（Cosmos Policy / UVA）闭环逐单元结果 → 主表档位口径。只读队友目录，不改任何源文件。

每个 loader 返回 {(T, level): dict(n, k, steps=[首次成功步...], note)}，level 用主表命名（none/A0/Em1/NR3/Bg5…）。
统一口径：只取第一遍闭环（primary / repeat0），H=400，失败计入分母；步数只统计成功单元。
零剂量对照档（O0/Em0/Ef0/G0/K0/N_sham/NL0/NLw0 等）不进主表——LingBot/OpenWAM 主表也不列这些档。
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import csv
import glob
import json
import os
import collections

COSMOS = _wmvp.P("$COSMOS_DATA_ROOT/cosmos_policy/analysis/closed_loop_audit_20261111/unit_outcomes.csv")
COSMOS_CURRENT = _wmvp.P("$COSMOS_DATA_ROOT/optimism_update_20261114/unit_index.csv")   # T0/T1 当前保留集合（2026-11-15）
COSMOS_F = _wmvp.P("$COSMOS_DATA_ROOT/f_family_20261114/terminal_manifest.json")   # F 族（2026-11-15）
COSMOS_F_CR = _wmvp.P("$COSMOS_DATA_ROOT/f_family_20261114/condition_results.json")   # F 族逐格汇总：未触发置黑数、实际置黑步（2026-11-16 v6）
COSMOS_PAIRING = _wmvp.P("$COSMOS_DATA_ROOT/web_audit_20261115/pairing_audit/Cosmos_T0_T1_new_to_old_6400.csv")   # T0/T1 新旧配对表
VPP2_PERTURB = _wmvp.P("$COSMOS_DATA_ROOT/vpp2/results/perturbations_20261114")   # VPP2 正式扰动（只计数，不并入）
RESTORED = "original_episode_final_window_restored"   # unit_index.csv acquisition 的实际取值（v5 前误写为 original_episode_last_window_recovered）
T1_MIXED_NOTE = "T1 当前保留集合：部分单元为原闭环（只补想象帧），成败与步数取原闭环"
VPP2_RAW = _wmvp.P("$COSMOS_DATA_ROOT/web_audit_20261115/vpp2_raw_unit_counts.json")   # VPP2 逐单元原始记录（T4 步数）
UVA_NAS = _wmvp.P("$UVA_DATA_ROOT/20261102-native-candidate")

# phys_perturb.FAMILIES 的档位值 → 主表档位下标（第 0 档 = 零剂量，不进表）
EM = {1.5: 1, 2.0: 2, 4.0: 3, 8.0: 4}
EF = {0.8: 1, 0.6: 2, 0.4: 3, 0.2: 4}
GG = {0.9: 1, 0.75: 2, 0.55: 3, 0.4: 4}
NR = {0.05: 1, 0.1: 2, 0.2: 3, 0.3: 4, 0.4: 5}
NLW = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}
OCC = {8: 1, 16: 2, 24: 3, 32: 4, 40: 5}
BX = {"gaussian_blur": "Bg", "fog": "Bf", "zoom_blur": "Bz"}


# ---------------- F 族“置黑前已成功”统一口径（数据核对规则 7；2026-11-16 v6 三个模型共用）
# 逐单元：第一遍闭环成功且 首次成功步 ≤ 实际置黑步 → 置黑前已成功，置黑不可能影响其成败。
# （≤：第 s 步判成功时回合在第 s 步的规划调用读帧之前结束；LingBot-VA 与 Cosmos 的 blackout_applied 标记与此逐单元一致。）
# 计数为 0 的格不写说明（与 11-15 数据核对一致）。
def f_prior_success(units):
    """units: [dict(success, first_success_step, blackout_step, applied)]，blackout_step = 该格实际置黑步。"""
    n = len(units)
    early_u = [bool(u["success"]) and u["first_success_step"] is not None and int(u["first_success_step"]) <= int(u["blackout_step"]) for u in units]
    applied = sum(bool(u["applied"]) for u in units)
    steps = {int(u["blackout_step"]) for u in units}
    assert len(steps) == 1, steps
    if all(e == (not u["applied"]) for e, u in zip(early_u, units)):
        mode = "ends_on_success"          # 成功即结束回合：置黑前已成功的单元未触发置黑
    elif applied == n:
        mode = "runs_to_H"                # 成功后不提前结束：置黑照常施加，不改变已记录的成功
    else:
        raise AssertionError(("置黑标记与首次成功步不自洽", n, sum(early_u), applied))
    return dict(n=n, early=sum(early_u), applied=applied, step=steps.pop(), mode=mode)


def f_note(info, batch=""):
    n, e, a = info["n"], info["early"], info["applied"]
    if not e:
        return ""
    head = f"F 族{batch}：{n} 个独立单元，每单元两遍、取第一遍（不合计为 {2 * n}）；"
    if info["mode"] == "ends_on_success":
        return head + f"其中 {e} 个在置黑前已成功结束、未触发置黑，实际触发 {a} 个"
    return head + (f"其中 {e} 个在置黑前已成功（首次成功步 ≤ 实际置黑步 {info['step']}）；该模型成功后回合不提前结束，"
                   f"置黑仍按时施加（实际施加 {a} 个），不改变已记录的成功")


OURS_F_GLOB = _wmvp.P("$WMV_ROOT/libero/out/closed_main/T*_t*/{m}_famF.jsonl")


def ours_f(models=("lingbot", "openwam")):
    """LingBot-VA / OpenWAM-α F 族逐格：读 closed_main/T*/<model>_famF.jsonl 第一遍闭环（closed_loop），按 f_prior_success 统一口径。"""
    U = collections.defaultdict(list)
    for m in models:
        for f in sorted(glob.glob(OURS_F_GLOB.format(m=m))):
            T = os.path.basename(os.path.dirname(f)).split("_")[0]
            for line in open(f):
                r = json.loads(line); cl = r.get("closed_loop") or {}
                if cl.get("success") is None:
                    continue
                p = r["perturbation"]
                step = p["actual_blackout_step"] if r.get("blackout_applied") else p["expected_blackout_step"]
                assert not r.get("blackout_applied") or p["actual_blackout_step"] == p["expected_blackout_step"], r["unit_id"]
                U[(m, T, r["setting_id"])].append(dict(success=cl["success"], first_success_step=cl.get("first_success_step"),
                                                       blackout_step=step, applied=r.get("blackout_applied")))
    out = {}
    for key, us in U.items():
        info = f_prior_success(us); info["note"] = f_note(info); out[key] = info
    return out


def cosmos_level(c):
    if c == "baseline":
        return "none"
    if c == "N1_step50":
        return "N1_1"
    if c.startswith("NR_"):
        return f"NR{NR[round(float(c[3:]), 2)]}"
    if c.startswith("NL_"):
        i = int(c[3:]); return f"NL{i}" if i else None
    if c in ("O0", "Em0", "Ef0", "G0", "K0", "N_sham"):
        return None
    return c                                    # A0-4 P0-4 O1-5 Em1-4 Ef1-4 G1-4 K1-9 B0 Bg/Bf/Bz{2,5,9}


def cosmos():
    A = collections.defaultdict(lambda: dict(n=0, k=0, steps=[], anchors=set(), seeds=set(), note=""))
    for r in csv.DictReader(open(COSMOS, encoding="utf-8-sig")):
        l = cosmos_level(r["condition"])
        if l is None:
            continue
        a = A[(r["task"], l)]
        a["n"] += 1
        ok = r["primary_success"] == "True"
        a["k"] += ok
        if ok and r["primary_first_success_step"]:
            a["steps"].append(int(float(r["primary_first_success_step"])))
        a["anchors"].add(r["anchor"]); a["seeds"].add(r["seed"])
    # 2026-11-15 队友核对（web_audit_20261115）：T0/T1 改用当前保留集合 optimism_update_20261114/unit_index.csv（每条件 5 起点 × 10 seed）；
    # T0 = 11-13 全量重跑；T1 = 2101 条完整重跑 + 1099 条原闭环（末窗恢复只补想象帧，成败与步数取原闭环）。旧 11-11 批次的 T0/T1 不再进主表。
    for T in ("T0", "T1"):
        for key in [k for k in A if k[0] == T]:
            del A[key]
    for r in csv.DictReader(open(COSMOS_CURRENT, encoding="utf-8-sig")):
        if r["task"] not in ("T0", "T1") or r.get("cohort") != "current_retained_5x10" or str(r.get("repeat_index", "1")) not in ("1", ""):
            continue
        l = cosmos_level(r["condition"])
        if l is None:
            continue
        a = A[(r["task"], l)]
        a["n"] += 1; ok = r["actual_success"] == "True"; a["k"] += ok
        if ok and r["first_success_step"]:
            a["steps"].append(int(float(r["first_success_step"])))
        a["anchors"].add(r["anchor_index"]); a["seeds"].add(r["policy_seed"])
        a.setdefault("acq", collections.Counter())[r["acquisition"]] += 1
    # F 族（2026-11-15 并入）：f_family_20261114 terminal_manifest，主遍（primary_for_statistics），每格 50 单元
    FU = collections.defaultdict(list)
    for e in json.load(open(COSMOS_F)):
        if not e.get("primary_for_statistics"):
            continue
        T = e["unit_id"].split("/")[1]; a = A[(T, e["condition"])]
        FU[(T, e["condition"])].append(e)
        a["n"] += 1; ok = bool(e["actual_success"]); a["k"] += ok
        if ok and e.get("first_success_step"):
            a["steps"].append(int(e["first_success_step"]))
        a["anchors"].add(e["anchor"].get("anchor_id")); a["seeds"].add(e.get("policy_seed"))
    # 2026-11-16 v6（数据核对 2026-11-15 第五节 1）：按 acquisition 的实际取值计数。整格都是原窗口恢复 → 写明 r/n；
    # 恢复与重跑混合（T1 none）→ 沿用原说明。
    for key, a in A.items():
        acq = a.pop("acq", None)
        r = acq.get(RESTORED, 0) if acq else 0
        if r and r == a["n"]:
            a["note"] = f"T1 当前保留集合：{r}/{a['n']} 个单元为原闭环末窗恢复（只补想象帧，不增加闭环样本），成败与步数取原闭环"
        elif r or (acq and len(acq) > 1):
            a["note"] = T1_MIXED_NOTE
    # F 族：第一遍在置黑前已成功的单元数写进口径说明（数据核对规则 7）。逐单元按 f_prior_success 统一口径
    # （terminal_manifest 主遍的首次成功步 vs 该格实际置黑步），再与作者 condition_results.json 逐格核对 n / 成功数 / 未触发数 / 触发数。
    for c in json.load(open(COSMOS_F_CR)):
        key = (c["task"], c["condition"]); a = A.get(key)
        assert a is not None and a["n"] == c["completed_units"] and a["k"] == c["primary_successes"], key
        info = f_prior_success([dict(success=e["actual_success"], first_success_step=e.get("first_success_step"),
                                     blackout_step=c["actual_trigger_grid_step"],
                                     applied=(e.get("observation_perturbation") or {}).get("blackout_applied")) for e in FU[key]])
        assert (info["n"], info["early"], info["applied"]) == (a["n"], c["primary_ended_before_trigger"], c["primary_blackout_applied"]), (key, info)
        if info["early"]:
            a["note"] = f_note(info, batch=" 11-14 批次")
    for key, a in A.items():
        if len(a["seeds"]) == 1 and len(a["anchors"]) > 5:
            a["note"] = f"采样为 {len(a['anchors'])} 个官方起点 × 1 seed（不是 5 锚 × 10 seed）"
        del a["anchors"], a["seeds"]
    return dict(A)


def cosmos_meta():
    """页底模型说明 / 表注里 Cosmos Policy 的计数（2026-11-16 v6；数据核对 texts.py 同一口径，全部由源文件算出，不手抄）。"""
    NT = list(csv.DictReader(open(COSMOS_PAIRING, encoding="utf-8-sig")))
    cnt = collections.Counter((r["task"], r["match_status"]) for r in NT)
    acq = collections.Counter((r["task"], r["acquisition"]) for r in NT)
    tot = collections.Counter(r["task"] for r in NT)
    M = dict(t0_cur=tot["T0"], t0_m=cnt[("T0", "same_task_condition_anchor_seed")], t1_cur=tot["T1"],
             t1_m=cnt[("T1", "same_task_condition_anchor_seed")], t1_fresh=acq[("T1", "fresh_full_episode")], t1_rest=acq[("T1", RESTORED)])
    M["t0_new"] = M["t0_cur"] - M["t0_m"]
    CR = json.load(open(COSMOS_F_CR))
    steps = {}
    for c in CR:
        steps.setdefault(c["condition"], set()).add(c["actual_trigger_grid_step"])
    assert all(len(v) == 1 for v in steps.values()), steps
    M["fstep"] = {l: next(iter(v)) for l, v in steps.items()}
    ns = {c["completed_units"] for c in CR}
    assert len(ns) == 1, ns
    M["fn"] = ns.pop()
    M["early"] = {(c["task"], c["condition"]): c["primary_ended_before_trigger"] for c in CR}
    return M


def cosmos_f_text(M, sep="/"):
    return sep.join(str(M["fstep"][l]) for l in ("F60", "F160", "F260"))


def early_text(early, sep="，"):
    """early: {(T, level): 置黑前已成功数} → “F160：T0 10、T1 0、…，F260：…”（只列有非零的档）。"""
    parts = []
    for l in ("F60", "F160", "F260"):
        e = [(T, early.get((T, l), 0)) for T in ("T0", "T1", "T2", "T3", "T4")]
        if any(v for _, v in e):
            parts.append(f"{l}：" + "、".join(f"{T} {v}" for T, v in e))
    return sep.join(parts)


def cosmos_early_text(M):
    parts = []
    for l in ("F60", "F160", "F260"):
        e = [(T, M["early"].get((T, l), 0)) for T in ("T0", "T1", "T2", "T3", "T4")]
        if any(v for _, v in e):
            parts.append(f"{l}：" + "、".join(f"{T} {v}" for T, v in e))
    return "；".join(parts)


def vpp2_meta():
    """VPP2 T4 逐单元记录与正式扰动目录计数（2026-11-16 v6）。"""
    raw = [u for u in json.load(open(VPP2_RAW)) if u["task"] == "T4"]
    conds = [d for d in sorted(glob.glob(f"{VPP2_PERTURB}/T*/*/")) if os.path.exists(f"{d}/manifest.json")]
    done = sum(len(glob.glob(f"{d}/units/*/complete.json")) for d in conds)
    return dict(v4n=len(raw), v4k=sum(bool(u["success"]) for u in raw), vconds=len(conds), vdone=done)


def uva_level(fam, lv):
    lv = None if lv is None else float(lv)
    if fam in ("normal", "baseline"):
        return "none"
    if fam == "B0":
        return "B0"
    if fam in ("A", "P"):
        return f"{fam}{int(lv)}"
    if fam == "O":                              # UVA 用遮挡边长像素 8/16/24/32/40 = 1/16..5/16 × 128
        return f"O{OCC[int(lv)]}" if int(lv) in OCC else None
    if fam == "E_mass":
        return f"Em{EM[lv]}" if lv in EM else None
    if fam == "E_friction":
        return f"Ef{EF[round(lv, 2)]}" if round(lv, 2) in EF else None
    if fam == "G":
        return f"G{GG[round(lv, 2)]}" if round(lv, 2) in GG else None
    if fam == "N1":
        return "N1_1"
    if fam == "NR":
        return f"NR{NR[round(lv, 2)]}" if round(lv, 2) in NR else None
    if fam == "NL":
        return f"NL{int(lv)}" if lv >= 1 else None
    if fam == "NLw":
        return f"NLw{NLW[int(lv)]}" if int(lv) in NLW else None
    if fam in BX:
        return f"{BX[fam]}{int(lv)}"
    raise KeyError((fam, lv))


def uva(index_json):
    """index_json：build_uva_index.py 生成的 [{task, family, level, job, closed_dir, formal_open}]。
    closed_dir 下每单元 <unit>/repeat0/result.json。"""
    A = {}
    for c in json.load(open(index_json)):
        l = uva_level(c["family"], c["level"])
        if l is None:
            continue
        n = k = 0; steps = []
        for f in sorted(glob.glob(f"{c['closed_dir']}/*/repeat0/result.json")):
            r = json.load(open(f))
            n += 1
            if r.get("native_success"):
                k += 1
                if r.get("first_native_success_step") is not None:
                    steps.append(int(r["first_native_success_step"]))
        if n:
            A[(c["task"], l)] = dict(n=n, k=k, steps=steps, note=c.get("note") or "UVA 作者标记 formal_results=false（未正式发布，视觉复核未完成）")
    return A


# 2026-11-12：作者审计汇总（11-11 18:03）之后新完成的 UVA 闭环——T4 Ef 族 v6 修复版（作业 1643–1647，日志均 passed）
UVA_T4_EF_V6 = f"{UVA_NAS}/t4-friction-v6-closed-20261112/friction-v6/T4"
UVA_EXTRA = [dict(task="T4", family="E_friction", level=x, closed_dir=f"{UVA_T4_EF_V6}/E_friction_{x}/closed",
                  note="UVA 作者标记 formal_results=false；T4 Ef 为作者 11-12 v6 修复版闭环，作者审计汇总尚未覆盖，成功数与作者作业日志逐档核对一致")
             for x in (0.8, 0.6, 0.4, 0.2)]


def uva_all(index_json, extra_path):
    """index + UVA_EXTRA；extra 只补 index 里没有的格。"""
    import tempfile
    cells = json.load(open(index_json))
    have = {(c["task"], c["family"], c["level"]) for c in cells}
    cells += [c for c in UVA_EXTRA if (c["task"], c["family"], c["level"]) not in have]
    json.dump(cells, open(extra_path, "w"), ensure_ascii=False)
    return uva(extra_path)


# UVA T0/T1（2026-11-12 并入）：作者 11-02~11-04 成对开/闭环批次（task5 = T0、task3 = T1），由 uva_t01_scan.py 扫出 uva_t01_cells.json；
# 每格 50 单元、单遍、H=400，逐单元 result.json 与作者 probe-complete.json 逐格核对一致。不在作者 11-06 后的闭环审计汇总里。
UVA_T01_CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uva_t01_cells.json")
UVA_T01_NOTE = "UVA 作者标记 formal_results=false；T0/T1 取自作者 11-02~04 成对开/闭环批次（单遍），不在作者 11-06 后的闭环审计汇总里，成功数已与作者 probe-complete 逐单元核对一致"
UVA_T01_EF_NOTE = UVA_T01_NOTE + "；旧版摩擦实现，附带渲染改动，与 T2–T4 的 v5/v6 口径不同"
UVA_T01_EF_EXCLUDE = "UVA T0/T1 Ef：旧版摩擦实现（附带渲染改动），排除在一切跨任务 Ef 汇总和跨模型 Ef 对比之外（用户裁决 2026-11-12）；待作者用 v5/v6 重跑后换入"


def uva_t01(path=UVA_T01_CELLS):
    A = {}
    for c in json.load(open(path))["cells"]:
        if c["level"] is None or not c["n"]:
            continue
        A[(c["task"], c["level"])] = dict(n=c["n"], k=c["k"], steps=c["steps"],
                                         note=UVA_T01_EF_NOTE if c["family"] == "E_friction" else UVA_T01_NOTE,
                                         exclude_cross=UVA_T01_EF_EXCLUDE if c["family"] == "E_friction" else "")
    return A


# VPP2（队友 $COLLAB_COSMOS，2026-11-12）：只有无扰动基线闭环。T0–T3 = Dropbox 交付的 units.csv（作者核验上传）；
# T4 = NAS results/T4_main_v1/summary.json（逐单元未上传，无步数）。源文件已存档到 ext_snapshots/vpp2_20261112/。
VPP2_SNAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext_snapshots/vpp2_20261112")


def vpp2():
    A = {}
    for r in csv.DictReader(open(f"{VPP2_SNAP}/units.csv", encoding="utf-8-sig")):
        a = A.setdefault((r["task"], "none"), dict(n=0, k=0, steps=[], note=""))
        a["n"] += 1
        if r["success1"] == "True":
            a["k"] += 1; a["steps"].append(int(r["steps1"]))
    summ = json.load(open(f"{VPP2_SNAP}/task_summary.json"))
    for T, a in A.items():
        s = summ[T[0]]
        assert s["complete"] and s["successes"] == a["k"] and s["completed_units"] == a["n"], (T, s, a)
    s4 = json.load(open(f"{VPP2_SNAP}/T4_summary.json"))
    if s4.get("complete"):
        raw = [u for u in json.load(open(VPP2_RAW)) if u["task"] == "T4"]       # 2026-11-15 队友核对：T4 有 50 个逐单元 complete.json，补步数
        k = sum(bool(u["success"]) for u in raw)
        assert len(raw) == s4["completed_units"] and k == s4["successes"], (len(raw), k, s4)
        A[("T4", "none")] = dict(n=len(raw), k=k, steps=[int(u["first_success_step"]) for u in raw if u["success"]],
                                 note="T4 取作者逐单元原始记录（vpp2/results/T4_main_v1/units/*/complete.json），与作者汇总一致")
    return A

# UVA K 族（2026-11-14 并入）：作者 11-13 批次 k-family-20261113-v1，T1–T4 × K1–K9，每格 50 单元 × 2 遍（H=400），主表取第一遍。
# uva_k_scan.py 逐格核对：作者 probe-complete 状态 passed、逐单元与 repeat0/result.json 一致、K 参数与 plan.json 一致、K0 门槛日志通过；
# 36 格两遍结论全部 50/50 一致。作者批次不含 T0。
UVA_K_CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uva_k_cells.json")
UVA_K_NOTE = ("UVA 作者标记 formal_results=false；K 族取自作者 11-13 批次（每格 50 单元 × 2 遍，主表取第一遍），作者审计汇总尚未覆盖；"
              "逐单元成功数已与作者 probe-complete 核对一致，两遍结论 50/50 一致，K0 门槛通过")


# 2026-11-16 v6：T0 K 族 = 作者 11-14 正式批 t0-k-ef-20261115/K-formal-v3（9 档 × 50 单元 × 2 遍 = 900 条轨迹），同一 uva_k_scan.py 核对。
UVA_K_T0_NOTE = ("UVA 作者标记 formal_results=false；T0 K 族取自作者 11-14 正式批 K-formal-v3（每格 50 单元 × 2 遍，主表取第一遍），作者审计汇总尚未覆盖；"
                 "逐单元成功数已与作者 probe-complete 核对一致，两遍结论 {agree}/{n} 一致，K0 门槛通过")


def uva_k(path=UVA_K_CELLS):
    A = {}
    for c in json.load(open(path))["cells"]:
        if c["problems"] or not c["n"]:
            continue
        note = UVA_K_T0_NOTE.format(agree=c["repeat_agree"], n=c["n"]) if c.get("batch") == "t0_formal_v3_20261114" else UVA_K_NOTE
        A[(c["task"], c["level"])] = dict(n=c["n"], k=c["k"], steps=c["steps"], note=note)
    return A


def uva_k_meta(path=UVA_K_CELLS):
    d = json.load(open(path))
    t0 = [c for c in d["cells"] if c.get("batch") == "t0_formal_v3_20261114" and not c["problems"]]
    return dict(t0_cells=len(t0), t0_units=sum(c["n"] for c in t0), t0_traj=d.get("t0_total_formal_trajectories"),
                t0_agree=sum(c["repeat_agree"] for c in t0), t0_gate=all(c["gate_passed"] for c in t0),
                t0_probe_passed=all(c.get("probe_status") == "passed" for c in t0))


# VPP2 NLw（2026-11-14 并入）：作者 $COLLAB_COSMOS 的 NLw 续跑 results/NLw_20261112（Cosmos 与 VPP2 共用说明，VPP2 部分在 vpp2/results 下），
# 每单元两遍、第一遍为主（作者 first_repeat_primary）。T0 五档完成；T1 NLw4–32 完成、NLw64 因用户 11-14 要求停 NLw 只到 42/50（作者保留已完成数据）；
# T2–T4 未跑。逐单元读 units/<u>/closed_loop1/complete.json，与作者每档 summary.json 的 completed_units / successes 逐格核对。
VPP2_NLW = _wmvp.P("$COSMOS_DATA_ROOT/vpp2/results/NLw_20261112")
VPP2_NLW_NOTE = "VPP2 NLw 取自作者 11-12 起的 NLw 续跑（第一遍闭环，两遍保存）；作者标记想象判定待验收"


def vpp2_nlw(root=VPP2_NLW):
    A, check = {}, []
    for T in ("T0", "T1", "T2", "T3", "T4"):
        for d, lv in ((4, "NLw1"), (8, "NLw2"), (16, "NLw3"), (32, "NLw4"), (64, "NLw5")):
            cd = f"{root}/{T}/NLw_{d}"
            if not os.path.exists(f"{cd}/summary.json"):
                continue
            S = json.load(open(f"{cd}/summary.json"))
            n = k = 0; steps = []
            for f in sorted(glob.glob(f"{cd}/units/*/closed_loop1/complete.json")):
                if not os.path.exists(f.replace("/closed_loop1/complete.json", "/complete.json")):
                    continue                         # 单元未完成（第二遍没跑完，例如 11-14 停 NLw 时正在跑的单元）
                r = json.load(open(f)); n += 1
                if r.get("success"):
                    k += 1
                    if r.get("first_success_step") is not None:
                        steps.append(int(r["first_success_step"]))
            ok = (n == S["completed_units"] and k == S["successes"])
            check.append((T, lv, n, k, S["completed_units"], S["successes"], S["complete"], ok))
            assert ok, (T, lv, n, k, S["completed_units"], S["successes"])
            if n:
                note = VPP2_NLW_NOTE + ("" if S["complete"] else f"；未完成：{n}/{S['expected_units']}（用户 11-14 要求停 NLw，作者保留已完成单元）")
                A[(T, lv)] = dict(n=n, k=k, steps=steps, note=note)
    return A, check
