"""LingBot-VA vs OpenWAM-alpha, H = 400 (ruling 2026-10-25, 3): paired verdict cells + closed loop + time tables 4a / 4b / 5.

Cells: family A (A0-A4) and family P (P0-P4), t5 and t3, 50 units each, same anchors and seed rule for both models.
  LingBot-VA  verdict = cells_final rules (t5 v3 k = 24; t3 v4 conjunction k = 1 on the same-seed recomputation; units
              failing the recomputation gates excluded, as in cells_final_H400)
  OpenWAM-a   verdict = frozen libero_openwam predicate (t3 v4-recipe conjunction, k = 1); t5 PENDING (predicate gate
              not passed, final 2026-10-24) -> truth and closed-loop columns only
Truth (a) = open-loop replay, reach-anytime within 1..400; closed loop (b) = reference column, success within 400.
Danger rate = false-safe / real failures (model says success, simulator fails). Clopper-Pearson 95 %.
Time (eval_design section 28): real completion = first simulator success step; imagined completion = START of the
qualifying run (onset); closed loop = first run's first success step. Medians [IQR] over successful units; table 5 =
median paired difference imagined - real on units where both succeed. Resolution: LingBot 1 step, OpenWAM 4 steps.

  python compare_models.py   -> out/cells/compare_lingbot_openwam_H400.{json,md}
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import json
import numpy as np
from cells_final import fires, cp, K_T5, K_T3

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/cells"
H = 400
FAMS = (("famA", ("A0", "A1", "A2", "A3", "A4")), ("famP", ("P0", "P1", "P2", "P3", "P4")))


def onset(h, k):
    run = 0
    for i, x in enumerate(h):
        run = run + 1 if x else 0
        if run >= k:
            return i - k + 1
    return None


def lingbot_units(fam, t):
    thr5 = float(json.load(open(f"{ROOT}/libero/out/predicate_v3/t5/calibration.json"))["threshold"])
    v4 = {d: json.load(open(f"{ROOT}/libero/out/predicate_v4_t3/calibration_{d}.json"))["threshold"] for d in ("close", "bowl_out")}
    rescore = {json.loads(l)["unit"]: json.loads(l) for l in open(f"{ROOT}/libero/out/predicate_v4_t3/rescore/rescore_log.jsonl")}
    U, excl = [], []
    for l in open(f"{ROOT}/libero/out/runs/lingbot_{fam}/records_{t}.jsonl"):
        r = json.loads(l); tr = np.load(f"{ROOT}/{r['trajectory_file']}")
        if t == "t5":
            ok = tr["model_pred"][1:H + 1] >= thr5; k = K_T5
        else:
            if not rescore.get(r["unit_id"], {}).get("pass"):
                excl.append(r["unit_id"]); continue
            z = np.load(f"{ROOT}/libero/out/predicate_v4_t3/rescore/{r['unit_id']}.npz")
            ok = ((z["p_close"] >= v4["close"]) & (z["p_bowl"] < v4["bowl_out"]))[1:H + 1]; k = K_T3
        on = onset(ok, k)
        rs = r["real_first_success_step"]; real = rs is not None and rs <= H
        cl = r["closed_loop"]; cls = cl["first_success_step"] if cl["success"] and cl["first_success_step"] <= H else None
        U.append(dict(uid=r["unit_id"], level=r["setting_id"], verdict=on is not None, imag_step=on + 1 if on is not None else None,
                      real=real, real_step=rs if real else None, closed=cls is not None, closed_step=cls))
    return U, excl


def lingbot_base_units(t):
    """main-bank baseline (lingbot_base, run at H = 520 with the v1 predicate): truth and closed loop truncated to 400;
    verdict from the same-seed recomputation scored with the FINAL predicates (rescore_base.py; t5 v3 k = 24, t3 v4
    k = 1). Units that failed the equivalence gates (or are not recomputed yet) are excluded and counted."""
    thr5 = float(json.load(open(f"{ROOT}/libero/out/predicate_v3/t5/calibration.json"))["threshold"])
    v4 = {d: json.load(open(f"{ROOT}/libero/out/predicate_v4_t3/calibration_{d}.json"))["threshold"] for d in ("close", "bowl_out")}
    lp = f"{ROOT}/libero/out/rescore_base/rescore_log.jsonl"
    ok_units = {json.loads(l)["unit"] for l in open(lp) if json.loads(l)["pass"]} if os.path.exists(lp) else set()
    U, excl = [], []
    for l in open(f"{ROOT}/libero/out/runs/lingbot_base/records_{t}.jsonl"):
        r = json.loads(l)
        if r["unit_id"] not in ok_units:
            excl.append(r["unit_id"]); continue
        z = np.load(f"{ROOT}/libero/out/rescore_base/{r['unit_id']}.npz")
        if t == "t5":
            on = onset(z["v3"][1:H + 1] >= thr5, K_T5)
        else:
            on = onset(((z["p_close"] >= v4["close"]) & (z["p_bowl"] < v4["bowl_out"]))[1:H + 1], K_T3)
        rs = r["real_first_success_step"]; real = rs is not None and rs <= H
        cl = r["closed_loop"]; cls = cl["first_success_step"] if cl["success"] and cl["first_success_step"] <= H else None
        U.append(dict(uid=r["unit_id"], level="none", verdict=on is not None, imag_step=on + 1 if on is not None else None,
                      real=real, real_step=rs if real else None, closed=cls is not None, closed_step=cls))
    return U, excl


def openwam_units(fam, t):
    U = []
    for l in open(f"{ROOT}/libero/out/runs/openwam_{fam}/records_{t}.jsonl"):
        r = json.loads(l); cl = r["closed_loop"]
        U.append(dict(uid=r["unit_id"], level=r["setting_id"], verdict=None if r["model_verdict"] is None else bool(r["model_verdict"]),
                      imag_step=r.get("model_first_success_step"), real=bool(r["real_verdict"]), real_step=r["real_first_success_step"],
                      closed=bool(cl["success"]), closed_step=cl["first_success_step"] if cl["success"] else None))
    return U


def frac(x, n):
    return dict(x=x, n=n, rate=round(x / n, 3) if n else None, cp95=cp(x, n) if n else None)


def cell(U):
    n = len(U); fail = [u for u in U if not u["real"]]; succ = [u for u in U if u["real"]]
    c = dict(n=n, real=frac(sum(u["real"] for u in U), n), closed=frac(sum(u["closed"] for u in U), n))
    if any(u["verdict"] is None for u in U):
        c["verdict"] = "pending"
    else:
        c.update(model=frac(sum(u["verdict"] for u in U), n),
                 danger=frac(sum(u["verdict"] for u in fail), len(fail)),
                 false_danger=frac(sum(not u["verdict"] for u in succ), len(succ)))
    return c


def med_iqr(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    q = np.percentile(v, [25, 50, 75])
    return dict(n=len(v), median=float(q[1]), iqr=[float(q[0]), float(q[2])])


def times(U, pending):
    out = dict(real=med_iqr([u["real_step"] for u in U if u["real"]]), closed=med_iqr([u["closed_step"] for u in U if u["closed"]]))
    if not pending:
        d = [u["imag_step"] - u["real_step"] for u in U if u["verdict"] and u["real"]]
        out["imag_minus_real"] = med_iqr(d)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = dict(H=H, cells={}, family={}, excluded={}, units={})
    for model in ("lingbot", "openwam"):
        for t in ("t5", "t3"):
            for fam, levels in FAMS:
                if model == "lingbot":
                    U, ex = lingbot_units(fam, t); res["excluded"][f"lingbot/{t}/{fam}"] = ex
                else:
                    U = openwam_units(fam, t)
                res["units"][f"{model}/{t}/{fam}"] = U
                for L in levels:
                    res["cells"][f"{model}/{t}/{L}"] = cell([u for u in U if u["level"] == L])
                pending = any(u["verdict"] is None for u in U)
                res["family"][f"{model}/{t}/{fam}"] = dict(cell(U), times=times(U, pending),
                                                           times_by_level={L: times([u for u in U if u["level"] == L], pending) for L in levels})
    for model in ("lingbot", "openwam"):                         # main-bank baselines (column "no perturbation")
        for t in ("t5", "t3"):
            if model == "lingbot":
                U, ex = lingbot_base_units(t); res["excluded"][f"lingbot/{t}/base"] = ex
            else:
                U = openwam_units("base", t)
            res["units"][f"{model}/{t}/base"] = U; res["cells"][f"{model}/{t}/none"] = cell(U)
    noise = {}
    for fam, rep in (("famA", "verify_report_openwam_famA_obj.json"), ("famP", "verify_report_openwam_famP_pose.json")):
        v = json.load(open(f"{ROOT}/libero/out/runs/openwam_{fam}/{rep}"))
        noise[fam] = [c for c in v["checks"] if c["id"] == 9][0]["values"]["first_success_step_abs_diff"]
    res["openwam_closed_loop_noise"] = noise
    json.dump(res, open(f"{OUT}/compare_lingbot_openwam_H400.json", "w"), indent=1)
    write_md(res)


def write_md(res):
    C, Fm = res["cells"], res["family"]
    name = {"lingbot": "LingBot-VA", "openwam": "OpenWAM-α"}
    f = lambda d: f"{d['x']}/{d['n']}"
    pct = lambda d: f"{d['x']}/{d['n']}（{100 * d['rate']:.0f}%）" if d["n"] else "—"
    L = ["# LingBot-VA 与 OpenWAM-α 对比，H = 400（2026-10-25）", "",
         "同一组锚、同一种子规则，每格 50 个 unit。危险率 = 真实失败的 unit 中模型判成功的比例（false-safe）。"
         "OpenWAM-α 的 t5 判定列 pending：判据未过生成帧门槛，2026-10-24 定案；真值和闭环列有效。"
         "方括号为 Clopper-Pearson 95% 区间。", "",
         "## 表 1 逐档结果", "",
         "| 任务 | 档 | LingBot 真实成功 | LingBot 闭环 | LingBot 危险率 | OpenWAM 真实成功 | OpenWAM 闭环 | OpenWAM 危险率 |",
         "|---|---|---|---|---|---|---|---|"]
    for t in ("t5", "t3"):
        for fam, levels in FAMS:
            for lv in levels:
                a, b = C[f"lingbot/{t}/{lv}"], C[f"openwam/{t}/{lv}"]
                dz = lambda c: "pending" if c.get("verdict") == "pending" else (pct(c["danger"]) if c["danger"]["n"] else "无真实失败")
                L.append(f"| {t} | {lv} | {f(a['real'])} | {f(a['closed'])} | {dz(a)} | {f(b['real'])} | {f(b['closed'])} | {dz(b)} |")
    L += ["", "## 表 2 按族汇总（危险率带 95% 区间）", "",
          "| 模型 | 任务 | 族 | 真实成功 | 闭环成功 | 模型判成功 | 危险率 | 误判危险（真实成功判失败） |", "|---|---|---|---|---|---|---|---|"]
    for m in ("lingbot", "openwam"):
        for t in ("t5", "t3"):
            for fam, _ in FAMS:
                c = Fm[f"{m}/{t}/{fam}"]
                if c.get("verdict") == "pending":
                    L.append(f"| {name[m]} | {t} | {fam[-1]} | {f(c['real'])} | {f(c['closed'])} | pending | pending | pending |")
                else:
                    L.append(f"| {name[m]} | {t} | {fam[-1]} | {f(c['real'])} | {f(c['closed'])} | {f(c['model'])} | "
                             f"{pct(c['danger'])} {c['danger']['cp95']} | {f(c['false_danger'])} |")
    mi = lambda d: "无成功" if d is None else f"{d['median']:.0f} [{d['iqr'][0]:.0f}, {d['iqr'][1]:.0f}]（n={d['n']}）"
    for tab, key, title in (("4a", "real", "完成步数中位数，开环真值侧"), ("4b", "closed", "完成步数中位数，闭环侧（第一遍）"),
                            ("5", "imag_minus_real", "想象时间乐观度：想象完成步减真实完成步，配对差中位数（起点口径，双方都成功的 unit）")):
        L += ["", f"## 表 {tab} {title}", "", "| 模型 | 任务 | 族 A | 族 P |", "|---|---|---|---|"]
        for m in ("lingbot", "openwam"):
            for t in ("t5", "t3"):
                row = []
                for fam, _ in FAMS:
                    tt = Fm[f"{m}/{t}/{fam}"]["times"]
                    row.append("pending" if key not in tt else mi(tt[key]))
                L.append(f"| {name[m]} | {t} | {row[0]} | {row[1]} |")
    nz = res["openwam_closed_loop_noise"]
    ex = {k: len(v) for k, v in res["excluded"].items() if v}
    L += ["", "## 表注", "",
          "- 格内为中位数 [四分位距]（n = 参与计算的 unit 数），单位为环境步。",
          "- 时间分辨率：LingBot-VA 逐步解码，1 步；OpenWAM-α 视频步长 4，想象完成步为 4 的倍数。",
          "- 表 5 为负，表示想象比真实更早完成。",
          f"- 闭环测量噪声（OpenWAM-α，verify check 9，两遍首次成功步差）：族 A 中位数 {nz['famA']['median']}、最大 {nz['famA']['max']}；"
          f"族 P 中位数 {nz['famP']['median']}、最大 {nz['famP']['max']}（只有 1 个 unit 不为 0，差 25 步）。"
          "LingBot-VA 的闭环每个 unit 只跑了一遍，没有对应的噪声估计。",
          f"- LingBot-VA t3 的判定列沿用同种子重算，未过等价闸门的 unit 排除：{ex}。",
          "- 逐档的时间明细见 JSON（`times_by_level`）。"]
    open(f"{OUT}/compare_lingbot_openwam_H400.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
