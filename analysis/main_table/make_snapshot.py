"""从 closed_main 生成带日期的快照：长表 TSV + 逐档 md + 单文件 HTML 热力图。 python make_snapshot.py 20261110"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json, glob, os, math, statistics, collections, csv, sys, re
_wmvp.bootstrap()
import ext_models
D = sys.argv[1]
TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overview_template.html")
UVA_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uva_index.json")
CM = _wmvp.P("$WMV_ROOT/libero/out/closed_main")
os.chdir(CM)
def wilson(k, n):
    p = k/n; z = 1.96; d = 1 + z*z/n; c = p + z*z/(2*n); s = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-s)/d, (c+s)/d)
FAM = lambda l: ("E" if l[:2] in ("Em","Ef") else "N" if l[:2] in ("N1","NR","NL") else
                 "B" if l[:2] in ("Bg","Bf","Bz") or l=="B0" else "base" if l in ("none","base") else l[0])
ORDER = (["none","B0"] + [f"A{i}" for i in range(5)] + [f"P{i}" for i in range(5)] + [f"O{i}" for i in range(1,6)]
         + [f"Em{i}" for i in range(1,5)] + [f"Ef{i}" for i in range(1,5)] + [f"G{i}" for i in range(1,5)] + [f"K{i}" for i in range(1,10)] + ["F60","F160","F260"]
         + ["N1_1"] + [f"NR{i}" for i in range(1,6)] + [f"NL{i}" for i in range(1,6)] + [f"NLw{i}" for i in range(1,6)]
         + [f"B{t}{s}" for t in "gfz" for s in (2,5,9)])
def lvl(r):
    s = r.get("setting_id") or "none"
    return s.split("__")[-1] if "__" in s else s
MN = {"lingbot": "LingBot-VA", "openwam": "OpenWAM-α", "cosmos": "Cosmos Policy", "uva": "UVA", "vpp2": "VPP2"}
OURS, EXTM = ("lingbot", "openwam"), ("cosmos", "uva", "vpp2")
# 2026-11-16 v6：MNOTES 改由源文件计数生成（数据核对 2026-11-15 第五节 2：旧文案仍写 Cosmos 11-11 数据源 / T0 单 seed、VPP2 T4 无逐单元结果）。
# 数一律由 ext_models.*_meta() 从队友源文件算出，不手抄。旧文案见 make_snapshot.py.bak_20261116_pre_v6。
_CM, _VM, _UM = ext_models.cosmos_meta(), ext_models.vpp2_meta(), ext_models.uva_k_meta()
assert _CM["t1_m"] == _CM["t1_cur"], ("T1 不是全部有旧对应，需改说明", _CM)
_FS = _CM["fstep"]
_VPP2_PERT = (f"正式扰动 {_VM['vconds']} 个条件目录尚无完成单元，不用无扰动结果填入" if _VM["vdone"] == 0 else
              f"正式扰动 {_VM['vconds']} 个条件目录已有 {_VM['vdone']} 个完成单元，尚未并入")
assert _UM["t0_cells"] == 9 and _UM["t0_gate"] and _UM["t0_probe_passed"], _UM
MNOTES = {"Cosmos Policy": (f"队友 $COLLAB_COSMOS 提交，第一遍闭环，H=400。T0/T1 为 11-14 当前保留集合（optimism_update_20261114/unit_index.csv，每档 5 起点 × 10 seed）："
                            f"T0 为 11-13 全量重跑，当前 {_CM['t0_cur']} 条中只有 {_CM['t0_m']} 条与旧版同条件、同实际起点、同 seed 对应，{_CM['t0_new']} 条为新组合，"
                            f"与旧版（11-11 批次，非 A/P 族为 50 个官方起点 × 1 seed）的差别是更换样本集合，不是同一轨迹的复现变化；"
                            f"T1 当前 {_CM['t1_cur']} 条均有旧对应，其中 {_CM['t1_fresh']} 条完整重跑、{_CM['t1_rest']} 条为原闭环末窗恢复"
                            f"（只补想象帧，成败与步数取原闭环，不增加闭环样本；含恢复单元的格带斜纹）。"
                            f"T2–T4 取 11-11 闭环审计批次（cosmos_policy/analysis/closed_loop_audit_20261111/unit_outcomes.csv）。"
                            f"F 族取 f_family_20261114（每格 {_CM['fn']} 个独立单元，每单元两遍、取第一遍，不合计为 {2 * _CM['fn']}；实际置黑步 {ext_models.cosmos_f_text(_CM)}）；"
                            f"置黑前已成功结束的单元未触发置黑，这样的格带斜纹，悬停看个数。"),
          "UVA": ("队友 $COLLAB_UVA 提交（uva-validation 逐单元 result.json 第一遍）。五个任务都有。K 族 T1–T4 取自作者 11-13 批次（每格 50 单元 × 2 遍，主表取第一遍，两遍结论全部一致）；"
                  f"T0 的 K 族取自作者 11-14 正式批 K-formal-v3（t0-k-ef-20261115，{_UM['t0_cells']} 档 × {_UM['t0_units'] // _UM['t0_cells']} 单元 × 2 遍 = {_UM['t0_traj']} 条轨迹，主表取第一遍；"
                  f"作者 probe-complete 逐格 passed、逐单元与 result.json 一致、K0 门槛通过，两遍结论 {_UM['t0_agree']}/{_UM['t0_units']} 一致；11-16 并入）。"
                  "T0/T1 取自作者 11-02~04 的成对开/闭环批次（单遍、H=400），不在作者 11-06 后的闭环审计汇总里，成功数已与作者 probe-complete 逐单元核对一致；T0/T1 的 Ef 是作者旧版摩擦实现（T2–T4 后来因它会改动渲染换成 v5/v6 修复版重跑，T0/T1 没重跑），与 T2–T4 的 Ef 口径不同。作者标记 formal_results=false（未正式发布、视觉复核未完成），所有格子带斜纹。T4 的 Ef 族取自作者 11-12 的 v6 修复版闭环（作业 1643–1647，日志 passed），还不在作者审计汇总里，成功数已和作者作业日志逐档核对；T4 的 Bg/Bf/Bz 图像腐蚀 9 档取自作者 11-12 修复批次（t4-repair-20261110，作业 1621–1629，作者独立审计通过，两遍成功数一致；11-14 并入），9 档均为 0/50，与 UVA 在 T0–T3 上 B 族几乎全为 0 的表现一致。其余格子的成功数都和作者自己的审计逐格核对过，一致。"),
          "VPP2": (f"队友 $COLLAB_COSMOS 提交，2026-11-12。目前只有无扰动基线闭环（每任务 5 锚 × 10 seed，第一遍，H=400），其余扰动族显示“无”（{_VPP2_PERT}）；"
                   f"作者跑过的部分 NLw（T0 五档、T1 未跑完）按 11-14 决定不并入；"
                   f"T0–T3 取自 Dropbox 交付的 units.csv（作者核验上传），T4 取 NAS 上 {_VM['v4n']} 个逐单元原始记录（vpp2/results/T4_main_v1/units/*/complete.json，"
                   f"第一遍成功 {_VM['v4k']}/{_VM['v4n']}，与作者汇总一致），步数由这些单元算出。作者标记想象判定待验收。")}
OURS_F = ext_models.ours_f()
EXT = {"cosmos": ext_models.cosmos(), "uva": {**ext_models.uva_t01(), **(ext_models.uva_all(UVA_INDEX, UVA_INDEX.replace(".json", "_with_extra.json")) if os.path.exists(UVA_INDEX) else {}), **ext_models.uva_k()},
       "vpp2": ext_models.vpp2()}  # VPP2 NLw 不并入（用户 2026-11-14 决定；ext_models.vpp2_nlw 保留备查）
# 存档最低要求（protocol.yaml min_artifacts，2026-11-13）：来自 applies_to 源文件的格，n>=50 且 out/artifact_verify/<模型>_*.json
# 里该格（任务|档位）为 pass 才算"完成"；否则 complete = artifacts_unverified。老数据不追溯。
import yaml  # noqa: E402
_MA = yaml.safe_load(open(_wmvp.P("$WMV_ROOT/protocol.yaml"))).get("min_artifacts", {})
_MA_FILES = set(_MA.get("applies_to", []))
_MA_PASS = collections.defaultdict(dict)      # model -> {"T0|F60": status}
for _f in sorted(glob.glob(_wmvp.P("$WMV_ROOT/libero/out/artifact_verify/*.json"))):
    _j = json.load(open(_f))
    _m = os.path.basename(_f).split("_")[0]
    if "partial_test" in _f:
        continue
    for _k, _v in _j.get("cells", {}).items():
        _MA_PASS[_m][_k] = _v["status"]
def _med(fs):
    """中位数显示口径（用户裁决 2026-11-13）：整数照常；落在 .5 的保留 .5，不截断、不四舍五入。"""
    m = statistics.median(fs)
    return int(m) if float(m).is_integer() else round(float(m), 1)


rows, agg_all = [], {}
for d in sorted(x for x in os.listdir(".") if os.path.isdir(x) and x.startswith("T")):
    T = d.split("_")[0]
    for model in OURS + EXTM:
        A = collections.defaultdict(lambda: [0,0,[],""])
        for (t, l), e in EXT.get(model, {}).items():
            if t == T: A[l] = [e["n"], e["k"], list(e["steps"]), e["note"]]
        ma_levels = set()                                   # 需要存档核验的档
        for f in glob.glob(f"{d}/{model}_*.jsonl"):
            ma_src = os.path.basename(f) in _MA_FILES
            for line in open(f):
                r = json.loads(line); cl = r.get("closed_loop") or {}
                if cl.get("success") is None: continue
                a = A[lvl(r)]; a[0] += 1; a[1] += int(bool(cl["success"]))
                if ma_src: ma_levels.add(lvl(r))
                if cl.get("first_success_step"): a[2].append(cl["first_success_step"])
        if model in OURS:                                   # v6：F 族“置黑前已成功”口径说明，与 Cosmos 同一函数（ext_models.f_prior_success）
            for l in ("F60", "F160", "F260"):
                if l in A:
                    info = OURS_F[(model, T, l)]
                    assert info["n"] == A[l][0], (model, T, l, info["n"], A[l][0])
                    A[l][3] = info["note"]
        agg_all[(d, model)] = A
        for l in [x for x in ORDER if x in A] + sorted(set(A) - set(ORDER)):
            n, k, fs, note = A[l]; lo, hi = wilson(k, n)
            rows.append(dict(model=MN[model], task=T, task_internal=d, family=FAM(l), level=l, n=n, n_success=k,
                             success_rate=round(k/n,4), ci95_lo=round(lo,4), ci95_hi=round(hi,4),
                             median_first_success_step=_med(fs) if fs else "",
                             complete=("partial" if n < 50 else "y" if l not in ma_levels or _MA_PASS[model].get(f"{T}|{l}") == "pass"
                                       else "artifacts_unverified"), note=note,
                             exclude_cross=EXT.get(model, {}).get((T, l), {}).get("exclude_cross", "")))   # 非空 = 不进任何跨任务/跨模型汇总
with open(f"主表_长表_{D}.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
# md
L = [f"# 主表 1+2 逐档（闭环成功率 + 完成步数中位）· {D} 快照\n",
     "口径：O 族——LingBot 新任务只含 famO 修正批（closed_occ）；n<50 为在跑中间态；CI = Wilson 95%。\n",
     "“不跑‡”= LingBot-VA 的 B0 不单独跑：B 族空跑核验（10-29，T0/T1 各 2 个锚点）已证明 128 px 路径与 none 逐位相同，B 族以 none 为对照（裁决 2026-11-14：不补跑）；“无”= 该模型没有跑这一档；带 * 的格子有口径说明，见下；带 †排除 的格子保留在逐档表里，但不进任何跨任务 / 跨模型汇总（长表 exclude_cross 列）。\n"] + [f"- **{m}**：{t}\n" for m, t in MNOTES.items()]
for d in sorted(x for x in os.listdir(".") if os.path.isdir(x) and x.startswith("T")):
    L += [f"\n## {d}\n", "| 档位 | " + " | ".join(f"{MN[m]} 成功率 (k/n, 95%CI) | 步数中位" for m in OURS + EXTM) + " |", "|---" * (1 + 2 * len(OURS + EXTM)) + "|"]
    levels = [l for l in ORDER if any(l in agg_all[(d, m)] for m in OURS + EXTM)]
    for l in levels:
        row = [l]
        for m in OURS + EXTM:
            A = agg_all[(d, m)]
            if l in A:
                n, k, fs, note = A[l]; lo, hi = wilson(k, n)
                row += [f"{100*k/n:.0f}% ({k}/{n}, {100*lo:.0f}–{100*hi:.0f}){' *' if note else ''}{' †排除' if EXT.get(m, {}).get((d.split('_')[0], l), {}).get('exclude_cross') else ''}", f"{_med(fs)}" if fs else "—"]
            else:
                row += [("不跑‡" if (m, l) == ("lingbot", "B0") else "待出") if m in OURS else "无", "—"]
        L.append("| " + " | ".join(row) + " |")
open(f"主表_逐档_{D}.md", "w").write("\n".join(L) + "\n")
# 逐档逐单元首次成功步（供 Sheet 表 4b 分组合并中位数用；2026-11-14）
json.dump({f"{MN[m]}|{d.split('_')[0]}|{l}": dict(n=v[0], k=v[1], steps=sorted(v[2])) for (d, m), A in agg_all.items() for l, v in A.items()},
          open(f"主表_逐档步数_{D}.json", "w"), ensure_ascii=False)
# html: reuse previous template, swap data + date
tpl = open(TPL).read()
data = [{k: r[k] for k in ("model","task","level","n","success_rate","ci95_lo","ci95_hi","median_first_success_step","complete","note","exclude_cross")} for r in rows]
models = [MN[m] for m in OURS + EXTM if any(r["model"] == MN[m] for r in rows)]
for k, v in {"__DATA__": json.dumps(data, ensure_ascii=False), "__MODELS__": json.dumps(models, ensure_ascii=False),
             "__EXT__": json.dumps([MN[m] for m in EXTM], ensure_ascii=False), "__MNOTES__": json.dumps(MNOTES, ensure_ascii=False),
             "__DATE__": f"{D[:4]}-{D[4:6]}-{D[6:]}", "__NMODELS__": "二三四五六"[len(models) - 2],
             "__COSMOS_F_GLOSS__": f"，Cosmos Policy {ext_models.cosmos_f_text(_CM, ' / ')}（置黑前已成功结束的单元不触发置黑）"}.items():   # v6：缩写注释 F 行补 Cosmos
    tpl = tpl.replace(k, v)
import subprocess  # 六类汇总（cat_summary.py 读刚写出的长表）
subprocess.run([sys.executable, os.path.join(os.path.dirname(TPL), "cat_summary.py"), D], check=True)
tpl = tpl.replace("__CAT__", open(f"六类汇总_{D}.json", encoding="utf-8").read())
open(f"闭环鲁棒性总览_{D}.html", "w").write(tpl)
print("rows", len(rows), "partial", sum(r["complete"] != "y" for r in rows))
