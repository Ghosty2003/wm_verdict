"""需同步的数据说明（旧文字 → 新文字）。新文字中的数一律从 verified.json 取，不手抄。"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json

MATCHED = "same_task_condition_anchor_seed"
RESTORED = "original_episode_final_window_restored"
FRESH = "fresh_full_episode"


def _num(V):
    pm = V["pairing"]["match_status"]; aq = V["pairing"]["acquisition"]
    t0_cur = sum(v for k, v in pm.items() if k.startswith("T0|")); t1_cur = sum(v for k, v in pm.items() if k.startswith("T1|"))
    t0_m = pm.get(f"T0|{MATCHED}", 0); t1_m = pm.get(f"T1|{MATCHED}", 0)
    c = V["cells"]; fi = V["f_info"]
    early = {}
    for k, x in c.items():
        m, T, l = k.split("|")
        if m == "Cosmos Policy" and l in ("F60", "F160", "F260"):
            early[(T, l)] = x["early"]
    steps = {l: fi[f"T0|{l}"]["actual_step"] for l in ("F60", "F160", "F260")}
    assert all(fi[f"{T}|{l}"]["actual_step"] == steps[l] for T in ("T0", "T1", "T2", "T3", "T4") for l in steps)
    v4 = c["VPP2|T4|none"]
    ver = json.load(open(_wmvp.P("$WMV_ROOT/libero/out/verify_20261114/verify_independent_result.json")))["rows"]
    n_ver_cos = sum(1 for r in ver if r["model"] == "Cosmos Policy")
    n_f = sum(1 for k in c if k.startswith("Cosmos Policy|") and k.split("|")[2] in ("F60", "F160", "F260"))
    return dict(t0_cur=t0_cur, t0_m=t0_m, t0_new=t0_cur - t0_m, t1_cur=t1_cur, t1_m=t1_m,
                t1_fresh=aq.get(f"T1|{FRESH}", 0), t1_rest=aq.get(f"T1|{RESTORED}", 0), early=early, fstep=steps,
                fn=c["Cosmos Policy|T0|F60"]["n"], v4n=v4["n"], v4k=v4["k"], vconds=V["vpp2_perturb"]["conditions"],
                vdone=V["vpp2_perturb"].get("complete_json", 0), n_ver_cos=n_ver_cos, n_f=n_f)


def _early_text(x):
    parts = []
    for l in ("F60", "F160", "F260"):
        e = [(T, x["early"][(T, l)]) for T in ("T0", "T1", "T2", "T3", "T4")]
        if any(v for _, v in e):
            parts.append(f"{l}：" + "、".join(f"{T} {v}" for T, v in e))
    return "；".join(parts)


def edits(V, s):
    x = _num(V); fs = x["fstep"]
    assert x["t1_m"] == x["t1_cur"] and x["vdone"] == 0
    cos_mnote = (f"队友 $COLLAB_COSMOS 提交，第一遍闭环，H=400。T0/T1 为 11-14 当前保留集合（optimism_update_20261114/unit_index.csv，每档 5 起点 × 10 seed）："
                 f"T0 为 11-13 全量重跑，当前 {x['t0_cur']} 条中只有 {x['t0_m']} 条与旧版同条件、同实际起点、同 seed 对应，{x['t0_new']} 条为新组合，"
                 f"与旧版（11-11 批次，非 A/P 族为 50 个官方起点 × 1 seed）的差别是更换样本集合，不是同一轨迹的复现变化；"
                 f"T1 当前 {x['t1_cur']} 条均有旧对应，其中 {x['t1_fresh']} 条完整重跑、{x['t1_rest']} 条为原闭环末窗恢复"
                 f"（只补想象帧，成败与步数取原闭环，不增加闭环样本；含恢复单元的格带斜纹）。"
                 f"T2–T4 取 11-11 闭环审计批次（cosmos_policy/analysis/closed_loop_audit_20261111/unit_outcomes.csv）。"
                 f"F 族取 f_family_20261114（每格 {x['fn']} 个独立单元，每单元两遍、取第一遍，不合计为 {2 * x['fn']}；实际置黑步 {fs['F60']}/{fs['F160']}/{fs['F260']}）；"
                 f"置黑前已成功结束的单元未触发置黑，这样的格带斜纹，悬停看个数。")
    vpp_mnote = (f"队友 $COLLAB_COSMOS 提交，2026-11-12。目前只有无扰动基线闭环（每任务 5 锚 × 10 seed，第一遍，H=400），其余扰动族显示“无”"
                 f"（正式扰动 {x['vconds']} 个条件目录尚无完成单元，不用无扰动结果填入）；T0–T3 取自 Dropbox 交付的 units.csv（作者核验上传），"
                 f"T4 取 NAS 上 {x['v4n']} 个逐单元原始记录（vpp2/results/T4_main_v1/units/*/complete.json，第一遍成功 {x['v4k']}/{x['v4n']}，与作者汇总一致），"
                 f"步数由这些单元算出。作者标记想象判定待验收。")
    MN = json.JSONDecoder().raw_decode(s, s.index("const MNOTES=") + len("const MNOTES="))[0]
    model_note = (f"Cosmos Policy（$COLLAB_COSMOS）T0/T1 为当前保留集合（每档 5 起点 × 10 seed；T1 部分单元为原闭环末窗恢复，成败与步数取原闭环，不增加闭环样本），"
                  f"T2–T4 为 11-11 闭环审计批次，含 F 族（每格 {x['fn']} 个独立单元，取第一遍）；VPP2（$COLLAB_COSMOS）只有无扰动基线，T4 步数由 {x['v4n']} 个逐单元原始记录算出，其余族显示“无”")
    f_add = (f"OpenWAM-α 64/160/288、Cosmos Policy {fs['F60']}/{fs['F160']}/{fs['F260']}；Cosmos Policy 置黑前已成功结束、未触发置黑的单元数（每格 {x['fn']}）"
             f"{_early_text(x)}）")
    E = [
        dict(where="页底模型说明 MNOTES[Cosmos Policy]", model="Cosmos Policy", old=json.dumps(MN["Cosmos Policy"], ensure_ascii=False)[1:-1],
             new=json.dumps(cos_mnote, ensure_ascii=False)[1:-1],
             source="optimism_update_20261114/unit_index.csv；pairing_audit/Cosmos_T0_T1_new_to_old_6400.csv；f_upload_20261115/condition_results.json",
             batch="T0/T1 当前保留集合 + T2–T4 11-11 批次 + F 11-14", denom=f"T0 {x['t0_cur']} / T1 {x['t1_cur']} 条",
             reason="旧说明写数据源为 11-11 unit_outcomes.csv、T0 为 50 起点 × 1 seed，与当前 T0/T1 数据来源不符；补充 T0 新组合、T1 恢复口径与 F 未触发说明（规则 2/4/7）"),
        dict(where="页底模型说明 MNOTES[VPP2]", model="VPP2", old=json.dumps(MN["VPP2"], ensure_ascii=False)[1:-1],
             new=json.dumps(vpp_mnote, ensure_ascii=False)[1:-1],
             source="vpp2/results/T4_main_v1/units/*/complete.json；vpp2/results/perturbations_20261114/", batch="VPP2 无扰动主批次",
             denom=f"{x['v4n']}", reason="旧说明称 T4 只有作者汇总、没有逐单元结果和步数；实际 50 个逐单元原始记录存在且网页已显示步数（规则 5）"),
        dict(where="表 2b 宽表注", model="Cosmos Policy / VPP2", old="本页第 28–40 行的 Cosmos T0 单遍明细为队友原样保留", new=model_note,
             source="同上", reason="旧注指 Sheet 行号并称 Cosmos T0 为单遍明细，与当前 T0（5 起点 × 10 seed 全量重跑）不符"),
        dict(where="表 4b 宽表注", model="Cosmos Policy / VPP2", old="本页第 29–41 行的 Cosmos T0 单遍明细为队友原样保留", new=model_note,
             source="同上", reason="同上"),
        dict(where="表 2b / 4b 表注", model="Cosmos Policy / VPP2", old="第 28–40 行（表 4b 为第 29–41 行）的 Cosmos T0 单遍明细为队友原样保留", new=model_note,
             source="同上", reason="同上"),
        dict(where="表 4b 表注", model="VPP2", task="T4", old="无逐单元步数 = 作者只交了汇总（VPP2 T4）。", new="",
             source="vpp2/results/T4_main_v1/units/*/complete.json", reason="VPP2 T4 已有逐单元步数，表中已无“无逐单元步数”格，旧图例失效"),
        dict(where="表 2b/4b 表注与宽表注（F 置黑说明）", model="Cosmos Policy", cond="F60/F160/F260", old="OpenWAM-α 64/160/288）", new=f_add,
             source="f_upload_20261115/condition_results.json；f_family_20261114/results/*/units/*/complete.json", batch="F 族 11-14",
             denom=f"每格 {x['fn']}", reason="Cosmos F 已入表但未列其实际置黑步与未触发置黑数（规则 7）"),
        dict(where="缩写注释 F 行", model="Cosmos Policy", cond="F60/F160/F260", old="OpenWAM-α 64 / 160 / 288",
             new=f"OpenWAM-α 64 / 160 / 288，Cosmos Policy {fs['F60']} / {fs['F160']} / {fs['F260']}（置黑前已成功结束的单元不触发置黑）",
             source="f_upload_20261115/condition_results.json（actual_trigger_grid_step）", reason="同上"),
        dict(where="表 2b / 4b 表注（数据版本）", model="Cosmos Policy / VPP2",
             old="（closed_main/主表_长表_20261114.tsv + 主表_逐档步数_20261114.json）生成，不手抄，写入后逐格读回核对（2026-11-14）。",
             new="（closed_main/主表_长表_20261115all.tsv + 主表_逐档步数_20261115all.json）生成，不含 NLw（2026-11-15 用户决定），不手抄，写入后逐格读回核对（2026-11-15）。",
             source="results_page/make_page.py（表格取 cells/sheet_all_20261115all）；cells/sheet_noNLw_20261115/payload_noNLw.json 同句",
             reason="表格实际由 11-15 主表生成（Cosmos T0/T1 当前集合、F 族、VPP2 T4 步数在 11-15 换入），旧版本说明写 11-14"),
        dict(where="表 2b / 4b 宽表注（数据版本）", model="Cosmos Policy / VPP2",
             old="本宽表由 libero/export_sheet_all_20261114.py 从闭环主表生成（2026-11-14），不手抄",
             new="本宽表由 libero/export_sheet_all_20261114.py 从闭环主表生成（2026-11-15，不含 NLw），不手抄",
             source="同上", reason="同上"),
        dict(where="核对记录表 Cosmos Policy 行“独立来源”", model="Cosmos Policy",
             old="<td>作者 condition_metrics.csv</td>",
             new=f"<td>作者 condition_metrics.csv（11-14 主表版本 {x['n_ver_cos']} 格；11-15 换入的 T0/T1 当前保留集合与 F 族 {x['n_f']} 格不在该次独立重算内）</td>",
             source="verify_20261114/verify_independent_result.json", reason="核对记录是 11-14 主表版本，网页 Cosmos 数据已换为 11-15 版本，需注明覆盖范围（规则 2）"),
        dict(where="核对记录表 VPP2 行“独立来源”", model="VPP2",
             old="<td>作者 task_summary.json / T4_summary.json</td>",
             new=f"<td>作者 task_summary.json / T4_summary.json（11-14 主表版本；11-15 起 T4 步数取 {x['v4n']} 个逐单元 complete.json，不在该次独立重算内）</td>",
             source="verify_20261114/verify_independent_result.json", reason="同上"),
    ]
    return E
