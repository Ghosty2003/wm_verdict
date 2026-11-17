"""生成 diff_list.md（差异清单中文摘要）与 unresolved.md（未处理清单）。文中数字全部由脚本从
verified.json / check_media.json / diff_list.csv / edits.json / readback_result.json / teammate_clues_status.csv / condition_results.json 读出。
  python3 make_reports.py"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import collections
import csv
import json

import audit_lib as A

SA = _wmvp.P("$COSMOS_DATA_ROOT")
V = A.load_verified()
M = json.load(open(f"{A.HERE}/check_media.json", encoding="utf-8"))
D = list(csv.DictReader(open(f"{A.HERE}/diff_list.csv", encoding="utf-8-sig")))
E = json.load(open(f"{A.HERE}/edits.json", encoding="utf-8"))
RB = json.load(open(f"{A.HERE}/readback_result.json", encoding="utf-8"))
TC = list(csv.DictReader(open(f"{A.HERE}/teammate_clues_status.csv", encoding="utf-8-sig")))
CR = json.load(open(f"{SA}/f_upload_20261115/condition_results.json"))
pm = V["pairing"]["match_status"]; aq = V["pairing"]["acquisition"]
cells = V["cells"]

grp = collections.Counter()
for r in D:
    m = r["指标"]
    if m == "说明文字":
        grp[(r["模型"], "说明文字（" + r["位置"].split("（")[0] + "）")] += 1
    elif m == "DATA.note":
        kind = "F 未触发置黑口径" if r["条件"].startswith("F") else "T1 原窗口恢复口径"
        grp[(f"{r['模型']} {r['任务']}", f"DATA.note · {kind}")] += 1
    else:
        grp[(f"{r['模型']} {r['任务']}", m)] += 1
num_rows = [r for r in D if r["指标"] not in ("说明文字", "DATA.note", "CAT.noted")]
ek = collections.Counter(x["kind"] for x in E)
t1_restored_cells = sorted(r["条件"] for r in D if r["指标"] == "DATA.note" and r["任务"] == "T1" and not r["条件"].startswith("F"))
f_note_cells = sorted(f"{r['任务']} {r['条件']}" for r in D if r["指标"] == "DATA.note" and r["条件"].startswith("F"))
tc = collections.Counter((r["source"], r["metric"]) for r in TC)
tc_ok = all(r["teammate_eq_verified"] == "True" and r["page_before_eq_verified"] == "True" for r in TC)
k0 = [k for k, c in cells.items() if c["k"] == 0 and k.startswith("Cosmos Policy|T0")]
v = {T: cells[f"VPP2|{T}|none"] for T in A.TASKS}
v4 = v["T4"]

L = ["# Cosmos Policy / VPP2 网页数据核对 · 差异清单摘要（2026-11-15）", "",
     "被核对网页：results_page/index.html（v5，原样副本 index_before_audit.html）；修正副本 index_audited.html。核验值全部由 verify_sources.py 从队友源文件逐单元重算（只读）。", "",
     "## 一、结论", "",
     f"- 数值差异：**{len(num_rows)} 处**。Cosmos Policy {sum(1 for k in cells if k.startswith('Cosmos'))} 格、VPP2 {sum(1 for k in cells if k.startswith('VPP2'))} 格的 n、成功数/成功率、Wilson 95% CI、步数中位数，"
     f"嵌入 DATA {RB['checked']['A']} 项、六类汇总 CAT {RB['checked']['B']} 项、表 2b/4b 宽表与分组表 {RB['checked']['C']} 格，全部与源文件重算一致。",
     f"- 已确认差异 **{len(D)} 条**（diff_list.csv），全部是“口径说明 / 数据说明”类：网页实际改动 {len(E)} 处（DATA {ek['DATA']}、CAT {ek['CAT']}、说明文字 {ek['text']}），见 diff_check.txt。",
     f"- 读回核对（readback.py）：共比对 {RB['total_checked']} 项，不符 **{RB['mismatches']}** 处。", "",
     "## 二、差异分组（模型 / 任务 / 指标）", "", "| 模型 · 任务 | 指标 | 条数 |", "|---|---|---|"]
L += [f"| {a} | {b} | {n} |" for (a, b), n in sorted(grp.items())]
L += ["", "说明：",
      f"- T1 原窗口恢复口径（{len(t1_restored_cells)} 格：{'、'.join(t1_restored_cells)}）：这些格 50/50 个单元是原闭环末窗恢复，成败与步数取原闭环；网页原来只在 T1 none（49 恢复 + 1 重跑）带了口径说明。"
      "根因是主表 task2/ext_models.py 判断 acquisition 时写的字符串 original_episode_last_window_recovered 与源文件实际值 original_episode_final_window_restored 不一致（见第五节）。",
      f"- F 未触发置黑口径（{len(f_note_cells)} 格：{'、'.join(f_note_cells)}）：第一遍在置黑前已成功结束的单元数写入该格口径说明（规则 7）。",
      f"- CAT.noted：上述口径说明进入所在类后，按 cat_summary.py 同一规则重算“带口径说明”标记（{ek['CAT']} 格），成功数/分母不变。",
      "- 说明文字：页底模型说明（Cosmos 写成 11-11 旧数据源和“T0 50 起点 × 1 seed”；VPP2 写成“T4 没有逐单元结果、没有步数”）、表注/宽表注里的 Sheet 行号旧注、"
      "“无逐单元步数 = 作者只交了汇总（VPP2 T4）”、F 实际置黑步缺 Cosmos、表注数据版本仍写 11-14、核对记录表未注明 11-15 换入的数据不在 11-14 独立重算内。", "",
      "## 三、已确认无差异的大项", "",
      f"- Cosmos T0/T1 当前保留集合：unit_index.csv 6400 条（T0 {aq.get('T0|fresh_full_episode')} 完整重跑；T1 {aq.get('T1|fresh_full_episode')} 完整重跑 + {aq.get('T1|original_episode_final_window_restored')} 原窗口恢复）"
      f"逐条与 canonical_source 原始 complete.json 对照不一致 {V['checks']['t01_raw_vs_index']} 条；all_current.csv 对照不一致 {V['checks']['all_current_vs_raw']} 格。",
      f"- 队友线索 cosmos_T0_T1_differences.csv：成功率 {tc[('cosmos_T0_T1_differences.csv', 'success_rate')]} 条、步数 {tc[('cosmos_T0_T1_differences.csv', 'median_first_success_step')]} 条，"
      f"网页现值{'全部已等于' if tc_ok else '并非全部等于'}当前集合核验值（teammate_clues_status.csv），无需再改；T0 无成功单元的 {len(k0)} 格（{'、'.join(k.split('|')[2] for k in k0)}）步数为空、宽表显示“无成功”，不是 0。",
      f"- T1 baseline = {cells['Cosmos Policy|T1|none']['k']}/{cells['Cosmos Policy|T1|none']['n']}；配对表 T0 同条件同起点同 seed 对应 {pm.get('T0|same_task_condition_anchor_seed')} 条、新组合 {pm.get('T0|no_old_same_condition_anchor_seed')} 条；T1 对应 {pm.get('T1|same_task_condition_anchor_seed')} 条；"
      "原始记录中的 pairing_status 计数与配对表一致；配对键（任务+条件+锚点+policy_seed+环境 seed）无重复、新旧 policy_seed 无不一致。",
      f"- Cosmos F 15 格：原始 complete.json repeats[0]、terminal_manifest.json primary、condition_results.json 三方一致（不一致 {V['checks']['f_three_way']} 项），网页 n/成功数/步数中位与之一致；每格 50 个独立单元，未按两遍计 100。",
      "- Cosmos T2–T4 非 F 档：与 closed_loop_audit_20261111/unit_outcomes.csv 第一遍重算一致。",
      f"- 六类汇总：网页已按 P→环境·一次性、O→信息·一次性 归类，Cosmos/VPP2 各类 n/k/成功率/CI 与同一批次逐档值重新加总一致（未拼接旧分类汇总）。",
      f"- VPP2：T0–T4 无扰动成功率 {'、'.join(A.pct(v[T]['k'], v[T]['n']) for T in A.TASKS)} 与 250 个原始 complete.json 一致；T4 成功 {v4['k']}/{v4['n']}，"
      f"中位数 {A.fmt_med(v4['median'])}，Q1/Q3 = {v4['q1']}/{v4['q3']}（线性分位）。网页 DATA 已是 {A.fmt_med(v4['median'])}；表 4b 显示 [{v4['q1']:.0f}, {v4['q3']:.0f}] 是全表统一的“四分位取整”显示规则（export_wide_all.med 的 :.0f），非差异。"
      f"正式扰动 {V['vpp2_perturb']['conditions']} 个条件目录完成单元 {V['vpp2_perturb'].get('complete_json', 0)} 个，网页继续显示“无”，未用无扰动结果填入。",
      f"- 想象帧：网页没有“Cosmos T4 没有想象帧”的文字，无需修正。定位表 {M['T4_locator']['unique_annotation_id']} 个唯一 annotation_id（现实失败 {M['T4_locator']['real_fail']}、成功 {M['T4_locator']['real_success']}；unit_index 原始 {M['T4_unit_index']['rows']} 行含别名），"
      f"T1 补传 manifest {M['T1_932']['rows']} 条全部属于当前 T1 恢复集合、成败与 unit_index 一致 {M['T1_932']['outcome_matches_unit_index']} 条，回执 verified={M['T1_932']['receipt_verified']}、各文件均已校验={M['T1_932']['receipt_files_all_verified']}、manifest 字节数与回执一致。",
      "- NLw：网页 DATA/CAT/表格中仍不含 NLw，保持不纳入展示。", "",
      "## 四、改动位置", "", "见 diff_check.txt：按 edits.json 逆序还原 index_audited.html 与原副本逐字节相同；改动只落在嵌入 DATA / CAT / MNOTES 的 JSON 字符串、表注 li 文字、缩写注释 F 行文字、核对记录表两个 td 文字；无样式、布局、导航、组件、表格结构、模型顺序变化。", "",
      "## 五、主表本身的问题（只报告，未改主表）", "",
      "1. task2/ext_models.py `cosmos()` 的口径说明判断：`\"original_episode_last_window_recovered\" in acq` 与源文件实际值 `original_episode_final_window_restored` 不一致，导致全部来自原窗口恢复的 T1 格没有口径说明（只有混合的 T1 none 有）。数值不受影响。"
      "影响：closed_main/主表_长表_20261115all.tsv、主表_长表_20261115.tsv 的 note 列；六类汇总_20261115(all).tsv/.json 的 noted；闭环鲁棒性总览_20261115(all).html；results_page/index.html。",
      "2. task2/make_snapshot.py 硬编码的 MNOTES 仍是 11-12/11-14 文案（Cosmos 写 11-11 数据源和 T0 单 seed；VPP2 写 T4 无逐单元结果、无步数）。影响：closed_main/闭环鲁棒性总览_20261115(all).html 以及由它嵌入的 results_page/index.html。",
      "3. Google Sheet 表注（cells/sheet_noNLw_20261115/payload_noNLw.json 已写入并读回）：MODEL_NOTE 仍写“VPP2 T4 只有作者汇总、没有逐单元步数”和“第 28–40 行 Cosmos T0 单遍明细”，表 4b 注仍有“无逐单元步数 = 作者只交了汇总（VPP2 T4）”，F 置黑说明缺 Cosmos。按要求未写 Sheet。",
      "4. results_page/make_page.py 的表注取自 11-14 写入清单（sheet_all_20261114/payload_all.json），所以网页表注数据版本落后于表格本身。**若重新运行 make_page.py，本副本的修正会被覆盖**，需要先修 1–3 的源头。",
      "5. cat_summary.py 在 n=0 的行里 levels_missing 没有把 N1_1 换成 N1（VPP2 物理·一次性）。该字段在 n=0 时网页不显示，纯外观，未改。"]
open(f"{A.HERE}/diff_list.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---------------- unresolved.md
early = {(c["task"], c["condition"]): c for c in CR}
strict = "、".join(f"{c['task']} {c['condition']} {c['strict_repeat_differences']}" for c in CR if c["strict_repeat_differences"])
concl = sum(c["conclusion_differences"] for c in CR)
U = ["# 未处理清单（2026-11-15 Cosmos Policy / VPP2 网页数据核对）", "",
     "以下各项没有现成字段承载、超出“仅数据修正”范围、或需要人工 / numpy / GPU，按规则只列出，不扩展页面结构。", "",
     "1. **存帧覆盖表没有 Cosmos Policy / VPP2 行**（乐观率节“存帧覆盖（模型 × 任务）”只列 LingBot-VA、OpenWAM-α）。加行属于扩展表格内容，未做。可用的已核验计数："
     f"Cosmos T4 已导出终局集合 {M['T4_locator']['unique_annotation_id']} 个唯一 annotation_id（现实失败 {M['T4_locator']['real_fail']}、成功 {M['T4_locator']['real_success']}，是失败定向导出、不是 T4 全部实验数）；"
     f"Cosmos T1 恢复 {M['T1_restored_total']} 条中 {M['T1_932']['rows']} 条 PNG 已补传（回执 verified）；VPP2 无扰动 {sum(v[T]['n'] for T in A.TASKS)} 条第一遍记录（作者保存了规划预测）。覆盖表若要区分“帧已保存 / 已上传 / 网页已接入 / 人工已判”，需要新增列。",
     f"2. **F 族严格复现差异与成败一致性**：condition_results.json 中严格复现差异非 0 的格：{strict}；两遍结论不一致合计 {concl}。网页没有承载这两项的字段，只在 F 格口径说明里写了“两遍取第一遍、未触发置黑数”，这两项未上网页。",
     "3. **想象标签与乐观率**：T4 定位表 human_label 全空（" + "、".join(f"{k} {v}" for k, v in M["T4_locator"]["agent_label"].items()) + "，均为 agent 初判），"
     f"T1 补传 {M['T1_932']['rows']} 条 label_status 全为 agent_only_pending_human；F 15 格 imagined_success 全为 pending_visual_review。uncertain/pending 未推断为成败，网页没有给出 Cosmos/VPP2 经人工确认的乐观率，本次也未新增。",
     "4. **乐观率节“四模型预筛”表的 Cosmos/VPP2 行**：数字来自我们自己的 agent 预筛标签（不在队友核验源内），本次无法用队友文件复核其分母；其中说明“Cosmos T1 = 原执行末窗恢复（只覆盖主表 T1 现实失败的一部分）”描述的是预筛覆盖范围，未与当前 T1 集合（完整重跑 + 原窗口恢复）逐条对账。未改。",
     "5. **新/旧/交集三种分母的对照表**：网页只展示当前保留集合；旧批次（11-11）T0/T1 值、配对交集统计没有字段承载，只在页底模型说明中写明“更换样本集合，不是同一轨迹的复现变化”并保留旧来源名称。完整对照见队友 pairing_audit/Cosmos_pairing_by_condition.csv。",
     "6. **顶部统计条与核对记录总数**（“主表格数 1256 / 独立核对一致 1256/1256”）：是 11-14 主表版本（含 NLw、Cosmos 旧批次）的独立核对记录，涉及全部模型，未改；只在核对记录表 Cosmos/VPP2 行注明 11-15 换入的数据不在该次重算内。本次 readback 已对 Cosmos/VPP2 全部格做了源文件级复核，但网页上没有字段承载这次复核结果。",
     "7. **页首“与 Google Sheet 结果表逐格相同”**：表格数值仍与 Sheet 相同；表注文字本来就取自 11-14 写入清单，与 Sheet 11-15 表注不同，本次修正后仍不同（Sheet 侧也有旧文案，见 diff_list.md 第五节）。未改这句话、未写 Sheet。",
     "8. **Cosmos T2–T4 的原始轨迹层**：本次以 closed_loop_audit_20261111/unit_outcomes.csv（队友 11-11 逐单元审计层）重算，没有再逐个读取 T2–T4 的原始 complete.json，也没有重跑成功判定器。",
     "9. **VPP2 正式扰动**：" + f"{V['vpp2_perturb']['conditions']} 个条件目录完成单元 {V['vpp2_perturb'].get('complete_json', 0)} 个（据队友核对报告，作者 launches.json 记有用户暂停，本次未复查），网页保持“无”；不启动、不补跑。",
     "10. **需要 numpy 的项**：无。分位数按 numpy.percentile 默认线性插值用标准库实现并与 condition_results.json 的 Q1/Q3 一致。",
     "11. **源头修复**：ext_models.py 口径字符串、make_snapshot.py 的 MNOTES、Sheet 表注、make_page.py 表注来源均未改（主表与 Sheet 不在本次修改范围）。在这些源头修好之前，重新生成 index.html 会丢失本副本的修正。"]
open(f"{A.HERE}/unresolved.md", "w", encoding="utf-8").write("\n".join(U) + "\n")
print("ok", len(L), len(U))
