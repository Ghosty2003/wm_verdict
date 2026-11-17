"""脚本生成：终局想象乐观率报告（report.md）+ 第三轮给 Jasmine 的说明（round3_imagined/给Jasmine_r3说明.md）。数字全部读文件。"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import collections, csv, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); R3 = _wmvp.P("$WMV_ROOT/libero/out/annotation/round3_imagined")
pct = lambda x: f"{100*x:.0f}%"
cm = json.load(open(f"{HERE}/confusion_openwam_T1.json"))
cov = list(csv.DictReader(open(f"{HERE}/coverage.tsv"), delimiter="\t"))
L = ["# 终局想象乐观率（2026-11-13，脚本生成）", "",
     "口径：只读现有闭环想象帧，不补采、不新建判定器。想象末帧 = 最后一次规划调用的最后一帧。这不是被取消的逐块对账，不算逐块误差。", "",
     "## 1. 自动判定数字（主报告）", "",
     "| 模型 | 任务 | 判定器 | 现实失败单元 | 失败中想象判成功 | 乐观率 | Wilson 95% CI |", "|---|---|---|---|---|---|---|"]
m = cm["末帧(主口径)"]
L.append(f"| OpenWAM-α | T1 | {m['judge']} | {m['n_real_fail']} | {m['real_fail__imag_success']} | {pct(m['optimism_rate'])} | {pct(m['optimism_ci95'][0])}–{pct(m['optimism_ci95'][1])} |")
s = cm["第400步帧(敏感性)"]
L += ["", f"**显著性提醒：n = {m['n_real_fail']}，95% CI {pct(m['optimism_ci95'][0])}–{pct(m['optimism_ci95'][1])}。这一格只能说明\"在极少数失败单元上观察到想象判成功\"，不能当作强结论引用。**", "",
      f"表注：样本量只有 {m['n_real_fail']} 个现实失败单元（OpenWAM T1 的 687 补采子集），CI 很宽。只有这一格同时有部署判定器和闭环存帧："
      "LingBot T0/T1 有判定器但没有闭环想象帧；OpenWAM T0 没有部署判定器；T2–T4 两个模型都没有判定器。"
      f"OpenWAM 第 13 次调用的末帧名义上是第 416 步（闭环只执行到 400）；改用第 400 步那一帧作敏感性对照：{s['real_fail__imag_success']}/{s['n_real_fail']}（{pct(s['optimism_rate'])}，CI {pct(s['optimism_ci95'][0])}–{pct(s['optimism_ci95'][1])}）。", "",
      "## 2. 附录：现实成败 × 想象判定 2×2 混淆矩阵（有判定器且有帧的全部单元，含成功单元）", "",
      "OpenWAM-α · T1 · " + m["judge"] + f" · n = {m['n_units']}（主口径：末帧）", "",
      "| | 想象判成功 | 想象判失败 |", "|---|---|---|",
      f"| 现实成功 | {m['real_success__imag_success']} | {m['real_success__imag_fail']} |",
      f"| 现实失败 | {m['real_fail__imag_success']} | {m['real_fail__imag_fail']} |", "",
      f"现实成功中想象也判成功：{m['real_success__imag_success']}/{m['n_real_success']} = {pct(m['imag_agrees_on_success'])}（CI {pct(m['success_ci95'][0])}–{pct(m['success_ci95'][1])}）。其余 模型×任务 没有判定器，出不了矩阵。", "",
      "## 3. 覆盖面（存了闭环想象帧的单元，按 模型×任务×扰动族）", "", "| 模型 | 任务 | 族 | 单元数 | 现实失败 | 判定器 | 来源 |", "|---|---|---|---|---|---|---|"]
L += [f"| {r['model']} | {r['task']} | {r['family']} | {r['n_units']} | {r['n_real_fail']} | {r['judge']} | {r['sources']} |" for r in cov]
tot = collections.Counter(); [tot.update({(r['model'], r['task']): int(r['n_units'])}) for r in cov]
L += ["", "合计：" + "；".join(f"{m_} {t} {n}" for (m_, t), n in sorted(tot.items())) + f"；共 {sum(tot.values())} 个单元。", "",
      "## 4. 对比案例（想象帧条 vs 真实帧条，figures/）", "",
      "OpenWAM-α：想象把整件事做完了，现实没做完。LingBot-VA：存帧单元里没找到整任务层面的例子（LingBot 每 16 步重读真实画面，想象末帧基本贴着现实），"
      "最接近的是 T2 想象把炉灶点着了、现实没点着（子目标层面的乐观，壶仍在桌上）。候选从像素差排序前 84 个（B 族除外，模型读入的是腐蚀画面）中人工挑选。", ""]
for f in sorted(os.listdir(f"{HERE}/figures")):
    L.append(f"- figures/{f}")
open(f"{HERE}/report.md", "w").write("\n".join(L) + "\n")
M = json.load(open(f"{R3}/manifest_r3.json"))
c = collections.Counter((x["model"], x["task"], x["group"]) for x in M["cards"])
TN = {"t5": "T0", "t3": "T1", "t2": "T2", "t4": "T3", "t6": "T4"}
J = ["# 第三轮标注 r3_imagined 说明（给 Jasmine，2026-11-13，脚本生成）", "",
     f"卡片总数：{len(M['cards'])}（现实失败 {sum(1 for x in M['cards'] if x['group']=='fail_all')}，成功对照 {sum(1 for x in M['cards'] if x['group']=='success_control')}）。抽样种子 {M['seed']}，规则写在 manifest_r3.json 的 rule 字段。", "",
     "每张卡只放想象末帧（模型最后一次规划调用预测的最后一帧，主视角），不放真实画面；问题问\"想象画面是否显示任务已完成\"。参考图：T0/T1 沿用原来的，T2–T4 新做（真实起始帧、真实成功末帧、两张真实失败末帧）。", "",
     "| 模型 | 任务 | 现实失败 | 成功对照 | 合计 |", "|---|---|---|---|---|"]
for (mm, t) in sorted({(k[0], k[1]) for k in c}, key=lambda x: (x[0], TN[x[1]])):
    a, b = c[(mm, t, "fail_all")], c[(mm, t, "success_control")]
    J.append(f"| {mm} | {TN[t]} | {a} | {b} | {a+b} |")
J += ["", "抽样规则（卡片表注与论文表注用同一句）：现实失败单元除 OpenWAM T3/T4 外全收；OpenWAM T3、T4 的现实失败单元各按 numpy RandomState(crc32(\"20261113/('openwam', 'T3')\")，T4 同理) 无放回抽 60 个；成功对照为每个 模型×任务×扰动族 用 RandomState(crc32('20261113/r3/<模型>/<任务>/<族>')) 无放回抽 min(5, n) 个（主种子 20261113）。", "",
      "说明：现实失败那部分是终局乐观率的案例候选集——其余格全收，OpenWAM T3/T4 的失败单元各有八九百个，按固定种子各抽 60 个。B 族（图像腐蚀）单元的想象帧本身是雾 / 模糊样，照常判。", "",
      "## 需要你 push 的提交（wm_verdict_web，按顺序）", "",
      "1. `545e453` Round 2：想象帧卡片换成闭环真实画面卡片（5 个模型）；前端改为先读 .webp",
      "2. `489283c` Round 2：补上 LingBot-VA T2–T4 剩下的 265 张",
      "3. `ca0a7e0` Round 3（r3_imagined）：900 张想象末帧卡片 + T2–T4 参考图 + tools/new_round.py", "",
      "push 之后入库（服务密钥只在你本机）：", "",
      "```", "export SUPABASE_SERVICE_KEY=...",
      "# round 2 换卡（先不加 --yes 看备份和检查结果）",
      _wmvp.P("python3 tools/replace_round.py --round-id 2 --new-name round2_闭环真实画面 --manifest $WMV_ROOT/libero/out/annotation/round2_real/manifest_real.json"),
      "# round 3 新建（先不加 --yes；数据库给的 id 不是 3 会自动撤销并提示改目录名）",
      _wmvp.P("python3 tools/new_round.py --manifest $WMV_ROOT/libero/out/annotation/round3_imagined/manifest_r3.json --expect-id 3"), "```"]
open(f"{R3}/给Jasmine_r3说明.md", "w").write("\n".join(J) + "\n")
print("ok")
