"""定向补采清单（2026-11-10 决定 1）：摘要对比格 + 每任务每模型闭环成功率<50% 的格 + 无扰动基线格；每格全部 50 单元，扣除已补。"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import csv, json, glob, os, collections, statistics, sys
D = sys.argv[1] if len(sys.argv) > 1 else "20261110"
O = _wmvp.P("$WMV_ROOT/libero/out")
R = list(csv.DictReader(open(f"{O}/closed_main/主表_长表_{D}.tsv"), delimiter="\t"))
sel = collections.OrderedDict()
def add(r, why): sel.setdefault((r["model"], r["task"], r["level"]), dict(row=r, why=[]))["why"].append(why)
for r in R:
    if r["task"] == "T0" and r["level"] in ("NL5", "G4", "Bg2"): add(r, "摘要对比格")
    if float(r["success_rate"]) < 0.5: add(r, "成功率<50%")
    if r["level"] == "none": add(r, "无扰动基线")
TN = {5: "T0", 3: "T1", 2: "T2", 4: "T3", 6: "T4"}
def lv_of(uid): return uid.split("__")[-1] if uid.count("__") >= 2 else "none"
done = collections.Counter()
for f in glob.glob(f"{O}/runs/*/closed_frames_check.*.jsonl") + glob.glob(f"{O}/runs/*/*/closed_frames_check.*.jsonl"):
    for l in open(f):
        u = json.loads(l)["unit_id"]; done[("OpenWAM-α", TN[int(u.split("_t")[1][0])], lv_of(u))] += 1
for tid in (4, 6):
    for d in (f"{O}/runs/openwam_t{tid}/closed", f"{O}/runs/openwam_t{tid}/famB/closed_imagined"):
        for f in (os.listdir(d) if os.path.isdir(d) else []):
            done[("OpenWAM-α", TN[tid], f[:-4].split("__")[-1])] += 1
OW_MB = statistics.mean(os.path.getsize(f) for f in glob.glob(f"{O}/runs/openwam_*/closed_frames/*.npz")[:200]) / 1e6
LB_MB = 2 * statistics.mean(os.path.getsize(f) for f in glob.glob(f"{O}/runs/lingbot_t4/famB/latents/*.npz")[:100]) / 1e6 + 1.0
SEC = {"OpenWAM-α": 29, "LingBot-VA": 88}
tot = collections.defaultdict(lambda: [0, 0, 0, 0.0, 0.0]); lines = []
for (m, T, l), v in sel.items():
    need = 50 - min(50, done.get((m, T, l), 0)); t = tot[m]
    t[0] += 1; t[1] += 50; t[2] += need; t[3] += need * SEC[m] / 3600; t[4] += need * (OW_MB if m.startswith("Open") else LB_MB) / 1000
    lines.append(f"| {m} | {T} | {l} | {float(v['row']['success_rate']):.0%} | {'、'.join(v['why'])} | {need} |")
summ = ["| 模型 | 格数 | 单元总数 | 需补单元 | GPU 小时 | 空间 GB |", "|---|---|---|---|---|---|"]
for m, (c, u, n, h, g) in tot.items(): summ.append(f"| {m} | {c} | {u} | {n} | {h:.0f} | {g:.0f} |")
C, U, N, H, G = [sum(v[i] for v in tot.values()) for i in range(5)]
summ.append(f"| 合计 | {C} | {U} | {N} | {H:.0f}（3 台并行约 {H/3:.0f} 小时） | {G:.0f} |")
out = [f"# 定向补采清单（待确认）· {D}", "", f"由 out/task2/make_target_list.py 从 主表_长表_{D}.tsv 生成；每格补全部 50 单元，已补单元扣除。",
       f"耗时按实测：OpenWAM 29 s/单元、LingBot 88 s/单元（两次闭环）；空间按实测：OpenWAM {OW_MB:.0f} MB/单元、LingBot 约 {LB_MB:.0f} MB/单元。",
       "LingBot B 族 T2 的 9 格尚未跑，不在本清单；跑的时候直接带帧存档。", "", "## 汇总", ""] + summ + \
      ["", "## 逐格", "", "| 模型 | 任务 | 档位 | 闭环成功率 | 入选原因 | 需补单元 |", "|---|---|---|---|---|---|"] + lines
open(f"{O}/closed_main/定向补采清单_{D}.md", "w").write("\n".join(out) + "\n"); print("\n".join(summ)); print(len(lines), "cells")
