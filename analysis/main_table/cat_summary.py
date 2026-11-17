"""六类扰动汇总（用户 2026-11-12 给定：环境/物理/信息 × 一次性/持续）。
输入：closed_main/主表_长表_<D>.tsv（逐档 n / n_success）。每格 = 该类所有扰动档（不含基线/对照档 none、A0、P0、B0）合并的
闭环成功数 / 单元数，Wilson 95% CI；同时记下该模型在这一类缺哪些档（缺档时与其他模型不完全可比）。
输出：closed_main/六类汇总_<D>.tsv + .json（json 供总览 HTML 嵌入）。
  python3 cat_summary.py 20261112
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import csv
import json
import math
import sys

D = sys.argv[1]
CM = _wmvp.P("$WMV_ROOT/libero/out/closed_main")
CATS = [   # 用户 2026-11-12 给定的六类（环境 / 物理 / 信息 × 一次性 / 持续）
    ("env_once", "环境 · 一次性", "A 物体初始移位、P 机器人初始位姿偏移（t=0 一次性）、K 中途推一下（t∈{60,160,260}）",   # 2026-11-15 按扰动注册表：P 归环境·一次性
     [f"A{i}" for i in range(1, 5)] + [f"P{i}" for i in range(1, 5)] + [f"K{i}" for i in range(1, 10)]),
    ("env_always", "环境 · 持续", "Em 质量 ×1.5/2/4/8、Ef 摩擦 ×0.8~0.2（整段生效）",
     [f"Em{i}" for i in range(1, 5)] + [f"Ef{i}" for i in range(1, 5)]),
    ("phys_once", "物理 · 一次性", "N1 第 50 步动作置零（单步）",
     ["N1_1"]),
    ("phys_always", "物理 · 持续", "G 增益衰减、NR 随机丢指令、NL/NLw 动作延迟",
     [f"G{i}" for i in range(1, 5)] + [f"NR{i}" for i in range(1, 6)]
     + [f"NL{i}" for i in range(1, 6)] + [f"NLw{i}" for i in range(1, 6)]),
    ("info_once", "信息 · 一次性", "O 黑方块遮挡（只遮模型读入的第一帧）、F 单次全帧置黑（名义 t∈{60,160,260} 之后第一次规划调用读入的主视角帧）",   # 2026-11-15 按扰动注册表：O 归信息·一次性
     [f"O{i}" for i in range(1, 6)] + ["F60", "F160", "F260"]),
    ("info_always", "信息 · 持续", "B 族图像腐蚀 Bg/Bf/Bz（每一帧）",
     [f"B{t}{s}" for t in "gfz" for s in (2, 5, 9)]),
]
BASE = ("none",)
import os as _os
if _os.environ.get("WMV_NO_NLW"):   # 2026-11-15 用户：“去掉所有的 NLw”——物理·持续 不含 NLw 档
    CATS = [(k, n, d.replace("NL/NLw 动作延迟", "NL 动作延迟"), [l for l in ls if not l.startswith("NLw")]) for k, n, d, ls in CATS]


def wilson(k, n):
    p = k / n; z = 1.96; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


rows = list(csv.DictReader(open(f"{CM}/主表_长表_{D}.tsv", encoding="utf-8"), delimiter="\t"))
cell = {(r["model"], r["task"], r["level"]): r for r in rows}
models = list(dict.fromkeys(r["model"] for r in rows))
tasks = sorted({r["task"] for r in rows})
out = []
for T in tasks:
    for m in models:
        have = {l for (mm, tt, l) in cell if mm == m and tt == T}
        if not have:
            continue
        entries = [("base", "无扰动基线", "none", list(BASE))] + [(c[0], c[1], c[2], c[3]) for c in CATS]
        for key, name, desc, levels in entries:
            excl = [l for l in levels if l in have and cell[(m, T, l)].get("exclude_cross")]   # 裁决排除的格（如 UVA T0/T1 Ef）不进合并
            got = [l for l in levels if l in have and l not in excl]
            miss = [l for l in levels if l not in have]
            n = sum(int(cell[(m, T, l)]["n"]) for l in got)
            k = sum(int(cell[(m, T, l)]["n_success"]) for l in got)
            partial = [l for l in got if cell[(m, T, l)]["complete"] != "y"]
            noted = any(cell[(m, T, l)].get("note") for l in got)
            if n:
                lo, hi = wilson(k, n)
                out.append(dict(model=m, task=T, category=key, category_name=name, n=n, n_success=k,
                                success_rate=round(k / n, 4), ci95_lo=round(lo, 4), ci95_hi=round(hi, 4),
                                levels_used=" ".join(l.replace("N1_1", "N1") for l in got),
                                levels_missing=" ".join(l.replace("N1_1", "N1") for l in miss),
                                levels_partial=" ".join(partial), noted="y" if noted else "",
                                levels_excluded=" ".join(excl)))
            else:
                out.append(dict(model=m, task=T, category=key, category_name=name, n=0, n_success=0, success_rate="",
                                ci95_lo="", ci95_hi="", levels_used="", levels_missing=" ".join(miss), levels_partial="", noted="",
                                levels_excluded=" ".join(excl)))
with open(f"{CM}/六类汇总_{D}.tsv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter="\t"); w.writeheader(); w.writerows(out)
json.dump(dict(cats=[dict(key="base", name="无扰动基线", desc="none 档（主锚库，不加扰动）")] +
               [dict(key=c[0], name=c[1], desc=c[2]) for c in CATS], rows=out),
          open(f"{CM}/六类汇总_{D}.json", "w", encoding="utf-8"), ensure_ascii=False)
print("rows", len(out), "empty", sum(1 for r in out if not r["n"]))
