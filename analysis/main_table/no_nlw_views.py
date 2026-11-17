"""“去掉所有的 NLw”视图（用户 2026-11-15）。底层数据不动，生成 20261115 版展示文件：
  1) closed_main/主表_长表_20261115.tsv = 20261114 主表去掉 NLw1–5 行（其余逐字相同）；
  2) WMV_NO_NLW=1 python3 cat_summary.py 20261115 → 六类汇总_20261115（物理·持续 不含 NLw，按主表各档重新加总）；
  3) closed_main/闭环鲁棒性总览_20261115.html = 20261114 总览页换入去 NLw 的逐档数据与六类汇总，删去 NLw 的注释行与相关说明。
  python3 no_nlw_views.py [D]     （D 默认 20261115；2026-11-16 v6 起按日期参数化：读 主表_长表_<D>all.tsv / 闭环鲁棒性总览_<D>all.html，写 <D> 版）"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import csv, json, os, re, subprocess, sys
D = sys.argv[1] if len(sys.argv) > 1 else "20261115"
CM = _wmvp.P("$WMV_ROOT/libero/out/closed_main")
HERE = os.path.dirname(os.path.abspath(__file__))
src = list(csv.DictReader(open(f"{CM}/主表_长表_{D}all.tsv", encoding="utf-8"), delimiter="\t"))
keep = [r for r in src if not r["level"].startswith("NLw")]
with open(f"{CM}/主表_长表_{D}.tsv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(src[0]), delimiter="\t"); w.writeheader(); w.writerows(keep)
subprocess.run(["python3", f"{HERE}/cat_summary.py", D], check=True, env=dict(os.environ, WMV_NO_NLW="1"))
h = open(f"{CM}/闭环鲁棒性总览_{D}all.html", encoding="utf-8").read()
m = re.search(r"const DATA=(\[.*?\]);\n", h, re.S); data = json.loads(m.group(1))
h = h[:m.start(1)] + json.dumps([r for r in data if not r["level"].startswith("NLw")], ensure_ascii=False) + h[m.end(1):]
m = re.search(r"const CAT=(\{.*?\});\n", h, re.S)
h = h[:m.start(1)] + open(f"{CM}/六类汇总_{D}.json", encoding="utf-8").read() + h[m.end(1):]
reps = [("· 2026-11-14</title>", "· 2026-11-15（不含 NLw）</title>"),
        ("（2026-11-14 快照）", "（2026-11-14 快照，2026-11-15 起不含 NLw 长延迟族）"),
        ("物理 · 持续 = P、G、NR、NL、NLw；", "物理 · 持续 = P、G、NR、NL；"), ("物理 · 持续 = G、NR、NL、NLw；", "物理 · 持续 = G、NR、NL；"),
        ("没有 NLw 族。", ""), ("作者跑过的部分 NLw（T0 五档、T1 未跑完）按 11-14 决定不并入；", ""), ("作者跑过的部分 NLw（T0 五档、T1 未跑完）按 11-14 决定不并入。", ""),
        (",\n [\"NLw\",[\"NLw1\",\"NLw2\",\"NLw3\",\"NLw4\",\"NLw5\"]]", ""),
        ("N1/NR/NL/NLw 动作噪声丢帧延迟", "N1/NR/NL 动作噪声丢帧延迟")]
for a, b in reps:
    h = h.replace(a, b)
h = re.sub(r"<tr><td>NLw</td>.*?</tr>\n", "", h, flags=re.S)
h = h.replace("（LingBot-VA 每 16 步、OpenWAM-α 每 32 步重规划一次）", "（LingBot-VA 每 16 步、OpenWAM-α 每 32 步重规划一次）") 
open(f"{CM}/闭环鲁棒性总览_{D}.html", "w", encoding="utf-8").write(h)
left = [x for x in re.findall(r".{20}NLw.{20}", h)]
print("主表行", len(src), "→", len(keep), "；总览页残留 NLw 字样", len(left), left[:6])
