"""数据核对 2026-11-15 · 第 4 步：确认 index_audited.html 相对 index_before_audit.html 只改了数据与数据说明。
1) 系统 diff 统计；2) 按 edits.json 逆序还原 index_audited.html，必须逐字节等于 index_before_audit.html（证明除记录的改动外一字未改）；
3) 每处改动所在行与所在区域（DATA / CAT / MNOTES JSON、表 2b/4b 的格、表注 li、缩写注释、核对记录表格）；改动内容不含样式 / 脚本代码。
  python3 diff_check.py > diff_check.txt"""
import collections
import json
import re
import subprocess

import audit_lib as A

B = open(f"{A.HERE}/index_before_audit.html", encoding="utf-8").read()
X = open(f"{A.HERE}/index_audited.html", encoding="utf-8").read()
E = json.load(open(f"{A.HERE}/edits.json", encoding="utf-8"))

d = subprocess.run(["diff", f"{A.HERE}/index_before_audit.html", f"{A.HERE}/index_audited.html"], capture_output=True, text=True).stdout
hunks = [l for l in d.splitlines() if re.match(r"^\d", l)]
print("== 1. diff index_before_audit.html index_audited.html")
print(f"文件行数 {B.count(chr(10))} → {X.count(chr(10))}；字节 {len(B.encode())} → {len(X.encode())}；diff 块 {len(hunks)}：{', '.join(hunks)}")
print(f"（DATA / CAT 各占一整行，所以 diff 按行只能显示整行；下面按逐处改动列出）")

# 2. 逆序还原
s = X; states = []
for ed in reversed(E):
    assert "\n" not in ed["old"] and "\n" not in ed["new"]
    assert s[ed["pos"]:ed["pos"] + len(ed["new"])] == ed["new"], ("还原失败", ed["key"])
    states.append((ed, s[:ed["pos"]].count("\n") + 1))
    s = s[:ed["pos"]] + ed["old"] + s[ed["pos"] + len(ed["new"]):]
print("\n== 2. 按 edits.json 逆序还原后与原副本逐字节相同：", s == B, f"（改动 {len(E)} 处）")
assert s == B


# 3. 区域判定（以原副本中的位置）
def region(pos_line, ed):
    if ed["kind"] in ("DATA", "CAT"):
        return f"嵌入 {ed['kind']} JSON"
    if ed["kind"] in ("w2b", "g2b"):
        return "表 2b 静态表格行（仅 td 文字与沿用 heat_attr 的上色属性）"
    if ed["kind"] in ("w4b", "g4b"):
        return "表 4b 静态表格行（仅 td 文字）"
    return "数据说明：" + ed["key"]


bad = []
for ed, line in reversed(states):
    for frag in (ed["old"], ed["new"]):
        if re.search(r"<(style|script|nav|section|div|h\d|a |link|meta)|function|=>|\{\s*$", frag) and ed["kind"] not in ("DATA", "CAT"):
            bad.append(ed["key"])
    if ed["kind"] in ("w2b", "g2b", "w4b", "g4b"):
        tag_o = re.sub(r">[^<]*<", "><", ed["old"]); tag_n = re.sub(r">[^<]*<", "><", ed["new"])
        tag_o = re.sub(r' style="[^"]*"| class="[^"]*"', "", tag_o); tag_n = re.sub(r' style="[^"]*"| class="[^"]*"', "", tag_n)
        if tag_o != tag_n:
            bad.append(("表格结构变化", ed["key"]))
print("\n== 3. 不属于数据/数据说明的改动：", bad or "无")
print("\n== 4. 逐处改动（行号对两个文件相同：所有改动都不增删换行）")
cnt = collections.Counter()
for i, (ed, line) in enumerate(reversed(states), 1):
    cnt[region(line, ed)] += 1
    print(f"[{i}] 第 {line} 行 · {region(line, ed)} · {ed['key']}")
    print(f"     旧：{ed['old'][:400]}")
    print(f"     新：{ed['new'][:400]}")
print("\n== 5. 按区域计数")
for k, v in cnt.items():
    print(f"  {k}: {v}")
