"""同卷对比标注 · 第 2 步（v4，2026-11-16）：每个 任务×扰动档 出至多 5 页（第 1 组 … 第 5 组），每页四行（LingBot / OpenWAM / Cosmos / UVA）。
Jasmine：“反正我们的都不一样了，那就把原本的第一个 seed 第二个 seed 换成第一组第二组，也不用标是哪个 seed 了，都用齐全的数据”。
  → 不再要求四个模型同 seed；每个模型在每格里各自挑 5 个单元，放进第 1–5 组。页面、页面 id、卡片问题里都不出现 seed。
读 inventory.jsonl（inventory.py；complete = 真实画面与想象帧都有且同一次运行）与 v3 页面表
（wm_verdict_web/backups/samecase_pages.json.bak_20261116_pre_v4，即提交 809b862 / cd46a6b 上线的那一版）。只读，纯 python3。

每格每模型的挑法（确定性，不用随机数）：
  1) 保留：v3 第 g 页这一行用的单元，如果在 v4 里“齐全”，原样放在第 g 组（库里旧卡和它们的标注原位不动）。
  2) 补齐全：其余空位按组号从小到大，用“齐全”的单元补，排序键
       (库里已有卡数降序（round 2 + round 3），是否需要 CPU 解码 LingBot latent（不需要优先），真实画面是否还要回放（不需要优先），
        crc32("20261116/groups/<T>/<level>/<model>/<unit_id>"))；
  3) 补只有真实画面的：仍有空位时，先放 v3 第 g 页这一行原来的单元（有真实画面即可，保住旧卡位置），
       再按 (库里已有 round 2 卡优先, 不需回放优先, crc32(同上)) 补。这些位置的想象区显示“无”。
  同一 case（同一初始条件 + 同一策略种子；集群版与 H100 版、原执行与新重跑算同一个 case）在一格一模型里只用一次。
  只放在该格有真实画面的单元；想象帧只在“齐全”时放（真实画面与想象帧不同运行的，想象区记“无”，不混配）。
  原选择规则（v1–v3 select_pages.py）没有成功 / 失败分层，v4 也不加。NLw 档照旧不要（2026-11-15 用户）。
  一格的组数 = 四个模型里挑到的最多单元数（≤ 5）。
UVA 留白（2026-11-16 用户：“那些 uva 需要重新跑的 uva 的数据在标注网页上先留白，其他的都加上去”；
  范围按 out/teammate_prompts/uva_correction_rerun_20261116.md）：O 族 T0–T4 × O1–O5（UVA 现有为第 50 步遮挡，口径不同）、
  B 族 T0–T4 × Bg/Bf/Bz × 2/5/9（接入核查未完成；B0 照常放）、K 族 T1 K9 / T2 K9 / T3 K8 / T3 K9 / T4 K9、F 族（本来没跑）。
  这些格子的 UVA 行不放任何卡，行内只显示 “UVA: pending rerun”（页面表 rows[].note）；v3 落在这些格子上的 UVA 卡从页面移除（卡与库不动）。
输出 pages.json（出图用，带来源路径）、sample_manifest.json（逐格逐模型挑选记录）。  python3 select_pages.py
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import collections
import json
import os
import re
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = _wmvp.P("$WMV_WEB_ROOT")
V3_PAGES = f"{WEB}/backups/samecase_pages.json.bak_20261116_pre_v4"
TAG, K = "20261116/groups", 5
MODELS = ["lingbot", "openwam", "cosmos", "uva"]
LORDER = ["none", "B0", "A0", "A1", "A2", "A3", "A4", "P0", "P1", "P2", "P3", "P4", "O1", "O2", "O3", "O4", "O5",
          "Em1", "Em2", "Em3", "Em4", "Ef1", "Ef2", "Ef3", "Ef4", "G1", "G2", "G3", "G4", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9",
          "F60", "F160", "F260", "N1_1", "NR1", "NR2", "NR3", "NR4", "NR5", "NL1", "NL2", "NL3", "NL4", "NL5",
          "NLw1", "NLw2", "NLw3", "NLw4", "NLw5", "Bg2", "Bg5", "Bg9", "Bf2", "Bf5", "Bf9", "Bz2", "Bz5", "Bz9"]


UVA_BLANK_NOTE = "UVA: pending rerun"


def uva_blank(T, lv):
    return (lv[:1] == "O" or lv[:2] in ("Bg", "Bf", "Bz") or lv[:1] == "F"
            or (T, lv) in {("T1", "K9"), ("T2", "K9"), ("T3", "K8"), ("T3", "K9"), ("T4", "K9")})


def h(T, lv, m, uid):
    return zlib.crc32(f"{TAG}/{T}/{lv}/{m}/{uid}".encode())


def main():
    inv = [json.loads(l) for l in open(f"{HERE}/inventory.jsonl")]
    by = collections.defaultdict(dict)                                  # (T, lv, model) -> uid -> unit（只要有真实画面的）
    present = collections.defaultdict(set)
    for u in inv:
        if u["level"].startswith("NLw"):
            continue
        present[(u["T"], u["level"])].add(u["model"])
        if u["real"] != "none":
            by[(u["T"], u["level"], u["model"])][u["unit_id"]] = u
    old = collections.defaultdict(dict)                                 # (T, lv, model) -> g -> v3 那一行的 unit_id
    for p in json.load(open(V3_PAGES))["pages"]:
        for r in p["rows"]:
            uid = r.get("real") or r.get("imag")
            if uid:
                old[(p["T"], p["level"], r["model"])][p["seed_no"]] = uid
    pages, log = [], []
    keys = sorted(present, key=lambda k: (k[0], LORDER.index(k[1]) if k[1] in LORDER else 99, k[1]))
    for T, lv in keys:
        pick = {}
        for m in MODELS:
            U = by.get((T, lv, m), {}); O = old.get((T, lv, m), {})
            if m == "uva" and uva_blank(T, lv):                       # 留白：不放任何 UVA 单元
                U, O = {}, {}
            slots, how, used = [None] * K, [None] * K, set()
            for g in range(1, K + 1):                                  # 1) 原位保留齐全的旧单元
                u = U.get(O.get(g))
                if u and u["complete"] and u["case"] not in used:
                    slots[g - 1], how[g - 1] = u, "kept_complete"; used.add(u["case"])
            comp = sorted((u for u in U.values() if u["complete"]),
                          key=lambda u: (-(bool(u["r2_card"]) + bool(u["r3_card"])), u["imag"] == "latent", u["real"] == "replay", h(T, lv, m, u["unit_id"])))
            for g in range(K):                                          # 2) 补齐全
                if slots[g] is None:
                    nxt = next((u for u in comp if u["case"] not in used), None)
                    if nxt:
                        slots[g], how[g] = nxt, "complete"; used.add(nxt["case"])
            for g in range(K):                                          # 3a) 只有真实画面：先保旧位置
                u = U.get(O.get(g + 1))
                if slots[g] is None and u and u["case"] not in used:
                    slots[g], how[g] = u, "kept_real_only"; used.add(u["case"])
            rest = sorted(U.values(), key=lambda u: (not u["r2_card"], u["real"] == "replay", h(T, lv, m, u["unit_id"])))
            for g in range(K):                                          # 3b) 只有真实画面：其余
                if slots[g] is None:
                    nxt = next((u for u in rest if u["case"] not in used), None)
                    if nxt:
                        slots[g], how[g] = nxt, "real_only"; used.add(nxt["case"])
            pick[m] = (slots, how)
            log.append(dict(T=T, level=lv, model=m, n_units=len(U), n_complete=sum(u["complete"] for u in U.values()),
                            chosen=[u and u["unit_id"] for u in slots], how=how,
                            complete=[bool(u and u["complete"]) for u in slots], success=[u and u["success"] for u in slots],
                            machine=[u and u["machine"] for u in slots], v3=[O.get(g) for g in range(1, K + 1)]))
        n = max((max((i + 1 for i, x in enumerate(s) if x is not None), default=0) for s, _ in pick.values()), default=0)
        for g in range(1, n + 1):
            rows = []
            for m in MODELS:
                u, hw = pick[m][0][g - 1], pick[m][1][g - 1]
                rows.append(dict(model=m, in_cell=m in present[(T, lv)] or (m == "uva" and uva_blank(T, lv)), how=hw,
                                 note=UVA_BLANK_NOTE if m == "uva" and uva_blank(T, lv) else None,
                                 real=None if not u else dict(unit_id=u["unit_id"], src=u["real"], path=u["real_path"], success=u["success"],
                                                              first_success_step=u["first_success_step"], execution=u["execution"],
                                                              r2_card=u["r2_card"], run=u["real_run"], machine=u["machine"]),
                                 imag=None if not u or not u["complete"] else dict(unit_id=u["unit_id"], src=u["imag"], path=u["imag_path"],
                                                                                    key=u.get("imag_key"), r3_card=u["r3_card"], execution=u["execution"],
                                                                                    run=u["imag_run"], machine=u["machine"])))
            pages.append(dict(page_id=f"{T}|{lv}|g{g}", T=T, level=lv, group=g, rows=rows))
    json.dump(dict(rule=__doc__, tag=TAG, k=K, pages=pages), open(f"{HERE}/pages.json", "w"), ensure_ascii=False, indent=0)
    json.dump(log, open(f"{HERE}/sample_manifest.json", "w"), ensure_ascii=False, indent=0)
    # ---- 汇总
    st = collections.Counter()
    for p in pages:
        for r in p["rows"]:
            st[(r["model"], p["T"], "complete" if r["imag"] else "real_only" if r["real"] else "missing")] += 1
    print("cells", len(keys), "pages", len(pages), "cells<5", sum(1 for k in keys if sum(p["T"] == k[0] and p["level"] == k[1] for p in pages) < K))
    for m in MODELS:
        for T in ("T0", "T1", "T2", "T3", "T4"):
            print(m, T, {k: st[(m, T, k)] for k in ("complete", "real_only", "missing")})
    print("how", collections.Counter(h_ for x in log for h_ in x["how"]))
    bl = collections.Counter(re.match(r"[A-Za-z]+", p["level"]).group(0) for p in pages for r in p["rows"] if r.get("note"))
    print("uva blank pages by family", dict(bl), "total", sum(bl.values()))


if __name__ == "__main__":
    main()
