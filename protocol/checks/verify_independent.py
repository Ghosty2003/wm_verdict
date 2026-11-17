"""闭环主表独立核对（2026-11-14）。不调用 make_snapshot / ext_models 的计数代码，从原始文件重数后与主表逐档比对。
A 我们（LingBot-VA / OpenWAM-α）：runs/ 原始分片（gen_closed_main.SPEC 只用来确定哪些批次算数），自己做 H=400 截断，按 unit_id 去重并查冲突。
B Cosmos：作者 condition_metrics.csv。
C UVA：作者审计覆盖文件的第一遍成功数；审计未覆盖的格重读 NAS 逐单元 result.json（repeat0 / 单遍）。
D VPP2：作者 task_summary.json / T4_summary.json。
E 总览 HTML：DATA 逐档 == 主表；六类汇总 == 主表逐档重新加总。"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import csv, glob, json, os, re, sys, collections

OUT = os.path.dirname(os.path.abspath(__file__))
LIB = _wmvp.P("$WMV_ROOT/libero/out")
TSV = f"{LIB}/closed_main/主表_长表_20261114.tsv"
HTML = f"{LIB}/closed_main/闭环鲁棒性总览_20261114.html"
_wmvp.bootstrap()
from gen_closed_main import SPEC, RUNS            # 只取批次清单

T = {(r["model"], r["task"], r["level"]): r for r in csv.DictReader(open(TSV), delimiter="\t")}
TASK = {"T0_t5": "T0", "T1_t3": "T1", "T2_t2": "T2", "T3_t4": "T3", "T4_t6": "T4"}
REP = collections.OrderedDict()                   # (model, task, level) -> dict(source, n, k, ok, detail)
issues = []


def put(key, src, n, k, extra=""):
    t = T.get(key)
    ok = t is not None and int(t["n"]) == n and int(t["n_success"]) == k
    REP[key] = dict(source=src, n=n, k=k, table_n=t and int(t["n"]), table_k=t and int(t["n_success"]), ok=ok, extra=extra)


# ---------------- A 我们
MN = {"lingbot": "LingBot-VA", "openwam": "OpenWAM-α"}
for d in SPEC:
    for mk, model in MN.items():
        units = collections.defaultdict(dict)     # level -> unit_id -> (success, file)
        lines = collections.Counter(); ksum = collections.Counter(); conflicts = []; lvl_mismatch = 0; h520 = 0
        for name, pats in SPEC[d]:
            if not name.startswith(mk):
                continue
            for p in pats:
                for f in sorted(glob.glob(f"{RUNS}/{p}", recursive=True)):
                    for line in open(f):
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        cl = r.get("closed_loop")
                        if not isinstance(cl, dict) or cl.get("success") is None:
                            continue
                        ok = bool(cl["success"])
                        if r.get("H_env") == 520:                 # H=400 事后截断（eval_design §28）
                            h520 += 1
                            fs = cl.get("first_success_step")
                            ok = ok and fs is not None and fs <= 400
                        sid = r.get("setting_id") or "none"
                        lv = sid.split("__")[-1] if "__" in sid else sid
                        lv_u = r["unit_id"].split("__")[-1] if r["unit_id"].count("__") >= 2 else "none"
                        if lv_u != lv:
                            lvl_mismatch += 1
                        lines[lv] += 1; ksum[lv] += ok
                        u = units[lv]
                        if r["unit_id"] in u and u[r["unit_id"]][0] != ok:
                            conflicts.append((lv, r["unit_id"], u[r["unit_id"]][1], f))
                        u.setdefault(r["unit_id"], (ok, os.path.relpath(f, RUNS)))
        for lv, u in units.items():
            n_u = len(u); k = sum(v[0] for v in u.values())
            extra = []
            if lines[lv] != n_u:
                extra.append(f"重复行 {lines[lv] - n_u}")
            put((model, TASK[d], lv), "runs 原始分片", lines[lv], ksum[lv], "；".join(extra))
            if lines[lv] != n_u:
                issues.append(("dup", model, TASK[d], lv, lines[lv], n_u))
        if conflicts:
            issues.append(("conflict", model, TASK[d], conflicts[:5]))
        if lvl_mismatch:
            issues.append(("level_mismatch", model, TASK[d], lvl_mismatch))

# ---------------- B Cosmos（作者 condition_metrics）
NR = {0.05: 1, 0.1: 2, 0.2: 3, 0.3: 4, 0.4: 5}
def cos_level(c):
    if c == "baseline": return "none"
    if c == "N1_step50": return "N1_1"
    if c.startswith("NR_"): return f"NR{NR[round(float(c[3:]), 2)]}"
    if c.startswith("NL_"): i = int(c[3:]); return f"NL{i}" if i else None
    if c in ("O0", "Em0", "Ef0", "G0", "K0", "N_sham"): return None
    return c
CM = _wmvp.P("$COSMOS_DATA_ROOT/cosmos_policy/analysis/closed_loop_audit_20261111/condition_metrics.csv")
for r in csv.DictReader(open(CM, encoding="utf-8-sig")):
    lv = cos_level(r["condition"])
    if lv:
        put(("Cosmos Policy", r["task"], lv), "作者 condition_metrics.csv", int(r["n"]), int(r["successes"]))

# ---------------- C UVA
EM = {1.5: 1, 2.0: 2, 4.0: 3, 8.0: 4}; EF = {0.8: 1, 0.6: 2, 0.4: 3, 0.2: 4}; GG = {0.9: 1, 0.75: 2, 0.55: 3, 0.4: 4}
NLW = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}; OCC = {8: 1, 16: 2, 24: 3, 32: 4, 40: 5}; BX = {"gaussian_blur": "Bg", "fog": "Bf", "zoom_blur": "Bz"}
def uva_level(fam, lv):
    lv = None if lv is None else float(lv)
    if fam in ("normal", "baseline"): return "none"
    if fam == "B0": return "B0"
    if fam in ("A", "P"): return f"{fam}{int(lv)}"
    if fam == "O": return f"O{OCC[int(lv)]}" if int(lv) in OCC else None
    if fam == "E_mass": return f"Em{EM[lv]}" if lv in EM else None
    if fam == "E_friction": return f"Ef{EF[round(lv, 2)]}" if round(lv, 2) in EF else None
    if fam == "G": return f"G{GG[round(lv, 2)]}" if round(lv, 2) in GG else None
    if fam == "N1": return "N1_1"
    if fam == "NR": return f"NR{NR[round(lv, 2)]}" if round(lv, 2) in NR else None
    if fam == "NL": return f"NL{int(lv)}" if lv >= 1 else None
    if fam == "NLw": return f"NLw{NLW[int(lv)]}" if int(lv) in NLW else None
    if fam in BX: return f"{BX[fam]}{int(lv)}"
    return None
SP = _wmvp.P("$SCRATCH/uva_idx")
AUD = {}
for f in [f"{SP}/local-workspace/UVA-Validation/closed-only-scope-audit-coverage.json",
          f"{SP}/supplements/T2-T3-friction-20261112T020333Z/UVA-Validation/closed-only-scope-audit-coverage.json",
          f"{SP}/supplements/T4-complete-20261112T131453Z/UVA-Validation/closed-only-scope-audit-coverage.json"]:   # 时间顺序，后者覆盖前者
    for c in json.load(open(f))["cells"]:
        lv = uva_level(c["family"], c["level"])
        if lv and c.get("closed_successes_by_repeat"):
            AUD[(c["task"], lv)] = (c["closed_successes_by_repeat"][0], c["job"], os.path.basename(os.path.dirname(os.path.dirname(f))))


def raw_uva(closed_dir, pat):
    n = k = 0
    for f in sorted(glob.glob(f"{closed_dir}/{pat}")):
        r = json.load(open(f)); n += 1; k += bool(r.get("native_success"))
    return n, k


uva_keys = [k for k in T if k[0] == "UVA"]
t01 = {(c["task"], c["level"]): c for c in json.load(open(f"{LIB}/task2/uva_t01_cells.json"))["cells"]}
kc = {(c["task"], c["level"]): c for c in json.load(open(f"{LIB}/task2/uva_k_cells.json"))["cells"]}
IDX = {}
for c in json.load(open(f"{LIB}/task2/uva_index.json")):
    lv = uva_level(c["family"], c["level"])
    if lv:
        IDX[(c["task"], lv)] = c
TID = {"T0": 5, "T1": 3}
for key in uva_keys:
    _, t, lv = key
    raw = author = None; src = []
    if (t, lv) in kc:                                                  # K 族
        c = kc[(t, lv)]; raw = raw_uva(c["closed_dir"], "*/repeat0/result.json"); src.append(f"K 逐单元 repeat0（作者第二遍 {c.get('k_repeat1')}）")
    elif t in TID and (t, lv) in t01:                                  # T0/T1：两任务可能同目录，按 task<id>_ 前缀筛
        c = t01[(t, lv)]; raw = raw_uva(c["closed_dir"], f"task{TID[t]}_*/result.json"); src.append("T0/T1 逐单元（按任务前缀）")
    elif (t, lv) in IDX and os.path.isdir(IDX[(t, lv)].get("closed_dir") or ""):
        c = IDX[(t, lv)]; raw = raw_uva(c["closed_dir"], "*/repeat0/result.json"); src.append("索引目录逐单元 repeat0")
        if c.get("author_successes_by_repeat"):
            author = c["author_successes_by_repeat"][0]
    elif t == "T4" and lv.startswith("Ef"):
        x = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}[int(lv[2])]
        d = _wmvp.P(f"$UVA_DATA_ROOT/20261102-native-candidate/t4-friction-v6-closed-20261112/friction-v6/T4/E_friction_{x}/closed")
        raw = raw_uva(d, "*/repeat0/result.json"); src.append("T4 Ef v6 逐单元 repeat0")
    if (t, lv) in AUD:
        author = AUD[(t, lv)][0]; src.append(f"作者审计 {AUD[(t, lv)][2]} 作业 {AUD[(t, lv)][1]}")
    tn, tk = int(T[key]["n"]), int(T[key]["n_success"])
    if raw and raw[0]:
        put(key, " + ".join(src), raw[0], raw[1], "" if author is None else f"作者第一遍成功数 {author}" + ("（一致）" if author == raw[1] else "（不一致）"))
        if author is not None and author != tk and not (t, lv) in kc:
            issues.append(("uva_author_vs_table", key, author, tk))
    elif author is not None:
        put(key, " + ".join(src) + "（无原始目录，只对作者数）", tn, author, "n 取主表")
    else:
        REP[key] = dict(source="无独立来源", n=None, k=None, table_n=tn, table_k=tk, ok=None, extra="")

# ---------------- D VPP2
V = f"{LIB}/task2/ext_snapshots/vpp2_20261112"
ts = json.load(open(f"{V}/task_summary.json")); t4 = json.load(open(f"{V}/T4_summary.json"))
for t, d in {**ts, "T4": t4}.items():
    put(("VPP2", t, "none"), "作者 task_summary.json / T4_summary.json", int(d["completed_units"]), int(d["successes"]),
        "作者 complete=" + str(d.get("complete")))
json.dump(dict(task_summary=ts, T4=t4), open(f"{OUT}/vpp2_author_summaries_seen.json", "w"), ensure_ascii=False, indent=1)

# ---------------- E HTML
s = open(HTML).read()
def const(name):
    i = s.index(f"const {name}="); j = s.index("\n", i)
    t = s[i + len(f"const {name}="):j].rstrip().rstrip(";")
    return json.loads(t)
H = const("DATA")
hd = {(r["model"], r["task"], r["level"]): r for r in H}
html_bad = []
assert len(H) == len(T) == len(hd), (len(H), len(T), len(hd))
for key, t in T.items():
    h = hd.get(key)
    if h is None or int(h["n"]) != int(t["n"]) or abs(float(h["success_rate"]) - float(t["success_rate"])) > 1e-9 \
            or str(h["median_first_success_step"]) != str(t["median_first_success_step"]) or h["complete"] != t["complete"]:
        html_bad.append((key, h, {k: t[k] for k in ("n", "success_rate", "median_first_success_step", "complete")}))
cat = const("CAT"); cat_bad = []
NM = lambda l: "N1_1" if l == "N1" else l          # 六类汇总里 N1 档写作 N1，主表叫 N1_1
for r in cat["rows"]:
    lv = [NM(l) for l in r["levels_used"].split()]
    n = sum(int(T[(r["model"], r["task"], l)]["n"]) for l in lv)
    k = sum(int(T[(r["model"], r["task"], l)]["n_success"]) for l in lv)
    if (n, k) != (r["n"], r["n_success"]):
        cat_bad.append((r["model"], r["task"], r["category"], (r["n"], r["n_success"]), (n, k)))
    for l in map(NM, r["levels_missing"].split()):
        if (r["model"], r["task"], l) in T and T[(r["model"], r["task"], l)]["complete"] == "y" and not T[(r["model"], r["task"], l)]["exclude_cross"]:
            cat_bad.append(("missing-but-present", r["model"], r["task"], l))

json.dump(dict(rows=[dict(model=k[0], task=k[1], level=k[2], **v) for k, v in REP.items()], issues=issues,
               html_bad=html_bad, cat_bad=cat_bad), open(f"{OUT}/verify_independent_result.json", "w"), ensure_ascii=False, indent=1, default=str)
by = collections.Counter((k[0], v["ok"]) for k, v in REP.items())
print("逐档比对（模型, 一致?）:", dict(by))
print("主表格数", len(T), "已核对", sum(1 for v in REP.values() if v["ok"] is not None), "无独立来源",
      [k for k, v in REP.items() if v["ok"] is None])
print("主表有但核对没覆盖:", [k for k in T if k not in REP])
print("核对有但主表没有:", [k for k in REP if k not in T][:20])
print("不一致:", [(k, v) for k, v in REP.items() if v["ok"] is False][:30])
print("issues:", issues[:20])
print("HTML 逐档不一致:", len(html_bad), html_bad[:3]); print("HTML 六类汇总不一致:", len(cat_bad), cat_bad[:5])
print("VPP2 主表:", [(k[1], T[k]["n"], T[k]["n_success"]) for k in T if k[0] == "VPP2" and k[2] == "none"])
