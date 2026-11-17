"""UVA 档位 → NAS 闭环原始目录索引（供 ext_models.uva 读逐单元 result.json）。
来源：UVA 作者自己的 closed-only-scope-audit-coverage.json（T2/T3 非基线 + T4 全部）与 T2-T3-complete-sync-audit-summary.json
（T2/T3 基线），每格只认作者审计通过的那个 job；job → NAS 目录取自作者同步时写的 raw-synced.json 的 remote_source。
  python build_uva_index.py <dropbox 下载的 UVA 元数据根目录>
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob
import json
import os
import sys

ROOT = sys.argv[1]
MAIN50 = _wmvp.P("$UVA_DATA_ROOT/20261102-native-candidate/shared-main50-20261105-8552440c")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uva_index.json")


def newest(name):
    fs = sorted(glob.glob(f"{ROOT}/**/{name}", recursive=True), key=lambda f: ("supplements" in f, f))
    return fs[-1]


raw = {}
for f in sorted(glob.glob(f"{ROOT}/**/raw-synced.json", recursive=True), key=lambda f: ("supplements" in f, f)):
    r = json.load(open(f))
    if f.endswith("/closed/raw-synced.json") or r.get("remote_source", "").endswith("/closed"):
        raw[str(r["job"])] = r            # supplements 排后，覆盖旧同步
cov = json.load(open(newest("closed-only-scope-audit-coverage.json")))
t23 = json.load(open(newest("T2-T3-complete-sync-audit-summary.json")))

NC = _wmvp.P("$UVA_DATA_ROOT/20261102-native-candidate")


def fallback(c):
    """原协议共享批次（raw-synced 尚未同步到 Dropbox 的）按作者 NAS 目录约定定位，存在且含 50 单元才认。"""
    p = c["audit_report"].split("\\")
    if p[0] == "shared-mass15-raw":
        d = f"{NC}/shared-mass15-20261105/{p[1]}/closed"
    elif p[0] == "shared-perturbation-raw" and p[1] in ("mass-remaining", "gain"):
        d = f"{NC}/shared-{p[1]}-20261105/{p[2]}/{p[3]}/closed"
    elif p[0] == "shared-perturbation-raw" and p[1] == "ap":
        d = f"{NC}/shared-ap-20261105/{p[2]}/{p[3].replace('_', '')}/closed"
    else:
        return None
    return d if len(glob.glob(f"{d}/*/repeat0/result.json")) == 50 else None


cells, missing = [], []
for c in cov["cells"]:
    if not (c["computation_complete"] and c["full_raw_independent_audit_passed"]):
        continue
    cells.append(dict(task=c["task"], family=c["family"], level=c["level"], job=str(c["job"]),
                      author_successes_by_repeat=c["closed_successes_by_repeat"], audit_report=c["audit_report"]))
for T, t in t23["tasks"].items():
    b = t["baseline"]
    cells.append(dict(task=T, family="baseline", level=None, job=None,          # shared-main50 原始目录（作者 audit_report 同名）
                      author_successes_by_repeat=b["closed_successes_by_repeat"], audit_report=b["audit_report"],
                      closed_dir=f"{MAIN50}/{T}/closed"))
for c in cells:
    if "closed_dir" in c:
        continue
    if c["job"] and c["job"] in raw:
        c["closed_dir"] = raw[c["job"]]["remote_source"]
    elif fallback(c):
        c["closed_dir"] = fallback(c)
    else:
        missing.append(c)
json.dump(cells, open(OUT, "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "with dir", sum("closed_dir" in c for c in cells))
for c in missing:
    print("NO DIR", c["task"], c["family"], c["level"], c["job"], c["audit_report"])
