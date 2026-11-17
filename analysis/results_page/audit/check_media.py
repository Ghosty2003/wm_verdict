"""数据核对 2026-11-15 · 规则 8：Cosmos T4 终局帧定位表与 T1 932 条补传媒体的计数核对（只读，标准库）。
输出 check_media.json。  python3 check_media.py"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import collections
import csv
import hashlib
import json
import os

SA = _wmvp.P("$COSMOS_DATA_ROOT")
HO = f"{SA}/web_audit_20261115/handoff"
OUT = os.path.dirname(os.path.abspath(__file__))
res = {}

# ---- T4 定位表
L = list(csv.DictReader(open(f"{HO}/Cosmos_T4_terminal_frame_locator.csv", encoding="utf-8-sig")))
ids = collections.Counter(r["annotation_id"] for r in L)
by_id = {r["annotation_id"]: r for r in L}
res["T4_locator"] = dict(rows=len(L), unique_annotation_id=len(ids), dup_ids=sum(v > 1 for v in ids.values()),
                         real_fail=sum(r["actual_success"] == "False" for r in by_id.values()),
                         real_success=sum(r["actual_success"] == "True" for r in by_id.values()),
                         tasks=dict(collections.Counter(r["task"] for r in L)),
                         archives=dict(collections.Counter(r["dropbox_archive"] for r in L)),
                         human_label=dict(collections.Counter(r["human_label"] or "(空)" for r in by_id.values())),
                         agent_label=dict(collections.Counter(r["agent_label"] or "(空)" for r in by_id.values())))
UI = [r for r in csv.DictReader(open(f"{SA}/optimism_update_20261114/unit_index.csv", encoding="utf-8-sig")) if r["task"] == "T4"]
res["T4_unit_index"] = dict(rows=len(UI), unique_annotation_id=len({r["annotation_id"] for r in UI}),
                            locator_ids_in_unit_index=len(set(ids) & {r["annotation_id"] for r in UI}))

# ---- T1 932 补传
M = list(csv.DictReader(open(f"{HO}/T1_932_media_manifest_20261115.csv", encoding="utf-8-sig")))
cur = {r["unit_id"]: r for r in csv.DictReader(open(f"{SA}/optimism_update_20261114/unit_index.csv", encoding="utf-8-sig"))
       if r["task"] == "T1" and r["cohort"] == "current_retained_5x10"}
rec = json.load(open(f"{SA}/t1_terminal_gap_upload_20261115/UPLOAD_VERIFIED.json"))
rec2 = json.load(open(f"{HO}/T1_932_MEDIA_UPLOAD_VERIFIED.json"))
mf = f"{HO}/T1_932_media_manifest_20261115.csv"
sha = hashlib.sha256(open(mf, "rb").read()).hexdigest()
idx = json.load(open(f"{SA}/t1_terminal_gap_upload_20261115/archives/Cosmos_T1_terminal_gap932_part000.tar.zst.index.json"))
res["T1_932"] = dict(rows=len(M), unique_units=len({r["unit_id"] for r in M}),
                     acquisition=dict(collections.Counter(r["acquisition"] for r in M)),
                     in_current_restored=sum(1 for r in M if cur.get(r["unit_id"], {}).get("acquisition") == "original_episode_final_window_restored"),
                     outcome_matches_unit_index=sum(1 for r in M if cur.get(r["unit_id"]) and cur[r["unit_id"]]["actual_success"] == r["actual_success"]),
                     png_paths_unique=len({r["terminal_png"] for r in M}), npz_paths_unique=len({r["prediction_npz"] for r in M}),
                     png_archive=dict(collections.Counter(r["png_archive"] for r in M)),
                     archive_index_files=idx.get("files"), archive_index_dropbox_hash_matches_receipt=(
                         idx["hashes"].get("dropbox") == [f["dropbox"] for f in rec["files"] if f["file"].endswith(".tar.zst")][0]),
                     label_status=dict(collections.Counter(r["label_status"] for r in M)),
                     human_label=dict(collections.Counter(r["human_label"] or "(空)" for r in M)),
                     receipt_verified=rec.get("verified"), receipt_units=rec.get("units"), receipt_png=rec.get("png"), receipt_npz=rec.get("npz"),
                     receipt_files_all_verified=all(f["verified"] for f in rec["files"]), receipt_same_as_handoff=(rec == rec2),
                     manifest_bytes=os.path.getsize(mf),
                     manifest_bytes_in_receipt=[f["bytes"] for f in rec["files"] if f["file"] == "unit_media_manifest.csv"][0],
                     manifest_sha256_local=sha, receipt_scope=rec.get("scope"))
res["T1_restored_total"] = sum(1 for r in cur.values() if r["acquisition"] == "original_episode_final_window_restored")
json.dump(res, open(f"{OUT}/check_media.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
