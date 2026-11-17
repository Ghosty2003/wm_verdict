"""同卷对比标注 · 第 1 步：四模型逐单元盘点。只读，不跑模型。
v4（2026-11-16，Jasmine：“把原本的第一个 seed 第二个 seed 换成第一组第二组……都用齐全的数据”）：
  页面不再按同 seed 对齐，每个模型在每个 任务×档位 格里各自挑单元。本脚本给每个单元标明：
    real / imag          真实画面、想象末帧来源（have / replay / latent / none）
    real_run / imag_run  画面与想象帧各自属于哪一次运行（运行标识，见下）
    complete             real 与 imag 都有，且 real_run == imag_run（同一次运行）
    machine              cluster / H100
    case                 同一初始条件 + 同一策略种子的单元归为一个 case（同一 case 的集群版与 H100 版、原执行与新重跑，挑页时只取一个）
  运行标识（口径文件 out/h100/口径决定_20261116.md：同一单元的真实成败与想象帧必须来自同一次运行）：
    rec:<uid>        原执行。真实画面 = 原执行存的画面，或按原执行动作在 x86 上回放（回放成败与记录一致，check_real_consistency 核对）；
                     想象帧 = 原执行自己存的想象（closed npz / closed_imagined / famB closed_frames latents / famF closed），
                     或逐位复现原执行最后一次规划的重算（OpenWAM lastcall gate_pass；Cosmos 末窗补算：输入哈希一致且 commanded 动作逐值一致）。
                     OpenWAM ow_backfill（closed_frames 是另跑的一遍）：只有它的 actions 与原执行动作逐位相等才算 rec，否则记 file:<该文件>。
    file:<路径>      自带真实画面与想象帧的单次运行文件：集群 x86 全量重跑（OpenWAM x86_rerun、LingBot rerun_closed、OpenWAM rerun_closed/phys）、
                     H100 运行（只用 out/h100/_pull/ledger.tsv 里 sha256 已校验的 npz）、Cosmos T0/T1 11-13 全量新重跑、Cosmos F 族（closed_loop1）。
  已有卡沿用：库里同 unit_id 的 round 2 / round 3 卡，来源按出卡时的数据（round 3 / 缓存 = optimism/units.json 的 frames；decode 日志的 decoded_from）。
  UVA：closed.npz 只有状态与动作，原执行未存想象；队友 terminal-imagination-20261114/tail32 是从第 368 步观测事后预测的 32 步
       （plan 里 frame_source = "Posthoc forecast ... not original terminal planning imagination"，native_terminal_metric_eligible = false），
       不是该次运行自己的规划想象，不用。UVA 全部只有真实画面。
  主表范围改用 closed_main/主表_长表_20261116.tsv（加入 Cosmos F 族、UVA T0 K 族；NLw 已去掉）。
输出：inventory.jsonl（每单元一行）、inventory_summary.json。
  python inventory.py        （需要 numpy：在 CPU 节点上用 lerobot_libero 解释器跑）
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = _wmvp.P("$WMV_ROOT")
LIB = f"{ROOT}/libero/out"
RUNS = f"{LIB}/runs"
H100 = f"{LIB}/h100"
WEB = _wmvp.P("$WMV_WEB_ROOT")
SA = _wmvp.P("$COSMOS_DATA_ROOT")
UVA = _wmvp.P("$UVA_DATA_ROOT/20261102-native-candidate")
HERE = os.path.dirname(os.path.abspath(__file__))
_wmvp.bootstrap()
import replay_frames as RF   # noqa: E402
import ext_models as E       # noqa: E402

TN = {5: "T0", 3: "T1", 2: "T2", 4: "T3", 6: "T4"}
TID = {v: k for k, v in TN.items()}
TT = {"T0": "t5", "T1": "t3", "T2": "t2", "T3": "t4", "T4": "t6"}
MT = list(csv.DictReader(open(f"{LIB}/closed_main/主表_长表_20261116.tsv"), delimiter="\t"))
MKEY = {"LingBot-VA": "lingbot", "OpenWAM-α": "openwam", "Cosmos Policy": "cosmos", "UVA": "uva"}
MAIN = {(MKEY[r["model"]], r["task"], r["level"]) for r in MT if r["model"] in MKEY}
OUT = []
NOTE = collections.Counter()


def jl(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


# ---------- 我们的种子表：(tid, 前缀, anchor) -> {seed: k}（只用于 slot 记录，v4 挑页不再按 slot 对齐）
SEEDINV = collections.defaultdict(dict)
for f in glob.glob(f"{RUNS}/*/records*.jsonl") + glob.glob(f"{RUNS}/*/*/records*.jsonl"):
    for r in jl(f):
        m = re.search(r"libero10_t(\d)_([a-z]+)(\d+)", r.get("anchor_id") or "")
        k = r.get("seed_index")
        if k is None:
            mk = re.search(r"__k(\d+)", r.get("unit_id", "")); k = mk and int(mk.group(1))
        if m and k is not None and r.get("imagination_seed") is not None:
            SEEDINV[(int(m.group(1)), m.group(2), int(m.group(3)))][int(r["imagination_seed"])] = int(k)


def slot_ours(uid):
    m = re.search(r"libero10_t(\d)_([a-z]+)(\d+)__k(\d+)", uid)
    return f"{m.group(2)}{int(m.group(3))}_k{int(m.group(4))}" if m else None


def slot_ext(tid, name, level=None):
    m = re.search(r"([a-z]+)(\d+)_k(\d+)$", name)
    if m:
        return f"{m.group(1)}{int(m.group(2))}_k{int(m.group(3))}"
    m = re.search(r"anchor(\d+)_seed(\d+)", name)
    if m:
        a, s = int(m.group(1)), int(m.group(2))
        pre = "fa" if level and level[0] == "A" else "fp" if level and level[0] == "P" else "a"
        k = SEEDINV.get((tid, pre, a), {}).get(s)
        return f"{pre}{a}_k{k}" if k is not None else f"ext_init{a}_s{s}"
    return f"ext_{name}"


def card(rnd, uid):
    for x in ("webp", "png"):
        p = f"{WEB}/cards/r{rnd}/{uid}.{x}"
        if os.path.exists(p):
            return p
    return None


def case_of(model, T, level, uid):
    """同一初始条件 + 同一策略种子 → 同一 case（去掉 __x86rerun / __h100 / rerun1113 / f1114 这类运行标记）。"""
    b = re.sub(r"__(x86rerun|h100)$", "", uid)
    b = b.replace("cosmos_rerun1113_", "cosmos_").replace("cosmos_f1114_", "cosmos_")
    return f"{model}|{T}|{level}|{b}"


def add(model, T, level, uid, slot, real_src, real_path, success, fss, imag_src, imag_path, exec_note, real_run=None, imag_run=None,
        machine="cluster", extra=None):
    if (model, T, level) not in MAIN:
        NOTE[f"not_in_main|{model}|{T}"] += 1
        return
    has_r, has_i = real_src not in (None, "none"), imag_src not in (None, "none")
    OUT.append(dict(model=model, T=T, level=level, unit_id=uid, slot=slot, case=case_of(model, T, level, uid),
                    real=real_src or "none", real_path=real_path, success=success, first_success_step=fss,
                    imag=imag_src or "none", imag_path=imag_path, execution=exec_note, machine=machine,
                    real_run=real_run if has_r else None, imag_run=imag_run if has_i else None,
                    complete=bool(has_r and has_i and real_run and real_run == imag_run),
                    r2_card=card(2, uid), r3_card=card(3, uid), **(extra or {})))


def npz_outcome(p, key="success"):
    s = np.asarray(np.load(p)[key]).astype(bool).reshape(-1)
    return bool(s.any()), (int(np.argmax(s)) + 1 if s.any() else None)


# ---------- 我们：想象帧来源表（optimism/units.json：OpenWAM imagined / LingBot latents）
IMAG = {u["unit_id"]: u for u in json.load(open(f"{LIB}/optimism/units.json"))}
DECODED_FROM = {}                                                  # 同卷 v1–v3 在 CPU / GPU 上解码出的 LingBot 想象卡
for f in glob.glob(f"{HERE}/render.decode*.jsonl"):
    for l in open(f):
        x = json.loads(l)
        if isinstance(x.get("result"), dict) and x["result"].get("decoded_from"):
            DECODED_FROM[x["unit_id"]] = x["result"]["decoded_from"]
REC_IMAG_SOURCES = {"ow_phys", "ow_famB", "ow_famF", "lb_famF", "lb_famB_t2"}   # 原执行自己存的想象

# ---------- 我们：真实帧（集群）
seen = {}
J = RF.jobs() + RF.ow_t2rerun_jobs((2, 4, 6))
for j in J:
    have = f"{RUNS}/{j['run']}/replay_frames/{j['uid']}.npz"
    src = "have" if os.path.exists(have) else "replay" if os.path.exists(j["act"][0]) else "none"
    prev = seen.get(j["uid"])
    if prev and {"have": 2, "replay": 1, "none": 0}[prev[0]] >= {"have": 2, "replay": 1, "none": 0}[src]:
        continue
    # 回放的是哪一次运行的动作：rerun_closed 的重跑文件 → file:<该文件>；其余都是原执行的动作 → rec
    run = f"file:{os.path.normpath(j['act'][0])}" if j["run"].endswith("rerun_closed") else f"rec:{j['uid']}"
    seen[j["uid"]] = (src, have, dict(j, real_run=run))
for tid in (2, 4, 6):                                               # LingBot 集群全量重跑：同一 npz 里有逐步真实画面与想象 latent
    for r in jl(f"{RUNS}/lingbot_t{tid}/rerun_closed/*.jsonl"):
        p = os.path.normpath(f"{ROOT}/{r['frames_file']}")
        if os.path.exists(p) and (r["unit_id"] not in seen or seen[r["unit_id"]][0] != "have"):
            seen[r["unit_id"]] = ("have", p, dict(model="lingbot", tid=tid, level=r["level"], run=f"lingbot_t{tid}/rerun_closed", rec=r["rerun"],
                                                  lb=True, real_run=f"file:{p}", lb_rerun_npz=p))
for u in IMAG.values():                                             # F 族：原执行存档 npz（LingBot 存了画面；OpenWAM 按 executed 回放）
    if u["source"] in ("lb_famF", "ow_famF") and u["unit_id"] not in seen:
        replay_ow = u["source"] == "ow_famF"
        seen[u["unit_id"]] = ("replay" if replay_ow else "have", u["frames"],
                              dict(model=u["model"], tid=TID[u["T"]], level=u["level"], run="openwam_famF" if replay_ow else "lingbot_famF",
                                   rec=dict(success=u["success"], first_success_step=None), act=(u["frames"], "executed"), real_run=f"rec:{u['unit_id']}"))
        if replay_ow and os.path.exists(f"{RUNS}/openwam_famF/replay_frames/{u['unit_id']}.npz"):
            seen[u["unit_id"]] = ("have", f"{RUNS}/openwam_famF/replay_frames/{u['unit_id']}.npz", seen[u["unit_id"]][2])
# OpenWAM 集群 x86 全量重跑（同一 npz 有逐步真实画面与想象帧；新 unit_id = <uid>__x86rerun）
X86 = {}
for r in jl(f"{RUNS}/openwam_*/x86_rerun.*.jsonl") + jl(f"{RUNS}/openwam_*/*/x86_rerun.*.jsonl"):
    p = f"{RUNS}/{r['run']}/x86_rerun/{r['unit_id']}.npz"
    if "error" not in r and os.path.exists(p):
        X86[r["unit_id"]] = (p, r)
LASTCALL = {}
for r in jl(f"{RUNS}/openwam_*/lastcall.*.jsonl") + jl(f"{RUNS}/openwam_*/*/lastcall.*.jsonl"):
    p = f"{RUNS}/{r['run']}/lastcall/{r['unit_id']}.npz"
    if r.get("gate_pass") and os.path.exists(p):
        LASTCALL[r["unit_id"]] = p


def backfill_same(uid, j):
    """ow_backfill：closed_frames 的 actions 与原执行动作（回放所用）逐位相等 → 同一次运行。"""
    try:
        a = np.asarray(np.load(IMAG[uid]["frames"])["actions"])
        b = np.asarray(np.load(j["act"][0])[j["act"][1]])
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    except Exception as ex:  # noqa: BLE001
        NOTE[f"backfill_cmp_error|{ex!r}"[:120]] += 1
        return False


for uid, (src, path, j) in seen.items():
    m = j["model"]; T = TN[j["tid"]]; lv = j["level"]
    rec = j.get("rec") or {}
    iu = IMAG.get(uid)
    if m == "openwam" and uid in X86:
        p, r = X86[uid]
        add(m, T, lv, f"{uid}__x86rerun", slot_ours(uid), "have", p, bool(r["rerun_success"]), r["rerun_first_success_step"], "have", p,
            "x86 full closed-loop rerun 20261115", f"file:{p}", f"file:{p}",
            extra=dict(run=j["run"], imag_source="x86_rerun", orig_unit_id=uid,
                       recorded_success=bool(rec.get("success")), recorded_first_success_step=rec.get("first_success_step")))
        continue
    real_run = j["real_run"]
    isrc, ipath, irun, isource = "none", None, None, None
    if m == "openwam":
        if iu:
            isrc, ipath, isource = "have", iu["frames"], iu["source"]
            if iu["source"] in REC_IMAG_SOURCES:
                irun = f"rec:{uid}"
            elif iu["source"] == "ow_backfill":
                same = j.get("act") and backfill_same(uid, j)
                NOTE[f"ow_backfill_actions_bitwise|{bool(same)}"] += 1
                irun = f"rec:{uid}" if same else f"file:{os.path.normpath(iu['frames'])}"
            else:                                                       # ow_rerun：rerun_closed/phys 文件本身
                irun = f"file:{os.path.normpath(iu['frames'])}"
        elif uid in LASTCALL:
            isrc, ipath, irun, isource = "have", LASTCALL[uid], f"rec:{uid}", "lastcall"
    elif m == "lingbot":
        npz = j.get("lb_rerun_npz")
        prov = (iu and iu["frames"]) or DECODED_FROM.get(uid)
        prov = prov and os.path.normpath(prov)               # 已有卡 / 缓存当时解码所用的文件
        if card(3, uid) or os.path.exists(f"{LIB}/optimism/cache/{uid}.npz"):
            if prov:
                isrc, ipath = "have", card(3, uid) or f"{LIB}/optimism/cache/{uid}.npz"
                isource = (iu and iu["source"]) or "decoded"
                irun = f"rec:{uid}" if iu and iu["source"] in REC_IMAG_SOURCES else f"file:{prov}"
            else:
                NOTE["lingbot_r3_card_unknown_provenance"] += 1
        elif npz:                                                          # 同一 rerun npz 里的想象 latent（需解码）
            isrc, ipath, irun, isource = "latent", npz, f"file:{npz}", "lb_rerun_npz"
        elif iu:
            isrc, ipath, isource = "latent", iu["frames"], iu["source"]
            irun = f"rec:{uid}" if iu["source"] in REC_IMAG_SOURCES else f"file:{os.path.normpath(iu['frames'])}"
    rs = src if src != "replay" else "replay"
    add(m, T, lv, uid, slot_ours(uid), rs, path if src != "replay" else j["act"][0], bool(rec.get("success")),
        rec.get("first_success_step"), isrc, ipath, j["run"], real_run, irun,
        extra=dict(run=j["run"], imag_source=isource, imag_key=(iu or {}).get("key") or ("imag_latents" if isource == "lb_rerun_npz" else None)))

# ---------- H100 运行（台账 sha256 已校验的 npz；每个 npz 自带 meta_json.machine = H100；真实画面与想象帧同一次运行）
H100_SEEN = 0
for l in open(f"{H100}/_pull/ledger.tsv"):
    rel = l.split("\t")[0]
    if not (rel.startswith("runs/") and rel.endswith(".npz")):
        continue
    p = f"{H100}/{rel}"
    uid = os.path.basename(rel)[:-4]
    mm = re.match(r"(lingbot|openwam)_libero10_t(\d)_[a-z]+\d+__k\d+(?:__(.+))?$", uid)
    if not mm or not os.path.exists(p):
        NOTE["h100_skip"] += 1; continue
    m, T, lv = mm.group(1), TN[int(mm.group(2))], mm.group(3) or "none"
    try:
        z = np.load(p); meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        if meta.get("machine") != "H100":
            NOTE["h100_meta_not_H100"] += 1; continue
        ok, fs = npz_outcome(p)
        if m == "openwam":
            isrc = "have" if "imagined" in z.files and "agentview_steps" in z.files else "none"
        else:
            isrc = "have" if "imag_decoded" in z.files else ("latent" if "imag_latents_00" in z.files else "none")
    except Exception as ex:  # noqa: BLE001
        NOTE[f"h100_read_error|{ex!r}"[:120]] += 1; continue
    H100_SEEN += 1
    add(m, T, lv, f"{uid}__h100", slot_ours(uid), "have", p, ok, fs, isrc, p if isrc != "none" else None,
        "H100 closed-loop run (pulled, sha256 verified)", f"file:{p}", f"file:{p}", machine="H100",
        extra=dict(run=rel.split("/")[1], imag_source="h100", orig_unit_id=uid, imag_key="imag_decoded" if isrc == "have" and m == "lingbot" else
                   ("imag_latents" if m == "lingbot" else "imagined"), h100_worker=meta.get("worker")))

# ---------- Cosmos
# 末窗补算过闸：输入哈希一致 + commanded 动作逐值一致
GATE_LW = set()
for T in ("T2", "T3"):
    for x in json.load(open(f"{SA}/terminal_imagination_20261113/last_window_manifest_{T}.json")):
        if x.get("commanded_actions_match_original") and x.get("input_matches_original_hash"):
            GATE_LW.add(x["terminal_png"])
TERM = {}
for d in glob.glob(f"{SA}/terminal_imagination_20261113/last_window_recomputed/Cosmos_T*"):
    m = re.match(r"Cosmos_(T\d)_(.+)_(\d+)_seed(\d+)_[0-9a-f]+$", os.path.basename(d))
    if m:
        png = glob.glob(f"{d}/*_terminal.png")
        if png:
            if png[0] in GATE_LW:
                TERM[(m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))] = png[0]
            else:
                NOTE["cosmos_lastwindow_not_gated"] += 1
for r in json.load(open(f"{LIB}/optimism/prescreen/cosmos_rows.json")):          # 原执行已存预测导出的终局帧（frames/）；末窗补算的同样要过闸
    if r["model"] != "Cosmos" or ("last_window_recomputed" in r["png"] and r["png"] not in GATE_LW):
        continue
    m = re.match(r"(\d+)_seed(\d+)$", r["unit_id"])
    if m:
        for lv in r["all_levels"]:
            TERM.setdefault((r["T"], lv, int(m.group(1)), int(m.group(2))), r["png"])
T1OK = set()                                                        # T1 末窗补算：bulk 清单里 commanded 动作逐值一致（输入哈希不一致的不会调用模型）
for f in glob.glob(f"{SA}/t1_terminal_restore_20261114/bulk_manifest_*.json") + [f"{SA}/t1_terminal_restore_20261114/last_window_manifest_T1.json"]:
    for x in json.load(open(f)):
        if x.get("commanded_actions_match_original") and x.get("input_matches_original_hash", True):
            if "terminal_png" in x:                                  # 8 个试点：按终局帧路径
                T1OK.add(x["terminal_png"])
            elif x.get("unit_id"):                                   # bulk：unit_id = T1/<condition>/anchorNN_seedS
                T1OK.add(x["unit_id"])
T1R = {}
for e in json.load(open(f"{SA}/t1_terminal_restore_20261114/terminal_manifest.json")):
    if e.get("terminal_png") and os.path.exists(e["terminal_png"]):
        if e["unit_id"] in T1OK or e["terminal_png"] in T1OK:
            T1R[os.path.dirname(e["original_source"])] = e["terminal_png"]
        else:
            NOTE["cosmos_t1_restore_not_gated"] += 1
CP = f"{SA}/cosmos_policy"
for r in csv.DictReader(open(f"{CP}/analysis/closed_loop_audit_20261111/unit_outcomes.csv", encoding="utf-8-sig")):
    if r["task"] == "T0":
        continue
    lv = E.cosmos_level(r["condition"])
    if lv is None:
        continue
    d = os.path.dirname(f"{CP}/{r['source']}")
    vid = f"{d}/closed_loop1.mp4" if os.path.exists(f"{d}/closed_loop1.mp4") else f"{d}/closed_loop.mp4"
    tid = TID[r["task"]]; name = os.path.basename(d); slot = slot_ext(tid, name, lv)
    ok = r["primary_success"] == "True"
    fs = int(float(r["primary_first_success_step"])) if ok and r["primary_first_success_step"] else None
    ipng = None
    m = re.search(r"anchor(\d+)_seed(\d+)", name)
    if r["task"] == "T1":
        ipng = T1R.get(d)
    elif m:
        lvkey = {"none": "baseline"}.get(lv, lv)
        for key in ((r["task"], r["condition"], int(m.group(1)), int(m.group(2))), (r["task"], lv, int(m.group(1)), int(m.group(2))),
                    (r["task"], lvkey, int(m.group(1)), int(m.group(2)))):
            if key in TERM:
                ipng = TERM[key]; break
    if not ipng and r["task"] == "T4":          # T4 原执行存了每次规划的预测：最后一次规划的 future_primary = 想象末帧
        pd_ = next((f"{d}/{x}" for x in ("closed_loop1_model_predictions", "closed_loop_model_predictions") if os.path.isdir(f"{d}/{x}")), None)
        if pd_ and os.path.exists(f"{pd_}/alignment.json"):
            q = json.load(open(f"{pd_}/alignment.json"))["queries"]
            if q and os.path.exists(f"{pd_}/{q[-1]['prediction_file']}"):
                ipng = f"{pd_}/{q[-1]['prediction_file']}#future_primary"
    uid = f"cosmos_libero10_{TT[r['task']]}_{name}__{lv}"
    add("cosmos", r["task"], lv, uid, slot, "have" if os.path.exists(vid) else "none", vid, ok, fs, "have" if ipng else "none", ipng,
        "original closed loop (audit 20261111)", f"rec:{uid}", f"rec:{uid}")
RR = f"{SA}/cosmos_closed_rerun_20261113"
for T in ("T0", "T1"):                                              # 11-13 全量新重跑：同一目录里真实视频与终局帧（v4 起 T1 也用）
    for res in glob.glob(f"{RR}/{T}/*/units/*/closed_loop_result.json"):
        d = os.path.dirname(res); cond = d.split("/")[-3]; lv = E.cosmos_level(cond)
        if lv is None:
            continue
        u = json.load(open(f"{d}/unit.json")); name = os.path.basename(d)
        slot = f"ext_init{u['anchor_index']}_s{u['policy_seed']}" if u.get("anchor_id", "").startswith("official_init") else slot_ext(TID[T], name, lv)
        rj = json.load(open(res)); png = glob.glob(f"{d}/*_terminal.png")
        add("cosmos", T, lv, f"cosmos_rerun1113_libero10_{TT[T]}_{name}__{lv}", slot, "have" if os.path.exists(f"{d}/real.mp4") else "none",
            f"{d}/real.mp4", bool(rj.get("success")), rj.get("first_success_step"), "have" if png else "none", png[0] if png else None,
            f"fresh full rerun 20261113 ({T})", f"file:{d}", f"file:{d}", extra=dict(anchor_id=u.get("anchor_id")))
FF = f"{SA}/f_family_20261114"                                      # F 族（11-14）：closed_loop1 = 主结果；同一目录里真实视频与终局帧
for e in json.load(open(f"{FF}/terminal_manifest.json")):
    if "/closed_loop1/" not in e["terminal_png"]:
        continue
    _, T, lv, name = e["unit_id"].split("/")
    d = os.path.dirname(e["terminal_png"])
    add("cosmos", T, lv, f"cosmos_f1114_libero10_{TT[T]}_{name}__{lv}", slot_ext(TID[T], name, lv),
        "have" if os.path.exists(f"{d}/real.mp4") else "none", f"{d}/real.mp4", bool(e["actual_success"]), e.get("first_success_step"),
        "have" if os.path.exists(e["terminal_png"]) else "none", e["terminal_png"], "F family closed_loop1 (20261114)", f"file:{d}", f"file:{d}",
        extra=dict(episode_end_env_step=e.get("episode_end_env_step")))


# ---------- UVA：真实视频；想象帧未存
def uva_dirs():
    for c in json.load(open(f"{LIB}/task2/uva_index.json")):
        lv = E.uva_level(c["family"], c["level"])
        if lv and c.get("closed_dir"):
            yield c["task"], lv, c["closed_dir"], "*/repeat0/result.json"
    for c in json.load(open(f"{LIB}/task2/uva_k_cells.json"))["cells"]:
        yield c["task"], c["level"], c["closed_dir"], "*/repeat0/result.json"
    for c in json.load(open(f"{LIB}/task2/uva_t01_cells.json"))["cells"]:
        yield c["task"], c["level"], c["closed_dir"], f"task{TID[c['task']]}_*/result.json"
    for i, x in {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}.items():
        yield "T4", f"Ef{i}", f"{UVA}/t4-friction-v6-closed-20261112/friction-v6/T4/E_friction_{x}/closed", "*/repeat0/result.json"
    for i in range(1, 10):                                           # v4：UVA T0 K（K-formal-v3）
        yield "T0", f"K{i}", f"{UVA}/t0-k-ef-20261115/K-formal-v3/output/K{i}/closed", "*/repeat0/result.json"


useen = {}
for T, lv, cd, pat in uva_dirs():
    for f in sorted(glob.glob(f"{cd}/{pat}")):
        u = os.path.basename(os.path.dirname(os.path.dirname(f))) if "repeat0" in f else os.path.basename(os.path.dirname(f))
        useen[(T, lv, u)] = f
for (T, lv, u), f in useen.items():
    r = json.load(open(f)); vid = f"{os.path.dirname(f)}/native-closed.mp4"
    uid = f"uva_libero10_{TT[T]}_{u}__{lv}"
    add("uva", T, lv, uid, slot_ext(TID[T], u, lv), "have" if os.path.exists(vid) else "none", vid, bool(r.get("native_success")),
        r.get("first_native_success_step"), "none", None, "UVA closed loop repeat0", f"rec:{uid}", None)

with open(f"{os.environ.get('WMV_INV_OUT', HERE)}/inventory.jsonl", "w") as fo:
    for o in OUT:
        fo.write(json.dumps(o, ensure_ascii=False) + "\n")
S = collections.Counter((o["model"], o["T"], o["machine"], o["real"], o["imag"], "complete" if o["complete"] else "-") for o in OUT)
mism = collections.Counter((o["model"], o["T"], o.get("imag_source")) for o in OUT if o["real"] != "none" and o["imag"] != "none" and not o["complete"])
json.dump(dict(n=len(OUT), by=[[*k, v] for k, v in sorted(S.items())], imag_present_but_other_run=[[*k, v] for k, v in sorted(mism.items(), key=str)],
               notes=dict(NOTE), h100_units=H100_SEEN), open(f"{os.environ.get('WMV_INV_OUT', HERE)}/inventory_summary.json", "w"), ensure_ascii=False, indent=1)
for k, v in sorted(S.items()):
    print(*k, v)
print("imag present but other run:", sorted(mism.items(), key=str))
print("notes", dict(NOTE))
print("units", len(OUT), "h100", H100_SEEN)
