"""生成 Dropbox 闭环主数据视图 closed_main/（方案 A，2026-11-08）。
原镜像不动；本脚本把闭环 jsonl 按论文任务编号合并到 out/closed_main/，再由调用方 rclone 同步。
可重复运行（整目录重建）。"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob
import json
import os
import shutil

RUNS = _wmvp.P("$WMV_ROOT/libero/out/runs")
DST = _wmvp.P("$WMV_ROOT/libero/out/closed_main")

# (论文任务目录, 输出文件名, 源 glob 列表)
SPEC = {
    "T0_t5": [
        ("lingbot_closed_EGNK.jsonl", ["lingbot_phys/closed_t5*.jsonl"]),
        ("openwam_closed_EGNK.jsonl", ["openwam_phys/closed_t5*.jsonl"]),
        ("lingbot_famB.jsonl", ["lingbot_famB/records_t5*.jsonl"]),
        ("openwam_famB.jsonl", ["openwam_famB/records_t5*.jsonl"]),
        ("lingbot_closed_APO_base.jsonl", ["lingbot_base/records_t5*.jsonl", "lingbot_famA/records_t5*.jsonl",
                                           "lingbot_famP/records_t5*.jsonl", "lingbot_famO/records_t5*.jsonl"]),
        ("openwam_closed_APO_base.jsonl", ["openwam_base/records_t5*.jsonl", "openwam_famA/records_t5*.jsonl",
                                           "openwam_famP/records_t5*.jsonl", "openwam_famO/records_t5*.jsonl"]),
    ],
    "T1_t3": [
        ("lingbot_closed_EGNK.jsonl", ["lingbot_phys/closed_t3*.jsonl"]),
        ("openwam_closed_EGNK.jsonl", ["openwam_phys/closed_t3*.jsonl"]),
        ("lingbot_famB.jsonl", ["lingbot_famB/records_t3*.jsonl"]),
        ("openwam_famB.jsonl", ["openwam_famB/records_t3*.jsonl"]),
        ("lingbot_closed_APO_base.jsonl", ["lingbot_base/records_t3*.jsonl", "lingbot_famA/records_t3*.jsonl",
                                           "lingbot_famP/records_t3*.jsonl", "lingbot_famO/records_t3*.jsonl"]),
        ("openwam_closed_APO_base.jsonl", ["openwam_base/records_t3*.jsonl", "openwam_famA/records_t3*.jsonl",
                                           "openwam_famP/records_t3*.jsonl", "openwam_famO/records_t3*.jsonl"]),
    ],
}
for tid, T in ((2, "T2_t2"), (4, "T3_t4"), (6, "T4_t6")):
    SPEC[T] = [
        ("lingbot_closed_EGN.jsonl", [f"lingbot_t{tid}/phys/closed_egn.*.jsonl"]),
        ("lingbot_closed_K.jsonl", [f"lingbot_t{tid}/phys/closed_K.*.jsonl"]),
        # famO 旧闭环（漏遮挡，2026-11-05 作废）不并入；O 族闭环只认 closed_occ 修正批
        ("lingbot_closed_AP_base.jsonl", [f"lingbot_t{tid}/base/closed.*.jsonl", f"lingbot_t{tid}/famA/closed*.jsonl",
                                          f"lingbot_t{tid}/famP/closed*.jsonl"]),
        ("lingbot_closed_famO_fix.jsonl", [f"lingbot_t{tid}/**/closed_occ.*.jsonl"]),
        ("lingbot_famB.jsonl", [f"lingbot_t{tid}/famB/records*.jsonl"]),
        ("openwam_closed_EGNK.jsonl", [f"openwam_t{tid}/phys/closed.*.jsonl"]),   # t2 closed.jsonl = duplicate base batch, base comes from records_base
        ("openwam_famB.jsonl", [f"openwam_t{tid}/famB/records*.jsonl"]),
        ("openwam_closed_APO_base.jsonl", [f"openwam_t{tid}/records_base.jsonl", f"openwam_t{tid}/records_famA.jsonl",
                                           f"openwam_t{tid}/records_famP.jsonl", f"openwam_t{tid}/records_famO.jsonl"]),
    ]


# F 族（单次全帧置黑，信息 × 一次性；2026-11-13 批准正式跑，H=400 原生，无需截断）
for T, tid in (("T0_t5", 5), ("T1_t3", 3), ("T2_t2", 2), ("T3_t4", 4), ("T4_t6", 6)):
    SPEC[T] += [("lingbot_famF.jsonl", [f"lingbot_famF/closed_t{tid}.*.jsonl"]),
                ("openwam_famF.jsonl", [f"openwam_famF/closed_t{tid}.*.jsonl"])]


def h400(line):
    """eval_design §28（spec revision #7）：H=520 批次按 H=400 事后截断为正式口径；原闭环结果保留在 closed_loop_h520。"""
    r = json.loads(line)
    cl = r.get("closed_loop")
    if r.get("H_env") == 520 and isinstance(cl, dict) and cl.get("success") is not None:
        fs = cl.get("first_success_step")
        ok = fs is not None and fs <= 400
        r["closed_loop_h520"] = cl
        r["closed_loop"] = dict(cl, success=ok, first_success_step=fs if ok else None, truncated_to_H=400)
        return json.dumps(r)
    return line


def main():
    for d in [x for x in os.listdir(DST) if x.startswith("T")] if os.path.isdir(DST) else []:
        shutil.rmtree(f"{DST}/{d}", ignore_errors=True)   # keep snapshot files at top level
    lines = ["# WM-Verdict 闭环主数据（closed_main）",
             "",
             "按论文任务编号（T0–T4）汇总两模型全部**闭环**记录；由 gen_closed_main.py 从 out/runs 原始分片自动合并，",
             "原始镜像在 libero/out/runs/（本目录只是视图，勿手改）。每行一个 unit：",
             "`closed_loop.success / first_success_step / rerun / conclusion_consistent` 为闭环主字段；",
             "famB 文件的闭环字段在行内 `closed_loop`；逐块想象帧 npz 在 runs/ 对应 closed/ 子目录（2026-11-08 起新增）。",
             "", "| 任务 | 文件 | 行数 |", "|---|---|---|"]
    for T, files in SPEC.items():
        os.makedirs(f"{DST}/{T}", exist_ok=True)
        for name, pats in files:
            rows = []
            for p in pats:
                for f in sorted(glob.glob(f"{RUNS}/{p}", recursive=True)):
                    for line in open(f).read().splitlines():
                        rows.append(h400(line))
            if rows:
                open(f"{DST}/{T}/{name}", "w").write("\n".join(rows) + "\n")
            lines.append(f"| {T} | {name} | {len(rows)}{'' if rows else '（待出）'} |")
    lines += ["", f"刷新时间见文件时间戳；在跑批次（OpenWAM t4/t6、famO 修正、famb_lb）每日自动更新。"]
    open(f"{DST}/README.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[7:]))


if __name__ == "__main__":
    main()
