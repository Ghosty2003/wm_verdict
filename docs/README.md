# WM-Verdict 闭环主数据（closed_main）

按论文任务编号（T0–T4）汇总两模型全部**闭环**记录；由 gen_closed_main.py 从 out/runs 原始分片自动合并，
原始镜像在 libero/out/runs/（本目录只是视图，勿手改）。每行一个 unit：
`closed_loop.success / first_success_step / rerun / conclusion_consistent` 为闭环主字段；
famB 文件的闭环字段在行内 `closed_loop`；逐块想象帧 npz 在 runs/ 对应 closed/ 子目录（2026-11-08 起新增）。

| 任务 | 文件 | 行数 |
|---|---|---|
| T0_t5 | lingbot_closed_EGNK.jsonl | 1850 |
| T0_t5 | openwam_closed_EGNK.jsonl | 1850 |
| T0_t5 | lingbot_famB.jsonl | 450 |
| T0_t5 | openwam_famB.jsonl | 500 |
| T0_t5 | lingbot_closed_APO_base.jsonl | 800 |
| T0_t5 | openwam_closed_APO_base.jsonl | 800 |
| T0_t5 | lingbot_famF.jsonl | 150 |
| T0_t5 | openwam_famF.jsonl | 150 |
| T1_t3 | lingbot_closed_EGNK.jsonl | 1850 |
| T1_t3 | openwam_closed_EGNK.jsonl | 1850 |
| T1_t3 | lingbot_famB.jsonl | 450 |
| T1_t3 | openwam_famB.jsonl | 500 |
| T1_t3 | lingbot_closed_APO_base.jsonl | 800 |
| T1_t3 | openwam_closed_APO_base.jsonl | 800 |
| T1_t3 | lingbot_famF.jsonl | 150 |
| T1_t3 | openwam_famF.jsonl | 150 |
| T2_t2 | lingbot_closed_EGN.jsonl | 1400 |
| T2_t2 | lingbot_closed_K.jsonl | 450 |
| T2_t2 | lingbot_closed_AP_base.jsonl | 550 |
| T2_t2 | lingbot_closed_famO_fix.jsonl | 250 |
| T2_t2 | lingbot_famB.jsonl | 450 |
| T2_t2 | openwam_closed_EGNK.jsonl | 1850 |
| T2_t2 | openwam_famB.jsonl | 500 |
| T2_t2 | openwam_closed_APO_base.jsonl | 800 |
| T2_t2 | lingbot_famF.jsonl | 150 |
| T2_t2 | openwam_famF.jsonl | 150 |
| T3_t4 | lingbot_closed_EGN.jsonl | 1400 |
| T3_t4 | lingbot_closed_K.jsonl | 450 |
| T3_t4 | lingbot_closed_AP_base.jsonl | 550 |
| T3_t4 | lingbot_closed_famO_fix.jsonl | 250 |
| T3_t4 | lingbot_famB.jsonl | 450 |
| T3_t4 | openwam_closed_EGNK.jsonl | 1850 |
| T3_t4 | openwam_famB.jsonl | 500 |
| T3_t4 | openwam_closed_APO_base.jsonl | 800 |
| T3_t4 | lingbot_famF.jsonl | 150 |
| T3_t4 | openwam_famF.jsonl | 150 |
| T4_t6 | lingbot_closed_EGN.jsonl | 1400 |
| T4_t6 | lingbot_closed_K.jsonl | 450 |
| T4_t6 | lingbot_closed_AP_base.jsonl | 550 |
| T4_t6 | lingbot_closed_famO_fix.jsonl | 250 |
| T4_t6 | lingbot_famB.jsonl | 450 |
| T4_t6 | openwam_closed_EGNK.jsonl | 1850 |
| T4_t6 | openwam_famB.jsonl | 500 |
| T4_t6 | openwam_closed_APO_base.jsonl | 800 |
| T4_t6 | lingbot_famF.jsonl | 150 |
| T4_t6 | openwam_famF.jsonl | 150 |

刷新时间见文件时间戳；在跑批次（OpenWAM t4/t6、famO 修正、famb_lb）每日自动更新。
