# 豁免登记（LIBERO）

逐位确定性只是配对轨（想象、真值重放）的硬停点。闭环参照列只要求两遍成功结论一致（2026-10-25 修订；2026-10-24 的 ±10 步容差已 superseded），首次成功步差作为测量噪声计入 verify check 9 的分布；逐位不一致照实登记于此。

| 编号 | unit | 范围 | 内容 | 依据 |
|---|---|---|---|---|
| #1 | openwam_libero10_t3_fa0__k2__A0 | closed-loop reference column | bit-level mismatch, success verdict consistent (run 1 first success 228; reruns [229, 229]) | ruling 2026-10-24 |
| #2 | openwam_libero10_t3_fa2__k3__A0 | closed-loop reference column | bit-level mismatch, success verdict consistent (run 1 first success 233; reruns [233, 233]) | ruling 2026-10-24 |
| #3 | openwam_libero10_t3_fa2__k7__A0 | closed-loop reference column | bit-level mismatch, success verdict consistent (run 1 first success 234; reruns [234, 234]) | ruling 2026-10-24 |
| #T2B | lingbot_libero10_t2_*__B* (9 档 × 50) | LingBot B 族 T2 | T2 B 族跨机器，经锚点桥验证：基线想象原在 A100（$NODE_CLOUD），B 族在 RTX 5090 运行；5090 重跑 T2 无扰动 50 单元开环真值成功 41/50（Wilson 95% 0.69–0.90），A100 原记录 40/50（0.67–0.89），区间重叠；sham 逐位/首步一致门由此桥替代；B 健康检查 5090 复跑与存档逐位一致 | 2026-11-10 决定 2 |
