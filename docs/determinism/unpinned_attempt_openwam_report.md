# H100 确定性核对（2026-11-16T04:06:42+0000）

H100：mhcl-h100-3 (NVIDIA H100 96GB, driver 560.35.05)；参照：cluster x86 RTX 5090 ($NODE_C/585/627)。同单元、同 seed、同脚本（H100 版只改选单元/输出路径/多存数据）。

## lingbot

- 单元数 0；完全一致 0；指令动作逐位一致 0；想象逐位一致 0
- 指令动作第一次不一致的步（1 起）：[]（最早 None，中位 None）
- 指令动作最大绝对差（各单元共同前缀内的最大值）：None
- 成败翻转 0；首次成功步变化 0

| 单元 | 族/档 | 指令动作首个不一致步 | 最大差 | states/画面首个不一致步 | 想象首个不一致块/调用 | x86 成败(步) | H100 成败(步) |
|---|---|---|---|---|---|---|---|

LingBot 的 states 列 = 逐步真实主视角画面首个不一致步（x86 参照没存 states）。H100 两遍之间是否逐位一致：

## openwam

- 单元数 13；完全一致 0；指令动作逐位一致 0；想象逐位一致 0
- 指令动作第一次不一致的步（1 起）：[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]（最早 1，中位 1.0）
- 指令动作最大绝对差（各单元共同前缀内的最大值）：2.0
- 成败翻转 0；首次成功步变化 6

| 单元 | 族/档 | 指令动作首个不一致步 | 最大差 | states/画面首个不一致步 | 想象首个不一致块/调用 | x86 成败(步) | H100 成败(步) |
|---|---|---|---|---|---|---|---|
| openwam_libero10_t5_a0__k0 | base/none | 1 | 0.0806 | 1 | 0 | True(160) | True(161) |
| openwam_libero10_t4_a0__k3 | base/none | 1 | 0.0412 | 1 | 0 | True(227) | True(227) |
| openwam_libero10_t6_a0__k0 | base/none | 1 | 2 | 1 | 0 | True(218) | True(216) |
| openwam_libero10_t4_fa0__k0__A1 | A/A1 | 1 | 2 | 1 | 0 | True(234) | True(226) |
| openwam_libero10_t6_fp0__k0__P1 | P/P1 | 1 | 2 | 1 | 0 | True(220) | True(220) |
| openwam_libero10_t4_a0__k3__O1 | O/O1 | 1 | 1.35 | 1 | 0 | True(226) | True(227) |
| openwam_libero10_t6_a0__k0__O4 | O/O4 | 1 | 1.86 | 1 | 0 | True(218) | True(317) |
| openwam_libero10_t2_a0__k0__Bg9 | B/Bg9 | 1 | 2 | 1 | 0 | False(None) | False(None) |
| openwam_libero10_t4_a2__k5__B0 | B/B0 | 1 | 1.99 | 1 | 0 | True(226) | True(226) |
| openwam_libero10_t4_a0__k1__K1 | K/K1 | 1 | 2 | 1 | 0 | False(None) | False(None) |
| openwam_libero10_t4_a0__k0__NL2 | N/NL2 | 1 | 0.0664 | 3 | 0 | True(247) | True(247) |
| openwam_libero10_t4_a0__k0__Em2 | E/Em2 | 1 | 1.98 | 1 | 0 | False(None) | False(None) |
| openwam_libero10_t6_a0__k1__G3 | G/G3 | 1 | 2 | 1 | 0 | True(260) | True(276) |
