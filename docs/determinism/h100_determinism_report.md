# H100 确定性核对（2026-11-16T04:29:54+0000）

H100：mhcl-h100-3 (NVIDIA H100 96GB, driver 560.35.05)；参照：cluster x86 RTX 5090 ($NODE_C/585/627)。同单元、同 seed、同脚本（H100 版只改选单元/输出路径/多存数据）。

## lingbot

- 单元数 15；完全一致 0；指令动作逐位一致 0；想象逐位一致 0
- 指令动作第一次不一致的步（1 起）：[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]（最早 1，中位 1.0）
- 指令动作最大绝对差（各单元共同前缀内的最大值）：2.04296875
- 成败翻转 1；首次成功步变化 11

| 单元 | 族/档 | 指令动作首个不一致步 | 最大差 | states/画面首个不一致步 | 想象首个不一致块/调用 | x86 成败(步) | H100 成败(步) |
|---|---|---|---|---|---|---|---|
| lingbot_libero10_t2_a0__k0 | base/none | 1 | 2.02 | 2 | 0 | True(255) | True(251) |
| lingbot_libero10_t4_a0__k2 | base/none | 1 | 2.02 | 2 | 0 | True(240) | True(242) |
| lingbot_libero10_t2_fa0__k0__A3 | famA/A3 | 1 | 2.04 | 2 | 0 | True(230) | False(None) |
| lingbot_libero10_t2_a0__k0__Ef1 | egn/Ef1 | 1 | 2.02 | 2 | 0 | True(252) | True(240) |
| lingbot_libero10_t4_a0__k0__G4 | egn/G4 | 1 | 2.03 | 2 | 0 | False(None) | False(None) |
| lingbot_libero10_t2_a0__k0__K1 | K/K1 | 1 | 2.02 | 2 | 0 | True(258) | True(239) |
| lingbot_libero10_t4_a0__k0__K1 | K/K1 | 1 | 2.04 | 2 | 0 | False(None) | False(None) |
| lingbot_libero10_t2_a0__k0__O2 | famO/O2 | 1 | 2.03 | 2 | 0 | True(265) | True(244) |
| lingbot_libero10_t2_fp0__k0__P1 | famP/P1 | 1 | 2.02 | 2 | 0 | True(261) | True(256) |
| lingbot_libero10_t6_a0__k0 | base/none | 1 | 2 | 2 | 0 | True(226) | True(235) |
| lingbot_libero10_t4_fp0__k0__P0 | famP/P0 | 1 | 2.04 | 2 | 0 | True(236) | True(234) |
| lingbot_libero10_t6_a0__k0__O4 | famO/O4 | 1 | 2.02 | 2 | 0 | True(233) | True(229) |
| lingbot_libero10_t6_a0__k0__NLw4 | egn/NLw4 | 1 | 2.04 | 34 | 0 | False(None) | False(None) |
| lingbot_libero10_t6_fa0__k5__A3 | famA/A3 | 1 | 1.99 | 2 | 0 | True(246) | True(242) |
| lingbot_libero10_t6_a0__k0__K3 | K/K3 | 1 | 2.02 | 2 | 0 | False(None) | False(None) |

LingBot 的 states 列 = 逐步真实主视角画面首个不一致步（x86 参照没存 states）。H100 两遍之间是否逐位一致：lingbot_libero10_t2_a0__k0=True, lingbot_libero10_t4_a0__k2=True, lingbot_libero10_t2_fa0__k0__A3=True, lingbot_libero10_t2_a0__k0__Ef1=True, lingbot_libero10_t4_a0__k0__G4=True, lingbot_libero10_t2_a0__k0__K1=True, lingbot_libero10_t4_a0__k0__K1=True, lingbot_libero10_t2_a0__k0__O2=True, lingbot_libero10_t2_fp0__k0__P1=True, lingbot_libero10_t6_a0__k0=True, lingbot_libero10_t4_fp0__k0__P0=True, lingbot_libero10_t6_a0__k0__O4=True, lingbot_libero10_t6_a0__k0__NLw4=True, lingbot_libero10_t6_fa0__k5__A3=True, lingbot_libero10_t6_a0__k0__K3=True

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

## 补充：发散起点与重复性

- 结论：物理仿真跨机器一致（OpenWAM 初始物理状态 13/13 逐位相同）；渲染几乎一致：T3/T4 初始画面逐位相同，T2（灶台/摩卡壶场景）腕部相机初始画面有 0.07% 像素差 1 个灰度级（主视角相同），属于不同 GPU 的 EGL 光栅化差异。主要分叉来自 GPU 架构不同导致的模型数值差：两个模型在第 1 次模型调用的输出就不同（输入画面完全相同的单元也一样），第 1 步指令动作差约 1e-3 量级，之后闭环把差异放大成不同轨迹；H100 自身完全可重复。
- H100 自身重复性：OpenWAM 13/13（绑核 vs 不绑核两次运行，commanded/states/想象帧/逐步画面全部逐位一致）；LingBot 两遍逐位一致 15/15。
- B 族腐蚀：H100 CPU（Xeon 8468）上 B 族腐蚀（LIBERO-plus gaussian_blur 等，numpy 1.22.4 / scipy 1.10.1 / skimage 0.21.0 同版本）与集群结果差 1 个灰度级、占 0.015% 像素（Bg9 第 0 次调用画布）；B0 与其他族的第 0 次调用画布逐位相同。
- 上表“最大差”是整条共同前缀里的最大值（2 左右 = 后面某步夹爪开/合符号相反）；第 1 步本身的差如下：

| 单元 | 第 1 步指令动作最大差 | 平均差 | 初始观测/画布相同 | 其他 |
|---|---|---|---|---|
| lingbot_libero10_t2_a0__k0 | 0.005804 | 0.00141 | False | 第 0 块想象 latent 最大差 0.199；x86 两遍一致 True |
| lingbot_libero10_t4_a0__k2 | 0.001831 | 0.000659 | True | 第 0 块想象 latent 最大差 0.188；x86 两遍一致 True |
| lingbot_libero10_t2_fa0__k0__A3 | 0.0006737 | 0.00021 | False | 第 0 块想象 latent 最大差 0.152；x86 两遍一致 True |
| lingbot_libero10_t2_a0__k0__Ef1 | 0.005804 | 0.00141 | False | 第 0 块想象 latent 最大差 0.199；x86 两遍一致 True |
| lingbot_libero10_t4_a0__k0__G4 | 0.001866 | 0.000685 | True | 第 0 块想象 latent 最大差 0.133；x86 两遍一致 True |
| lingbot_libero10_t2_a0__k0__K1 | 0.005804 | 0.00141 | False | 第 0 块想象 latent 最大差 0.199；x86 两遍一致 True |
| lingbot_libero10_t4_a0__k0__K1 | 0.001866 | 0.000685 | True | 第 0 块想象 latent 最大差 0.133；x86 两遍一致 True |
| lingbot_libero10_t2_a0__k0__O2 | 0.001522 | 0.000449 | False | 第 0 块想象 latent 最大差 0.176；x86 两遍一致 True |
| lingbot_libero10_t2_fp0__k0__P1 | 0.004146 | 0.00158 | False | 第 0 块想象 latent 最大差 0.582；x86 两遍一致 True |
| lingbot_libero10_t6_a0__k0 | 0.001522 | 0.000476 | True | 第 0 块想象 latent 最大差 0.266；x86 两遍一致 None |
| lingbot_libero10_t4_fp0__k0__P0 | 0.001037 | 0.000325 | True | 第 0 块想象 latent 最大差 0.129；x86 两遍一致 None |
| lingbot_libero10_t6_a0__k0__O4 | 0.001244 | 0.000331 | True | 第 0 块想象 latent 最大差 0.172；x86 两遍一致 None |
| lingbot_libero10_t6_a0__k0__NLw4 | 0.001522 | 0.000476 | True | 第 0 块想象 latent 最大差 0.266；x86 两遍一致 None |
| lingbot_libero10_t6_fa0__k5__A3 | 0.003906 | 0.000794 | True | 第 0 块想象 latent 最大差 0.217；x86 两遍一致 None |
| lingbot_libero10_t6_a0__k0__K3 | 0.001522 | 0.000476 | True | 第 0 块想象 latent 最大差 0.266；x86 两遍一致 None |
| openwam_libero10_t5_a0__k0 | 0.003872 | 0.00116 | True | 初始物理状态相同 True |
| openwam_libero10_t4_a0__k3 | 0.003906 | 0.000951 | True | 初始物理状态相同 True |
| openwam_libero10_t6_a0__k0 | 0.003906 | 0.00117 | True | 初始物理状态相同 True |
| openwam_libero10_t4_fa0__k0__A1 | 0.003922 | 0.00104 | True | 初始物理状态相同 True |
| openwam_libero10_t6_fp0__k0__P1 | 0.007324 | 0.00207 | True | 初始物理状态相同 True |
| openwam_libero10_t4_a0__k3__O1 | 0.001953 | 0.000427 | True | 初始物理状态相同 True |
| openwam_libero10_t6_a0__k0__O4 | 0.001968 | 0.000676 | True | 初始物理状态相同 True |
| openwam_libero10_t2_a0__k0__Bg9 | 0.00196 | 0.000838 | False | 初始物理状态相同 True |
| openwam_libero10_t4_a2__k5__B0 | 0.001831 | 0.000332 | True | 初始物理状态相同 True |
| openwam_libero10_t4_a0__k1__K1 | 0.001961 | 0.000717 | True | 初始物理状态相同 True |
| openwam_libero10_t4_a0__k0__NL2 | 0.003902 | 0.00131 | True | 初始物理状态相同 True |
| openwam_libero10_t4_a0__k0__Em2 | 0.003902 | 0.00131 | True | 初始物理状态相同 True |
| openwam_libero10_t6_a0__k1__G3 | 0.003906 | 0.0012 | True | 初始物理状态相同 True |

成败翻转的单元：lingbot_libero10_t2_fa0__k0__A3（x86 成功于第 230 步 → H100 失败）。生产照常进行，每个单元行里 machine = H100，可按机器分开分析。
