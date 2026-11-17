# LingBot T2/T4/T6 原记录（文本）在 Dropbox 上的路径清单（2026-11-14）

- 范围：lingbot_t{2,4,6} 下 base / famA / famO / famP / famB_bridge 五条线的全部 jsonl（records / truth / closed 等），不含帧和 npz。
- Dropbox 根：`$DROPBOX_REMOTE:wm_verdict/libero/out/runs/`，下表路径相对于这个根。
- 核对：2026-11-14 用 rclone check（Dropbox 内容哈希）逐个比对集群 NAS 原件，56 / 56 一致，没有缺的，不需要补传。

| 路径 | 行数 | SHA256（前 16 位） |
|---|---|---|
| `lingbot_t2/base/closed.0.jsonl` | 30 | `2af80b0ac90f0e92` |
| `lingbot_t2/base/closed.1.jsonl` | 20 | `1e0ff0c016f2f088` |
| `lingbot_t2/base/records.0.jsonl` | 13 | `2945f02a844a2ebb` |
| `lingbot_t2/base/records.1.jsonl` | 13 | `0e81e1874fd2772d` |
| `lingbot_t2/base/records.2.jsonl` | 12 | `9d0398eceb715256` |
| `lingbot_t2/base/records.3.jsonl` | 12 | `956bbb6822bed86b` |
| `lingbot_t2/base/truth.0.jsonl` | 50 | `c7f636174e404160` |
| `lingbot_t2/famA/closed.0.jsonl` | 116 | `b8a8eaf2c47cd521` |
| `lingbot_t2/famA/closed.1.jsonl` | 134 | `ef91bc7d06346d76` |
| `lingbot_t2/famA/records.0.jsonl` | 63 | `520e501e0422151e` |
| `lingbot_t2/famA/records.1.jsonl` | 63 | `c030f76df57b411d` |
| `lingbot_t2/famA/records.2.jsonl` | 62 | `6748166549ee0bf5` |
| `lingbot_t2/famA/records.3.jsonl` | 62 | `8a5d9039f1355c81` |
| `lingbot_t2/famA/truth.0.jsonl` | 250 | `9c92afc24fa8ee93` |
| `lingbot_t2/famO/closed.0.jsonl` | 122 | `4aaf0ded2e223f0a` |
| `lingbot_t2/famO/closed.1.jsonl` | 128 | `bcbff83ff584bf8e` |
| `lingbot_t2/famO/closed_occ.0.jsonl` | 250 | `e61122ff80df49bd` |
| `lingbot_t2/famO/records.0.jsonl` | 63 | `04d4fa5232a2dc33` |
| `lingbot_t2/famO/records.1.jsonl` | 63 | `80a349ac8cf3dd2d` |
| `lingbot_t2/famO/records.2.jsonl` | 62 | `baf02e9b8562b4b2` |
| `lingbot_t2/famO/records.3.jsonl` | 62 | `4ef0d7991ac996c6` |
| `lingbot_t2/famO/truth.0.jsonl` | 250 | `6f8e2c9c141d147f` |
| `lingbot_t2/famP/closed.0.jsonl` | 116 | `294239c6c5133dcc` |
| `lingbot_t2/famP/closed.1.jsonl` | 134 | `359d23dec16432cc` |
| `lingbot_t2/famP/records.0.jsonl` | 63 | `716b4eddfe387c1a` |
| `lingbot_t2/famP/records.1.jsonl` | 63 | `b71b3af423e60720` |
| `lingbot_t2/famP/records.2.jsonl` | 62 | `bc9423e62b6c7722` |
| `lingbot_t2/famP/records.3.jsonl` | 62 | `bfe3c281f4cb2c70` |
| `lingbot_t2/famP/truth.0.jsonl` | 250 | `1db2ba7818e384da` |
| `lingbot_t2/famB_bridge/records.jsonl` | 50 | `0a27cff00072baf0` |
| `lingbot_t4/base/closed.0.jsonl` | 50 | `4bbd7b8ce4962de7` |
| `lingbot_t4/base/records.0.jsonl` | 50 | `bcd437e11e476c19` |
| `lingbot_t4/base/truth.0.jsonl` | 50 | `17d41f7dda17c4d2` |
| `lingbot_t4/famA/closed.0.jsonl` | 250 | `a26becb09560422e` |
| `lingbot_t4/famA/records.0.jsonl` | 250 | `37ce534f24008992` |
| `lingbot_t4/famA/truth.0.jsonl` | 250 | `2fbc13296e47fb0b` |
| `lingbot_t4/famO/closed.0.jsonl` | 250 | `e70420386d3c078f` |
| `lingbot_t4/famO/closed_occ.0.jsonl` | 250 | `df8dd47652262b8d` |
| `lingbot_t4/famO/records.0.jsonl` | 250 | `f669af4b9a54c6f1` |
| `lingbot_t4/famO/truth.0.jsonl` | 250 | `d99974b95c2d099d` |
| `lingbot_t4/famP/closed.0.jsonl` | 250 | `2be3bd88248be28b` |
| `lingbot_t4/famP/records.0.jsonl` | 250 | `f731524eb834e4da` |
| `lingbot_t4/famP/truth.0.jsonl` | 250 | `64d5dc7c3f7f5bbf` |
| `lingbot_t6/base/closed.0.jsonl` | 50 | `0cd3e73c5c2a18c8` |
| `lingbot_t6/base/records.0.jsonl` | 50 | `879d6c126f786f9b` |
| `lingbot_t6/base/truth.0.jsonl` | 50 | `a85585b576a4b557` |
| `lingbot_t6/famA/closed.0.jsonl` | 250 | `b279b840fd6b4777` |
| `lingbot_t6/famA/records.0.jsonl` | 250 | `381bf4c597aaf3ac` |
| `lingbot_t6/famA/truth.0.jsonl` | 250 | `5dccb278d8c6c11a` |
| `lingbot_t6/famO/closed.0.jsonl` | 250 | `d17a426d28f95345` |
| `lingbot_t6/famO/closed_occ.0.jsonl` | 250 | `5d5295e748012f69` |
| `lingbot_t6/famO/records.0.jsonl` | 250 | `134378bac35c9151` |
| `lingbot_t6/famO/truth.0.jsonl` | 250 | `b368653ae90fe42d` |
| `lingbot_t6/famP/closed.0.jsonl` | 250 | `2a3b38dd775ca015` |
| `lingbot_t6/famP/records.0.jsonl` | 250 | `fdb360a5968b5f3b` |
| `lingbot_t6/famP/truth.0.jsonl` | 250 | `e4a8d9d9ad1922e7` |
