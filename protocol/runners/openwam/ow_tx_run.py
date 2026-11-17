"""OpenWAM-alpha production units for T3 (#4) / T4 (#6) (user ruling 2026-11-02: OpenWAM runs the new tasks in full;
the T2 downgrade stands). Per unit: open-loop imagination (13 chained calls, H = 400) + truth replay x2 (bit pair) +
closed loop x2 (conclusion consistency). famO occludes the first real frame only (closed loop: call 0), ratio x the
256 raw edge, via phys/ow_famO.closed_occ / lingbot_famO.occlude. Verdict column stays None (no T3/T4 predicate yet).
Seeds: the shared masters (t4 = 20261111, t6 = 20261112), RandomState(master)[(anchor index, k)] - same ruler as
LingBot. Resumable; families in order base -> famO -> famP -> famA (waits for the bank file).

  python ow_tx_run.py <tid>   (openwam interpreter, GPU)  -> out/runs/openwam_t<tid>/records_<fam>.jsonl
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import numpy as np

ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
PROMPT = {2: "turn on the stove and put the moka pot on it",
          4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
          6: "put the white mug on the plate and put the chocolate pudding to the right of the plate"}
MASTER = {2: 20261104, 4: 20261111, 6: 20261112}                 # t2 reinstated (user ruling 2026-11-03)
RATIOS = {"O1": 1 / 16, "O2": 1 / 8, "O3": 3 / 16, "O4": 1 / 4, "O5": 5 / 16}
K = 10


def banks_of(tid):
    return {"base": f"{ROOT}/libero/out/bank/libero_anchors_t{tid}.json",
            "famO": f"{ROOT}/libero/out/bank/libero_anchors_t{tid}.json",
            "famP": f"{ROOT}/libero/out/bank/familyP_t{tid}/familyP_t{tid}_anchors.json",
            "famA": f"{ROOT}/libero/out/bank/familyA_t{tid}/familyA_t{tid}_anchors.json"}


def units_of(fam, bank):
    A = bank["anchors"]
    if fam == "base":
        return [(a, a["anchor_id"], None, i) for i, a in enumerate(A)]
    if fam == "famO":
        return [(a, a["anchor_id"], L, i) for i, a in enumerate(A) for L in RATIOS]
    bases = sorted({a["base_id"] for a in A})
    return [(a, a["base_id"], a["level"], bases.index(a["base_id"])) for a in A]


def main():
    tid = int(sys.argv[1])
    import ow_runner as W
    from lingbot_famO import occlude
    from ow_famO import closed_occ
    W._setup()
    eng, pre, arch, t_load, ckname = W._engine()
    sim = W.Sim(f"/dev/shm/ow_t{tid}_{os.getpid()}")
    out = f"{ROOT}/libero/out/runs/openwam_t{tid}"
    for sub in ("actions",):
        os.makedirs(f"{out}/{sub}", exist_ok=True)
    seeds = np.random.RandomState(MASTER[tid]).randint(0, 2 ** 31 - 1, size=(16, K))
    for fam in ("base", "famO", "famP", "famA"):
        bp = banks_of(tid)[fam]
        while not os.path.exists(bp):
            if fam != "famA":
                raise FileNotFoundError(bp)
            time.sleep(300)
        bank = json.load(open(bp))
        for a in bank["anchors"]:
            if a.get("bddl_file") and not a["bddl_file"].startswith("/"):
                a["bddl_file"] = os.path.join(ROOT, a["bddl_file"])
        path = f"{out}/records_{fam}.jsonl"
        done = {json.loads(l)["unit_id"] for l in open(path)} if os.path.exists(path) else set()
        for a, s0, L, ai in units_of(fam, bank):
            a = dict(a, resolution=256)
            for k in range(K):
                uid = f"openwam_{s0}__k{k}" + (f"__{L}" if L else "")
                if uid in done:
                    continue
                t0 = time.time(); seed = int(seeds[ai, k])
                o = sim.reset(a)
                if fam == "famO":
                    o = dict(o); o["agentview_image"] = occlude(o["agentview_image"], RATIOS[L])
                frames, env7, t_calls = W.imagine(eng, o, PROMPT[tid], seed)
                r1 = sim.replay(a, env7); r2 = sim.replay(a, env7)
                same = bool(np.array_equal(r1["states"], r2["states"]) and np.array_equal(r1["success"], r2["success"]))
                ratio = RATIOS.get(L) if fam == "famO" else None
                ca, cs, cS = closed_occ(eng, sim, a, PROMPT[tid], seed, ratio)
                ca2, cs2, _ = closed_occ(eng, sim, a, PROMPT[tid], seed, ratio)
                s1, s2 = bool(np.asarray(cs, bool).any()), bool(np.asarray(cs2, bool).any())
                succ = np.asarray(r1["success"], bool)
                np.savez_compressed(f"{out}/actions/{uid}.npz", env=np.asarray(env7, np.float32),
                                    closed_loop=np.asarray(ca, np.float32))
                rec = dict(unit_id=uid, model="openwam_alpha_sim_libero", task_id=tid, s0_id=s0, anchor_id=a["anchor_id"],
                           seed_index=k, imagination_seed=seed, seed_master=MASTER[tid], setting_id=L or "none",
                           family=fam, H_env=W.H, criterion="reach_anytime", checkpoint=ckname,
                           model_verdict=None, model_verdict_note="no deployed T3/T4 predicate yet: pending",
                           real_verdict=bool(succ.any()),
                           real_first_success_step=int(np.argmax(succ)) + 1 if succ.any() else None,
                           replay_check_identical=same,
                           closed_loop=dict(success=s1, first_success_step=int(np.argmax(np.asarray(cs, bool))) + 1 if s1 else None,
                                            rerun=dict(success=s2), conclusion_consistent=bool(s1 == s2)),
                           seconds=round(time.time() - t0, 1), machine=os.uname().nodename)
                open(path, "a").write(json.dumps(rec) + "\n")
                print(uid, "open", rec["real_verdict"], "closed", s1, s2, "same", same, rec["seconds"], flush=True)
        print("OW_FAM_DONE", f"t{tid}", fam, flush=True)
    sim.close()
    print("OW_TASK_DONE", f"t{tid}")


if __name__ == "__main__":
    main()
