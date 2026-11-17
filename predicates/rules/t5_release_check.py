"""t5 "seated but not released" by simulator semantics (2026-10-21 ruling 2).

Question: if the state drawn by the imagined endpoints (book inside the caddy's back compartment, gripper still closed
on it) really happened, would the simulator report success? Read-only: bit-exact replays of the stored truth (a)
actions of every t5 unit of lingbot_famA / lingbot_famP, and per step LIBERO's own pieces of the goal predicate
In(black_book_1, desk_caddy_1_back_contain_region) = region.check_contact(book) and region.check_contain(book),
plus the robosuite grasp test (both finger pads in contact with the book; env._check_grasp).

Frames of interest: contain and grasp (book inside the compartment volume while still held). Reported: how many such
frames / units, and the simulator success flag on them (overall, and split by region contact).

  python t5_release_check.py     -> out/content_review/t5_release_check.json
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np

ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from libero_env_adapter import _make_env, _anchor

RUNS = {"famA": ("lingbot_famA", "familyA"), "famP": ("lingbot_famP", "familyP")}
BOOK, REGION = "black_book_1", "desk_caddy_1_back_contain_region"


def step_state(u):
    p = u._env.env
    book, reg = p.object_states_dict[BOOK], p.object_states_dict[REGION]
    return (bool(reg.check_contain(book)), bool(reg.check_contact(book)),
            bool(p._check_grasp(gripper=p.robots[0].gripper, object_geoms=p.objects_dict[BOOK])),
            bool(u._env.check_success()))


def main():
    rows, agg = [], np.zeros((2, 2, 2, 2), int)     # [contain, contact, grasp, success] frame counts
    for fam, (run, bank) in RUNS.items():
        for l in open(f"{ROOT}/libero/out/runs/{run}/records_t5.jsonl"):
            r = json.loads(l)
            a = _anchor(f"{ROOT}/libero/out/bank/{bank}/{bank}_anchors.json", r["anchor_id"])
            acts = np.load(f"{ROOT}/{r['action_file']}")["env"]
            tr = np.load(f"{ROOT}/{r['trajectory_file']}")
            env = _make_env(a, H=len(acts)); u = env.envs[0]
            S = []
            for x in acts:
                u.step(np.asarray(x, np.float32)); S.append(step_state(u))
            env.close()
            S = np.array(S)
            assert np.array_equal(S[:, 3], tr["real_success"]), r["unit_id"]     # same trajectory as the record
            for s in S:
                agg[tuple(int(v) for v in s)] += 1
            held_in = S[:, 0] & S[:, 2]
            rows.append(dict(unit=r["unit_id"], setting=r["setting_id"], real=bool(r["real_verdict"]),
                             frames_contain_and_grasp=int(held_in.sum()),
                             success_on_those=int((held_in & S[:, 3]).sum()),
                             frames_contain_grasp_contact=int((held_in & S[:, 1]).sum()),
                             success_on_contain_grasp_contact=int((held_in & S[:, 1] & S[:, 3]).sum()),
                             success_while_grasped=int((S[:, 2] & S[:, 3]).sum())))
            print(rows[-1], flush=True)

    def frac(n, d):
        return None if d == 0 else round(n / d, 4)
    cg = agg[1, :, 1, :]; cgc = agg[1, 1, 1, :]; cgn = agg[1, 0, 1, :]
    out = dict(
        units=len(rows), frames=int(agg.sum()),
        contain_and_grasp=dict(frames=int(cg.sum()), success_frames=int(cg[:, 1].sum()), success_rate=frac(cg[:, 1].sum(), cg.sum()),
                               units_with_any=sum(r["frames_contain_and_grasp"] > 0 for r in rows)),
        contain_grasp_with_region_contact=dict(frames=int(cgc.sum()), success_frames=int(cgc[1]), success_rate=frac(cgc[1], cgc.sum())),
        contain_grasp_without_region_contact=dict(frames=int(cgn.sum()), success_frames=int(cgn[1]), success_rate=frac(cgn[1], cgn.sum())),
        success_frames_while_grasped=int(agg[:, :, 1, 1].sum()), success_frames_total=int(agg[..., 1].sum()),
        joint_counts={f"contain{c}_contact{k}_grasp{g}_success{s}": int(agg[c, k, g, s])
                      for c in (0, 1) for k in (0, 1) for g in (0, 1) for s in (0, 1)},
        per_unit=rows)
    json.dump(out, open(f"{ROOT}/libero/out/content_review/t5_release_check.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_unit"}, indent=1))


if __name__ == "__main__":
    main()
