"""合并各模型抽样清单 → manifest_real.json（供 wm_verdict_web/tools/replace_round.py 入库）。只收已渲染出 PNG 的卡。"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TID = {"T0": "t5", "T1": "t3", "T2": "t2", "T3": "t4", "T4": "t6"}
GOAL = {
    "t5": "is the book inside the BACK compartment of the caddy?",
    "t3": "is the black bowl inside the bottom drawer AND the drawer fully closed? (A white gap line along the drawer front means it is NOT fully closed.)",
    "t2": "is the stove turned ON AND the moka pot standing on the stove?",
    "t4": "is the white mug on the LEFT plate AND the yellow-and-white mug on the RIGHT plate?",
    "t6": "is the white mug ON the plate AND the chocolate pudding to the RIGHT of the plate?",
}
HOW = ("Closed-loop deployment, REAL camera view (no imagination): 13 evenly spaced frames from the start of the episode "
       "to its final frame (yellow box, rightmost). Judge the boxed frame: ")
FAM = lambda lv: ("base" if lv == "none" else "E" if lv[:2] in ("Em", "Ef") else "N" if lv[:2] in ("N1", "NR", "NL") else
                  "B" if lv[:2] in ("Bg", "Bf", "Bz") or lv == "B0" else lv[0])


def main():
    sel = []
    for f in ("units_cosmos_uva.json", "units_ours.json"):
        if os.path.exists(f"{HERE}/{f}"):
            sel += json.load(open(f"{HERE}/{f}"))
    man, miss = [], 0
    for u in sel:
        png = f"{HERE}/cards/{u['unit_id']}.png"
        if not os.path.exists(png):
            miss += 1; continue
        t = TID[u["T"]]
        man.append(dict(model=u["model"], task=t, family=FAM(u["level"]), level=u["level"], unit_id=u["unit_id"],
                        png=png, sim_success=u["success"], sim_first_success_step=u["first_success_step"],
                        question=HOW + GOAL[t], labels=["looks_success", "borderline", "not_success", "uncertain"]))
    json.dump(man, open(f"{HERE}/manifest_real.json", "w"), ensure_ascii=False)
    print("manifest", len(man), "missing png", miss)


if __name__ == "__main__":
    main()
