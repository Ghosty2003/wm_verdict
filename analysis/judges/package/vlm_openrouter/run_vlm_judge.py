#!/usr/bin/env python3
"""VLM judge over the blind package via OpenRouter (qwen/qwen3-vl-235b-a22b-instruct). Stdlib only.

Prompt text is taken VERBATIM from ../prompt/judge_prompt.md (system prompt block + per-task title/criterion/rules),
so the VLM sees exactly the rubric the user reviewed. Each request = system prompt + task text + the task's
reference image (prompt/refs/<T>_reference_*.png) + the frame. Output: one JSON label per frame.

Key: read from env OPENROUTER_API_KEY only; never written anywhere. Results append to labels_qwen3vl235b.csv
(resumable: frame_ids already present are skipped). Raw responses (without the key) go to raw_responses.jsonl.

Usage:  python3 run_vlm_judge.py --limit 50          # pilot
        python3 run_vlm_judge.py --workers 16        # full run
"""
import argparse, base64, concurrent.futures as cf, csv, json, os, re, sys, threading, time, urllib.request, urllib.error
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
OUT = RAW = None  # set in main from --tag
LABELS = {"looks_success", "borderline", "not_success", "uncertain"}

def load_prompt():
    t = open(os.path.join(PKG, "prompt/judge_prompt.md"), encoding="utf-8").read()
    sys_block = re.search(r"## System prompt\s+```\n(.*?)```", t, re.S).group(1).strip()
    tasks = {}
    for m in re.finditer(r"### (T\d) \[(t\d)\] — (.*?)\nSUCCESS = (.*?)\n\nRules:\n(.*?)(?=\n### |\n## )", t, re.S):
        tasks[m.group(1)] = dict(title=m.group(3).strip(), crit=m.group(4).strip(), rules=m.group(5).strip())
    assert sys_block and len(tasks) == 5, (len(sys_block), list(tasks))
    return sys_block, tasks

UPSCALE = 0
MAX_TOKENS = 300
def b64(path, up=False):
    if up and UPSCALE:
        from PIL import Image; import io
        im = Image.open(path).convert("RGB"); k = max(1, -(-UPSCALE // min(im.size)))   # integer factor so the short side >= UPSCALE
        if k > 1: im = im.resize((im.width * k, im.height * k), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "PNG"); return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()

def ref_for(T):
    d = os.path.join(PKG, "prompt/refs"); f = [x for x in os.listdir(d) if x.startswith(f"{T}_reference")]
    return os.path.join(d, f[0])

NO_REF = False
def call(key, sys_block, task, T, frame_path, frame_id):
    if NO_REF:
        user_text = (f"Task {T}: {task['title']}\nSUCCESS = {task['crit']}\nRules for this task:\n{task['rules']}\n\n"
                     "The image is the frame to judge.\nLabel this imagined frame.")
        imgs = [b64(frame_path, up=True)]
    else:
        user_text = (f"Task {T}: {task['title']}\nSUCCESS = {task['crit']}\nRules for this task:\n{task['rules']}\n\n"
                     "The first image is the task reference (real simulator frames). The second image is the frame to judge.\n"
                     "Label this imagined frame.")
        imgs = [b64(ref_for(T)), b64(frame_path, up=True)]
    body = {"model": MODEL, "temperature": 0, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": sys_block},
                         {"role": "user", "content": [{"type": "text", "text": user_text}] +
                                                     [{"type": "image_url", "image_url": {"url": u}} for u in imgs]}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "X-Title": "wm-verdict-vlm-judge"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504): time.sleep(min(60, 2 ** attempt * 3)); continue
            return {"error": f"HTTP {e.code}", "detail": e.read().decode(errors="replace")[:500]}
        except Exception as e:
            time.sleep(min(60, 2 ** attempt * 3)); err = repr(e)
    return {"error": "retries_exhausted", "detail": err}

def parse(resp):
    try: msg = resp["choices"][0]["message"]
    except Exception: return None, None, "", ""
    txt = msg.get("content")
    if isinstance(txt, list):   # some providers return content parts
        txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
    if not txt:                 # reasoning models may leave content empty (e.g. ran out of tokens); look in the reasoning text
        txt = msg.get("reasoning") or ""
    m = re.search(r"\{[^{}]*\"label\"[^{}]*\}", txt, re.S) or re.search(r"\{.*\}", txt, re.S)
    if not m: return None, None, "", txt
    try:
        j = json.loads(m.group(0)); lab = str(j.get("label", "")).strip()
        return (lab if lab in LABELS else None), j.get("confidence"), str(j.get("note", ""))[:300], txt
    except Exception: return None, None, "", txt

def main():
    global MODEL, MAX_TOKENS
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="qwen3vl235b"); ap.add_argument("--ids", default=""); ap.add_argument("--upscale", type=int, default=0)
    ap.add_argument("--no-ref", action="store_true"); ap.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct"); ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--items", default=os.path.join(PKG, "items_blind.csv")); ap.add_argument("--frames-dir", default=os.path.join(PKG, "frames"))
    a = ap.parse_args(); global OUT, RAW, UPSCALE, NO_REF
    OUT = os.path.join(HERE, f"labels_{a.tag}.csv"); RAW = os.path.join(HERE, f"raw_responses_{a.tag}.jsonl"); UPSCALE = a.upscale; NO_REF = a.no_ref
    MODEL = a.model; MAX_TOKENS = a.max_tokens; key = os.environ.get("OPENROUTER_API_KEY")
    if not key: sys.exit("OPENROUTER_API_KEY not set (run: read -rsp 'key: ' OPENROUTER_API_KEY && export OPENROUTER_API_KEY)")
    sys_block, tasks = load_prompt()
    items = list(csv.DictReader(open(a.items, encoding="utf-8")))
    done = {r["frame_id"] for r in csv.DictReader(open(OUT, encoding="utf-8"))} if os.path.exists(OUT) else set()
    if a.ids:
        want = {l.strip() for l in open(a.ids) if l.strip()}; items = [r for r in items if r["frame_id"] in want]
    todo = [r for r in items if r["frame_id"] not in done]
    if a.limit:  # pilot: stratified first N per task
        per = max(1, a.limit // 5); seen = {}; sel = []
        for r in todo:
            if seen.get(r["task"], 0) < per: sel.append(r); seen[r["task"]] = seen.get(r["task"], 0) + 1
        todo = sel
    new = not os.path.exists(OUT); lock = threading.Lock()
    fo = open(OUT, "a", newline="", encoding="utf-8"); w = csv.writer(fo); fr = open(RAW, "a", encoding="utf-8")
    if new: w.writerow(["frame_id", "label", "confidence", "note"]); fo.flush()
    stats = {"ok": 0, "bad": 0, "in_tok": 0, "out_tok": 0, "cost": 0.0}; t0 = time.time()
    def one(r):
        resp = call(key, sys_block, tasks[r["task"]], r["task"], os.path.join(a.frames_dir, r["frame_id"] + ".png"), r["frame_id"])
        lab, conf, note, txt = parse(resp); u = resp.get("usage", {}) if isinstance(resp, dict) else {}
        with lock:
            fr.write(json.dumps({"frame_id": r["frame_id"], "resp_text": txt, "error": resp.get("error"), "usage": u,
                                 "model": resp.get("model"), "provider": resp.get("provider")}, ensure_ascii=False) + "\n"); fr.flush()
            if lab:
                w.writerow([r["frame_id"], lab, conf, note]); fo.flush(); stats["ok"] += 1
            else: stats["bad"] += 1
            stats["in_tok"] += u.get("prompt_tokens", 0) or 0; stats["out_tok"] += u.get("completion_tokens", 0) or 0
            stats["cost"] += float(u.get("cost", 0) or 0)
            n = stats["ok"] + stats["bad"]
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"{time.strftime('%H:%M:%S')} {n}/{len(todo)} ok={stats['ok']} bad={stats['bad']} "
                      f"{n/el:.2f}/s in_tok={stats['in_tok']} out_tok={stats['out_tok']} cost=${stats['cost']:.3f} "
                      f"eta={(len(todo)-n)/(n/el)/60:.1f}min", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, todo))
    print("DONE", stats, flush=True)

if __name__ == "__main__": main()
