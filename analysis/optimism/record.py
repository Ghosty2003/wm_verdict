"""记录一张看图板的代判结果（预筛乐观率）。
  python3 record.py <judge_name> <sheet_name> <labels> [notes_json]
labels：按板内编号 #1..#n 顺序的字母串，S=looks_success  B=borderline  N=not_success  U=uncertain，长度必须等于该板格数。
notes_json（可选）：{"3": "原因", ...}，只写有必要说明的格。
同一张板重复记录时以最后一次为准（汇总脚本按 sheet+slot 取最后一条）。"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = {"S": "looks_success", "B": "borderline", "N": "not_success", "U": "uncertain"}


def main():
    judge, sheet, labels = sys.argv[1], sys.argv[2], sys.argv[3].strip().upper()
    notes = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    sheets = json.load(open(f"{HERE}/{os.environ.get('PRESCREEN_SHEETS', 'sheets.json')}"))   # Cosmos 用 cosmos_sheets.json
    if sheet not in sheets:
        sys.exit(f"ERROR: unknown sheet {sheet}")
    items = sheets[sheet]
    if len(labels) != len(items) or any(c not in CODE for c in labels):
        sys.exit(f"ERROR: sheet {sheet} has {len(items)} slots; got {len(labels)} labels '{labels}' (use only S/B/N/U)")
    os.makedirs(f"{HERE}/judgments", exist_ok=True)
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    with open(f"{HERE}/judgments/{judge}.jsonl", "a") as f:
        for k, (item, c) in enumerate(zip(items, labels), 1):
            f.write(json.dumps(dict(judge=judge, sheet=sheet, slot=k, item=item, label=CODE[c], note=notes.get(str(k), ""), ts=ts),
                               ensure_ascii=False) + "\n")
    print(f"recorded {sheet}: {labels}")


if __name__ == "__main__":
    main()
