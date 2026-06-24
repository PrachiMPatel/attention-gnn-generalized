"""Audit data/wits_eval_cases.jsonl for gaps and coverage issues.

Prints:
  - per-class counts and per-class share
  - per-shell × per-class crosstab (powershell coverage of each class?)
  - per-source-bucket × per-class crosstab
  - lexical features per class (token counts, char length stats)
  - rule-of-thumb attack-surface coverage table (which attack families
    have any unsafe/extremely_unsafe examples? which only have safe?)
  - duplicate detection (near-dup by command stem)
  - smallest classes: list every record so we can eyeball balance
  - "labeled `safe` but contains attack-shaped substrings" — possible
    polluted-safe rows still left from the WITS test extraction.

Read-only; does not modify any data files.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "wits_eval_cases.jsonl"

records = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"Loaded {len(records)} records from {DATA.name}")

LABELS = ["safe", "maybe_safe", "unsafe", "extremely_unsafe"]


def head(title: str, width: int = 78) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


# ---------------------------------------------------------------------------
head("1. Per-class counts")
# ---------------------------------------------------------------------------
by_v = Counter(r["verdict"] for r in records)
total = sum(by_v.values())
for v in LABELS:
    n = by_v.get(v, 0)
    print(f"  {v:<20s} {n:4d}  ({100*n/total:5.1f}%)")


# ---------------------------------------------------------------------------
head("2. Shell × class crosstab")
# ---------------------------------------------------------------------------
ct = defaultdict(lambda: Counter())
for r in records:
    ct[r["shell"]][r["verdict"]] += 1
print(f"  {'shell':<12s} " + " ".join(f"{v:>17s}" for v in LABELS) + "   total")
for sh, c in ct.items():
    row = " ".join(f"{c.get(v,0):17d}" for v in LABELS)
    print(f"  {sh:<12s} {row}   {sum(c.values())}")


# ---------------------------------------------------------------------------
head("3. Source bucket × class")
# ---------------------------------------------------------------------------

def bucket_of(r):
    src = r.get("source", "")
    if src.startswith("reviewer:"):
        # collapse e.g. "reviewer:rc6-env-prefix-aug" -> "reviewer:rc6-env-prefix"
        tail = src.split(":", 1)[1]
        tail = re.sub(r"-aug$", "", tail)
        return f"reviewer:{tail}"
    if src.startswith("car:") or src.startswith("wits:"):
        # collapse to last path segment
        last = src.split("/")[-1] if "/" in src else src.split(":", 1)[-1]
        return f"wits-tests:{last}"
    return src or "(none)"


by_bucket = defaultdict(lambda: Counter())
for r in records:
    by_bucket[bucket_of(r)][r["verdict"]] += 1

print(f"  {'bucket':<48s} " + " ".join(f"{v:>5s}" for v in ["safe","maybe","unsf","ext"]) + "  total")
for b, c in sorted(by_bucket.items(), key=lambda kv: -sum(kv[1].values())):
    row = " ".join(f"{c.get(v,0):5d}" for v in LABELS)
    print(f"  {b:<48s} {row}  {sum(c.values()):5d}")


# ---------------------------------------------------------------------------
head("4. Length distribution per class (chars)")
# ---------------------------------------------------------------------------
import statistics
for v in LABELS:
    lens = [len(r["command"]) for r in records if r["verdict"] == v]
    if not lens:
        print(f"  {v:<20s} EMPTY")
        continue
    print(f"  {v:<20s} n={len(lens):4d}  min={min(lens):3d}  med={int(statistics.median(lens)):3d}"
          f"  mean={statistics.mean(lens):5.1f}  max={max(lens):4d}")


# ---------------------------------------------------------------------------
head("5. Top first-tokens per class (most frequent 'argv0')")
# ---------------------------------------------------------------------------
def argv0(cmd: str) -> str:
    # Strip leading env-prefix assignments.
    parts = cmd.strip().split()
    i = 0
    while i < len(parts) and re.match(r"^[A-Z_][A-Z0-9_]*=", parts[i]):
        i += 1
    return parts[i] if i < len(parts) else ""


for v in LABELS:
    c = Counter(argv0(r["command"]) for r in records if r["verdict"] == v)
    top = ", ".join(f"{k}({n})" for k, n in c.most_common(10))
    print(f"  {v:<20s} {top}")


# ---------------------------------------------------------------------------
head("6. Attack-surface coverage check")
# ---------------------------------------------------------------------------
# For each attack pattern: count cases (any verdict) and unsafe/extreme cases.
PATTERNS = [
    ("env-prefix RCE",           r"^[A-Z_]+=\S+\s+\S"),
    ("env-wrapper",              r"^env\s+[A-Z_]+="),
    ("git -c key=val",           r"^git\s+-c\s+\S+="),
    ("find -exec",               r"\bfind\b.*\b-exec(dir)?\b"),
    ("find -ok",                 r"\bfind\b.*\b-ok\b"),
    ("sed e/w flag",             r"\bsed\b.*'[^']*[eEwWR][^']*'"),
    ("awk system/pipe-to-shell", r"\bawk\b.*(system|\|\s*\")"),
    ("python -m pip",            r"\bpython3?\s+-m\s+pip\b"),
    ("npx code-runner",          r"^npx\s+(tsx|ts-node|concurrently)\b"),
    ("gh api DELETE",            r"\bgh\s+api\b.*\b(DELETE|PATCH|PUT)\b"),
    ("gh api -X glued",          r"\bgh\s+api\b.*-X(DELETE|PATCH|PUT)"),
    ("curl -X DELETE",           r"\bcurl\b.*-X\s*(DELETE|PATCH|PUT)"),
    ("rm -rf",                   r"\brm\s+-[rRf]+\b"),
    ("dd of=",                   r"\bdd\s+.*\bof="),
    ("chmod 777",                r"\bchmod\b\s+(-R\s+)?[0-7]?77[7]?"),
    ("curl | sh",                r"\bcurl\b.*\|\s*(ba)?sh\b"),
    ("wget | sh",                r"\bwget\b.*\|\s*(ba)?sh\b"),
    ("sudo",                     r"^sudo\b"),
    ("kubectl delete",           r"\bkubectl\s+delete\b"),
    ("kubectl patch",            r"\bkubectl\s+patch\b"),
    ("aws iam attach",           r"\baws\s+iam\s+attach"),
    ("aws s3 put public ACL",    r"\baws\s+s3(api)?\s+put-bucket-acl"),
    ("aws delete-",              r"\baws\s+\S+\s+delete-"),
    ("az * keys",                r"\baz\s+\S.*\skeys\b"),
    ("az * firewall-rule",       r"\baz\s+.*firewall-rule"),
    ("az role assignment",       r"\baz\s+role\s+assignment\s+create"),
    ("base64 -d | sh",           r"base64\b.*\|\s*(ba)?sh\b"),
    ("eval $",                   r"\beval\s+\S"),
    ("ssh user@host",            r"^ssh\s+\S+@"),
    ("scp file user@",           r"^scp\b.*\s+\S+@"),
    ("rsync remote",             r"^rsync\b.*\s+\S+@"),
    ("redirect to /etc",         r"(>|>>)\s*\"?/etc/"),
    ("redirect to /root",        r"(>|>>)\s*\"?/root/"),
    ("redirect to ~/.ssh",       r"(>|>>)\s*\"?~?/?\.ssh/"),
    ("&> redirect",              r"&>+\s*/"),
    ("docker run --privileged",  r"\bdocker\s+run\b.*--privileged"),
    ("docker exec",              r"\bdocker\s+exec\b"),
    ("PowerShell iex/irm",       r"\b(iex|irm|Invoke-Expression|Invoke-RestMethod)\b"),
    ("PowerShell -enc",          r"-e(nc(odedcommand)?)?\b\s+[A-Za-z0-9+/=]{40,}"),
    ("PowerShell Set-Execution", r"Set-ExecutionPolicy"),
    ("git push --force",         r"\bgit\s+push\b.*--force"),
    ("git reset --hard",         r"\bgit\s+reset\b\s+--hard"),
    ("git clean -fdx",           r"\bgit\s+clean\b.*-fd"),
    ("history -c / shred",       r"^(history\s+-c|shred\b)"),
    ("/dev/sd or /dev/nvme",     r"/dev/(sd|nvme|hd|xvd)"),
    ("crontab -e/-l",            r"\bcrontab\b\s+(-[el])"),
    ("systemctl",                r"\bsystemctl\b"),
    ("iptables",                 r"\biptables\b"),
    ("npm publish",              r"\bnpm\s+publish\b"),
    ("pip uninstall -y",         r"\bpip\s+uninstall\b.*-y"),
]

print(f"  {'pattern':<30s} {'total':>6s} {'safe':>5s} {'maybe':>5s} {'unsf':>5s} {'ext':>5s}   {'unsafe-share':>12s}")
gaps = []
for name, pat in PATTERNS:
    rx = re.compile(pat)
    hits = [r for r in records if rx.search(r["command"])]
    c = Counter(r["verdict"] for r in hits)
    danger = c.get("unsafe", 0) + c.get("extremely_unsafe", 0)
    total_h = sum(c.values())
    share = (danger / total_h) if total_h else 0.0
    print(f"  {name:<30s} {total_h:6d} {c.get('safe',0):5d} {c.get('maybe_safe',0):5d}"
          f" {c.get('unsafe',0):5d} {c.get('extremely_unsafe',0):5d}   {share*100:11.1f}%")
    if total_h < 3:
        gaps.append((name, total_h, "few-or-no"))
    elif danger == 0 and total_h > 0:
        gaps.append((name, total_h, "no-danger-examples"))

print("\nGaps flagged for review:")
for name, n, why in gaps:
    print(f"  - {name:<30s} n={n}  ({why})")


# ---------------------------------------------------------------------------
head("7. Duplicate/near-duplicate detection")
# ---------------------------------------------------------------------------
def normalize(cmd: str) -> str:
    cmd = cmd.strip().lower()
    # squash whitespace
    cmd = re.sub(r"\s+", " ", cmd)
    return cmd

norm_to_records = defaultdict(list)
for r in records:
    norm_to_records[normalize(r["command"])].append(r)
collisions = [(k, v) for k, v in norm_to_records.items() if len(v) > 1]
print(f"  {len(collisions)} exact-after-normalize collisions (different shells / sources):")
for k, v in collisions[:10]:
    verdicts = ", ".join(f"{r['verdict']}({r.get('shell','?')})" for r in v)
    print(f"    {k[:80]!r}   ->  {verdicts}")
if len(collisions) > 10:
    print(f"    ... and {len(collisions) - 10} more")


# ---------------------------------------------------------------------------
head("8. 'safe' rows that contain attack-shaped substrings (sanity check)")
# ---------------------------------------------------------------------------
RED_FLAGS = [
    ("env-prefix",       re.compile(r"^[A-Z_]+=")),
    ("git -c key=",      re.compile(r"^git\s+-c\s+\S+=")),
    ("find -exec",       re.compile(r"\bfind\b.*\b-exec(dir)?\b")),
    ("sed e/w flag",     re.compile(r"\bsed\b.*'[^']*[eEwWR][^']*'")),
    ("awk pipe-to-sh",   re.compile(r"\bawk\b.*\|\s*\"?(/?bin/)?(ba)?sh\"?")),
    ("curl | sh",        re.compile(r"\bcurl\b.*\|\s*(ba)?sh\b")),
    ("rm -rf /",         re.compile(r"\brm\s+-[rRf]+\s+/(\s|$)")),
]
suspicious = []
for r in records:
    if r["verdict"] != "safe":
        continue
    for name, rx in RED_FLAGS:
        if rx.search(r["command"]):
            suspicious.append((name, r))
            break
print(f"  {len(suspicious)} `safe` rows match attack-shaped patterns (manual review needed):")
for name, r in suspicious[:30]:
    print(f"    [{name}] {r['command'][:100]!r}  src={r.get('source','')}")
if len(suspicious) > 30:
    print(f"    ... and {len(suspicious) - 30} more")


# ---------------------------------------------------------------------------
head("9. Per-class samples (smallest classes get listed in full)")
# ---------------------------------------------------------------------------
for v in LABELS:
    sub = [r for r in records if r["verdict"] == v]
    print(f"\n  -- {v} (n={len(sub)}) --")
    if len(sub) <= 25:
        for r in sub:
            print(f"    [{r.get('shell','?')}] {r['command'][:110]!r}   src={r.get('source','')[:30]}")
    else:
        # Show a representative sample: 5 head, 5 tail, 5 random middle.
        import random
        rng = random.Random(0)
        idxs = list(range(len(sub)))
        rng.shuffle(idxs)
        sample = sub[:5] + [sub[i] for i in idxs[:10]] + sub[-5:]
        for r in sample:
            print(f"    [{r.get('shell','?')}] {r['command'][:110]!r}   src={r.get('source','')[:30]}")


# ---------------------------------------------------------------------------
head("10. Summary recommendations")
# ---------------------------------------------------------------------------
print(f"""
- total cases   : {len(records)}
- class shares  : safe={by_v.get('safe',0)/total:.0%}, maybe_safe={by_v.get('maybe_safe',0)/total:.0%}, unsafe={by_v.get('unsafe',0)/total:.0%}, extremely_unsafe={by_v.get('extremely_unsafe',0)/total:.0%}
- shells        : bash={sum(1 for r in records if r['shell']=='bash')}, powershell={sum(1 for r in records if r['shell']=='powershell')}
- collisions    : {len(collisions)} exact-after-normalize
- suspicious safe rows: {len(suspicious)} (see section 8)
- attack-surface gaps: {len(gaps)} patterns short on examples / lacking danger labels (see section 6)
""")
