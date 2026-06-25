"""Look at the empty Sonnet responses to understand WHY they're empty.

Are they CAPI errors? Are they Sonnet refusals? Empty 200 OK? Or
something else? The fix differs.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# 1. Open the cache, find the empty entries, look at usage / latency / context.
p = REPO / "outputs" / "judge_cache.jsonl"
empties = []
non_empties = []
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    t = r["text"]
    if len(t.strip()) == 0:
        empties.append(r)
    else:
        non_empties.append(r)
print(f"cache: {len(empties)} empty / {len(non_empties)} non-empty")

# 2. Latency distribution for empties vs non-empties.
def stat(rs, label):
    lats = [float(r.get("latency_ms") or 0) for r in rs]
    if lats:
        print(f"  {label}: n={len(lats)} mean={sum(lats)/len(lats):.0f}ms "
              f"min={min(lats):.0f} max={max(lats):.0f}")
stat(empties, "empty   ")
stat(non_empties, "filled  ")

# 3. Usage tokens reported for empties vs non-empties.
for label, rs in [("empty", empties), ("filled", non_empties)]:
    usages = [r.get("usage") or {} for r in rs]
    prompts = [u.get("prompt_tokens") for u in usages if u.get("prompt_tokens") is not None]
    comps   = [u.get("completion_tokens") for u in usages if u.get("completion_tokens") is not None]
    print(f"  {label}: prompts reported={len(prompts)}/{len(rs)}  completions reported={len(comps)}/{len(rs)}")

# 4. Cross-reference empties to the source rows.
DATASET = REPO / "data" / "d3_transcript_cases.jsonl"
recs = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
# Rebuild cache keys for each test rec and see which ended up empty.
from data.judge_runner import JUDGE_SYSTEM, build_user_prompt, _cache_key
key_to_empty = {}
for r in empties:  key_to_empty[r["key"]] = "empty"
for r in non_empties: key_to_empty[r["key"]] = "filled"

# Also need wits rule ids per row -- pull from the prior shim output.
SHIM_IN = None
for f in (REPO / "outputs").glob("d3_wits_input_*.jsonl"):
    SHIM_IN = f
# Re-run shim to get rule ids matched to rows.
import subprocess, os
env = os.environ.copy()
env["WITS_DIST"] = os.environ.get("WITS_DIST", "c:/dev/what-in-the-shell-fresh/dist/index.cjs")
shim_input = "\n".join(json.dumps({"command": r["proposed_command"], "shell": r["shell"]}, ensure_ascii=False)
                       for r in recs if r.get("split") == "test") + "\n"
proc = subprocess.run(
    ["node", str(REPO / "data" / "_wits_score_shim.cjs")],
    input=shim_input,
    capture_output=True, text=True, env=env, check=False,
    encoding="utf-8", errors="replace",
)
wits_for_test = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
test_recs = [r for r in recs if r.get("split") == "test"]
assert len(wits_for_test) == len(test_recs)

# Now find which test rows mapped to empty cache entries.
hits = []
for rec, w in zip(test_recs, wits_for_test):
    analysis = {
        "rule_hits": [{"ruleId": rid, "severity": "info", "message": ""} for rid in w.get("rule_ids", [])],
        "effects":   [],
    }
    user_p = build_user_prompt(
        command=rec["proposed_command"], intention=rec.get("intention",""),
        transcript=rec["transcript"], analysis=analysis,
    )
    key = _cache_key("claude-sonnet-4.6", JUDGE_SYSTEM, user_p)
    status = key_to_empty.get(key, "(not cached)")
    hits.append((rec, status, w["verdict"], key, user_p))

# 5. Show stats per status, per truth label, per bucket.
print(f"\nstatus distribution: {dict(Counter(s for _, s, _, _, _ in hits))}")
for status in ("empty", "filled", "(not cached)"):
    sub = [h for h in hits if h[1] == status]
    if not sub: continue
    print(f"\n{status}: {len(sub)} rows")
    print(f"  truth: {dict(Counter(rec['decision'] for rec, _, _, _, _ in sub))}")
    print(f"  buckets: {dict(Counter(rec['report_bucket'] for rec, _, _, _, _ in sub).most_common(5))}")

# 6. Show 3 specific empty rows: case name + a small chunk of the prompt sent.
print("\n=== 3 EMPTY-RESPONSE EXAMPLES ===")
empty_rows = [h for h in hits if h[1] == "empty"]
for rec, status, wv, key, user_p in empty_rows[:3]:
    print(f"\n--- {rec['case_name']}  (truth={rec['decision']}, bucket={rec['report_bucket']}) ---")
    print(f"  command   : {rec['proposed_command'][:120]}")
    print(f"  intention : {rec.get('intention','')[:120]}")
    print(f"  WITS      : {wv}")
    print(f"  user prompt length: {len(user_p)} chars")
    print(f"  prompt tail (last 350 chars):")
    print("    " + user_p[-350:].replace("\n", "\n    "))

# 7. Direct retry of ONE empty case to see if it's repeatable or transient.
print("\n=== RETRY ONE EMPTY ROW (fresh CAPI call, bypass cache) ===")
if empty_rows:
    rec, status, wv, key, user_p = empty_rows[0]
    from data.judge_runner import _call_capi_sonnet
    import time
    t0 = time.perf_counter()
    try:
        resp = _call_capi_sonnet("claude-sonnet-4.6", JUDGE_SYSTEM, user_p, max_tokens=1024, temperature=0.0)
        text = resp["choices"][0]["message"]["content"]
        print(f"  fresh call took {(time.perf_counter()-t0)*1000:.0f}ms")
        print(f"  response length: {len(text)} chars")
        print(f"  response: {text[:600]!r}")
        print(f"  usage: {resp.get('usage')}")
    except Exception as e:
        print(f"  fresh call FAILED: {type(e).__name__}: {e}")
