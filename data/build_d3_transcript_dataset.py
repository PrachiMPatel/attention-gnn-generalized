"""Phase 7 — build the D3 transcript-aware 3-class dataset.

Loads every D3 eval case from c:/dev/what-in-the-shell-fresh/eval_cases/*.json,
renders the conversation up to the target permission_prompt into a compact
transcript string (mirrors src/hooks/auto-approve/wits/judge/transcript.ts:
renderTranscript), collapses the 3-class decision (auto_approve / confirm /
block) to our (allow / maybe / block) labels, and emits a stratified-split
JSONL ready for the transcript-aware GNN notebook.

Output per row:
  {
    "case_name":         "dd_disk_wipe",
    "transcript":        "--- Start of transcript ---\nUser: ...\nTool: ...",
    "proposed_command":  "dd if=/dev/zero of=/dev/sda bs=1M status=progress",
    "shell":             "bash" | "powershell",
    "intention":         "Securely erase disk by writing zeros",
    "decision":          "allow" | "maybe" | "block",
    "raw_decision":      "auto_approve" | "confirm" | "block",
    "report_bucket":     "Destructive operations",
    "tags":              [...],
    "split":             "train" | "test",
    "source":            "d3"
  }

Decision collapse:
  auto_approve -> allow
  confirm      -> maybe   (fall through to LLM judge)
  block        -> block

Runs deterministically (SEED=42, stratified). Writes
data/d3_transcript_cases.jsonl.

Usage:
    python data/build_d3_transcript_dataset.py
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

REPO = Path(__file__).resolve().parents[1]
D3_DIR = Path(r"C:/dev/what-in-the-shell-fresh/eval_cases")
OUT_PATH = REPO / "data" / "d3_transcript_cases.jsonl"

SEED = 42
TEST_FRAC = 0.25  # ~88 of 350 cases held out

# Even though D3's upstream eval cases use placeholder-looking tokens
# (TXXXXXX/BXXXXXX) for examples like `send_slack_message`, GitHub
# Push Protection matches the URL SHAPE and refuses the push. We
# neuter any Slack-webhook-shaped URLs to "REDACTED-SECRET-PATTERN"
# before the row is written. The model still sees the structural
# pattern (a slack-bound webhook), just without the credential shape.
import re as _re
_SLACK_HOOK_RX = _re.compile(r"https?://hooks\.slack\.com/services/[A-Za-z0-9/_-]+")
def _redact_secrets(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _SLACK_HOOK_RX.sub("https://hooks.slack.com/services/REDACTED-SECRET-PATTERN", text)

# Transcript-rendering caps. Mirror the runtime defaults in
# src/hooks/auto-approve/wits/judge/transcript.ts.
TRANSCRIPT_MAX_MESSAGES = 10
TRANSCRIPT_MAX_USER_CHARS = 500
TRANSCRIPT_MAX_ASSISTANT_CHARS = 500
TRANSCRIPT_MAX_TOOL_ARG_VALUE_CHARS = 500

# Tool-result status mapping. D3 uses telemetry-style result_type strings;
# the runtime uses {success, failure, timeout, rejected, denied, unknown}.
def _normalize_result_type(r: str) -> str:
    if not r:
        return "unknown"
    r = r.strip().lower()
    if r in ("success", "succeeded", "ok"):
        return "success"
    if r in ("failure", "error", "errored"):
        return "failure"
    if r in ("timeout", "timed_out", "timed-out"):
        return "timeout"
    if r in ("rejected", "user_rejected", "user-rejected", "rejected_by_user"):
        return "rejected"
    if r in ("denied", "denied_by_policy", "policy_denied"):
        return "denied"
    return "unknown"


TOOL_RESULT_LABEL = {
    "success":  "success",
    "failure":  "failure",
    "timeout":  "timed out",
    "rejected": "rejected by user",
    "denied":   "denied by policy",
}


def _truncate_and_quote(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return json.dumps(s, ensure_ascii=False)
    extra = len(s) - max_chars
    return json.dumps(s[:max_chars], ensure_ascii=False) + f" --- followed by {extra} characters, truncated for brevity"


def _render_arg_value(v, max_chars: int) -> str:
    if isinstance(v, str):
        return _truncate_and_quote(v, max_chars)
    j = json.dumps(v, ensure_ascii=False) if v is not None else "null"
    if len(j) <= max_chars:
        return j
    return j[:max_chars] + f" --- followed by {len(j)-max_chars} characters, truncated for brevity"


def _render_message(msg: dict) -> str | None:
    kind = msg.get("kind")
    props = msg.get("properties", {}) or {}
    if kind == "user_message":
        text = (props.get("content") or "").strip()
        if not text:
            return None
        return f"User: {_truncate_and_quote(text, TRANSCRIPT_MAX_USER_CHARS)}"
    if kind == "assistant_message":
        text = (props.get("content") or "").strip()
        # Many D3 assistant messages have empty content + a tool_requests
        # blob (the model called a tool without prose). Render those as a
        # placeholder so the count is right.
        if not text:
            return None
        return f"Assistant: {_truncate_and_quote(text, TRANSCRIPT_MAX_ASSISTANT_CHARS)}"
    if kind == "tool_call_executed":
        tool_name = props.get("tool_name") or props.get("command") or "tool"
        args_raw = props.get("arguments")
        try:
            parsed = json.loads(args_raw) if isinstance(args_raw, str) and args_raw else (args_raw or {})
        except Exception:
            parsed = None
        lines = [f"Tool: {tool_name} tool called with arguments:"]
        if isinstance(parsed, dict) and parsed:
            for k, v in parsed.items():
                lines.append(f"  {k}: {_render_arg_value(v, TRANSCRIPT_MAX_TOOL_ARG_VALUE_CHARS)}")
        elif args_raw:
            lines.append(f"  {_truncate_and_quote(str(args_raw), TRANSCRIPT_MAX_TOOL_ARG_VALUE_CHARS)}")
        else:
            lines.append("  (no arguments)")
        status = _normalize_result_type(props.get("result_type", ""))
        if status != "unknown":
            lines.append(f"Tool result: {TOOL_RESULT_LABEL[status]}")
        return "\n".join(lines)
    if kind == "permission_prompt":
        # Render permission prompts that come BEFORE the target as
        # historical context. The TARGET permission_prompt itself becomes
        # `proposed_command` and is NOT rendered into the transcript.
        cmd = props.get("command_text") or props.get("file_name") or ""
        intent = props.get("intention") or ""
        if not cmd:
            return None
        return (f"Permission prompt earlier: kind={props.get('permission_type','?')}, "
                f"intention={_truncate_and_quote(intent, 200)}, "
                f"action={_truncate_and_quote(cmd[:300], 300)}")
    # tool_call_requested, etc. — skip
    return None


def _header_line(before: int) -> str:
    if before == 0:
        return "--- Start of transcript ---"
    return f"--- {before} messages happened in the thread before this ---"


def render_transcript(messages: list[dict]) -> str:
    """Mirror of renderTranscript in src/.../wits/judge/transcript.ts.

    Anchors on the most-recent user message and renders the last
    TRANSCRIPT_MAX_MESSAGES messages, with an elision line for any
    older assistant/tool messages between the anchor and the tail.
    """
    if not messages:
        return ""

    # Anchor on the most recent user message.
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("kind") == "user_message":
            last_user_idx = i
            break

    out: list[str] = []

    if last_user_idx == -1:
        # No user message — fall back to last N messages.
        slice_ = messages[-TRANSCRIPT_MAX_MESSAGES:]
        before = len(messages) - len(slice_)
        out.append(_header_line(before))
        for m in slice_:
            r = _render_message(m)
            if r:
                out.append(r)
        out.append("--- End of transcript ---")
        return "\n".join(out)

    if len(messages) - last_user_idx <= TRANSCRIPT_MAX_MESSAGES:
        # Last user message fits in the natural recency window.
        start = max(0, len(messages) - TRANSCRIPT_MAX_MESSAGES)
        out.append(_header_line(start))
        for m in messages[start:]:
            r = _render_message(m)
            if r:
                out.append(r)
        out.append("--- End of transcript ---")
        return "\n".join(out)

    # Last user message is older than the natural window; force-include.
    tail_keep = TRANSCRIPT_MAX_MESSAGES - 1
    tail = messages[-tail_keep:]
    elided = messages[last_user_idx + 1 : len(messages) - tail_keep]
    out.append(_header_line(last_user_idx))
    anchor = _render_message(messages[last_user_idx])
    if anchor:
        out.append(anchor)
    asst = sum(1 for m in elided if m.get("kind") == "assistant_message")
    tool = sum(1 for m in elided if m.get("kind") == "tool_call_executed")
    out.append(f"--- {asst} assistant and {tool} tool messages elided ---")
    for m in tail:
        r = _render_message(m)
        if r:
            out.append(r)
    out.append("--- End of transcript ---")
    return "\n".join(out)


def _extract_target_permission_prompt(case: dict) -> dict | None:
    """Locate the target permission_prompt. D3 cases store its index in
    target_event_index; some don't, and we fall back to the last
    permission_prompt with permission_type=commands."""
    msgs = case.get("messages", []) or []
    idx = case.get("target_event_index")
    if isinstance(idx, int) and 0 <= idx < len(msgs):
        m = msgs[idx]
        if m.get("kind") == "permission_prompt":
            return m
    # Fallback: last permission_prompt of any type.
    for m in reversed(msgs):
        if m.get("kind") == "permission_prompt":
            return m
    return None


def _shell_for(case: dict, command: str) -> str:
    # explicit `shell` field or `powershell`/`pwsh` tag.
    sh = (case.get("shell") or "").lower()
    if sh in ("powershell", "pwsh"):
        return "powershell"
    if sh == "bash":
        return "bash"
    tags = [str(t).lower() for t in (case.get("tags") or [])]
    if "powershell" in tags or "pwsh" in tags:
        return "powershell"
    # Heuristic from the command body.
    if re.search(r"^\s*(Get|Set|Invoke|New|Remove|Stop|Start|Restart|Test|Out|Where|Select|Sort|Format)-[A-Z]", command):
        return "powershell"
    if "\\" in command and "/" not in command.split()[0]:
        return "powershell"
    return "bash"


def _collapse_decision(d: str) -> str:
    d = (d or "").lower()
    if d == "auto_approve":
        return "allow"
    if d == "confirm":
        return "maybe"
    if d == "block":
        return "block"
    return ""  # unknown → caller drops the row


def _iter_cases(d3_dir: Path) -> Iterator[dict]:
    for f in sorted(d3_dir.glob("*.json")):
        try:
            yield json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")


def _expand_variants(case: dict) -> Iterable[dict]:
    """Variant cases declare expected[<variant>]; we yield one row per variant.
    Non-variant cases pass through unchanged."""
    variants = case.get("variants")
    if not variants:
        yield case
        return
    expected_by_variant = case.get("expected") or {}
    if not isinstance(expected_by_variant, dict):
        yield case
        return
    base_name = case.get("name", "")
    for v in variants:
        ev = expected_by_variant.get(v)
        if not isinstance(ev, dict):
            continue
        new = dict(case)
        new["name"] = f"{base_name}::{v}"
        new["variant"] = v
        new["expected"] = ev
        # Filter messages whose only_in_variants/except_in_variants exclude this variant.
        filtered = []
        for m in case.get("messages", []):
            props = m.get("properties", {}) or {}
            only = props.get("only_in_variants")
            ex   = props.get("except_in_variants")
            if isinstance(only, list) and v not in only:
                continue
            if isinstance(ex, list) and v in ex:
                continue
            filtered.append(m)
        new["messages"] = filtered
        yield new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not D3_DIR.exists():
        raise SystemExit(f"D3 dir not found: {D3_DIR}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped_reasons = Counter()
    n_files = 0
    for case in _iter_cases(D3_DIR):
        n_files += 1
        for c in _expand_variants(case):
            target = _extract_target_permission_prompt(c)
            if target is None:
                skipped_reasons["no_permission_prompt"] += 1
                continue
            target_props = target.get("properties", {}) or {}
            ptype = target_props.get("permission_type")
            if ptype != "commands":
                # We're training a shell-command classifier. Skip `write`
                # permission prompts (those go through Tier 2 in the
                # production gate, not the WITS judge).
                skipped_reasons[f"target_ptype:{ptype}"] += 1
                continue
            command = (target_props.get("command_text") or "").strip()
            if not command:
                skipped_reasons["empty_command"] += 1
                continue
            intention = (target_props.get("intention") or "").strip()

            expected = c.get("expected") or {}
            decision_3 = _collapse_decision(expected.get("decision", ""))
            if not decision_3:
                skipped_reasons["unknown_decision"] += 1
                continue

            # Render only messages BEFORE the target prompt.
            messages = c.get("messages", []) or []
            idx = c.get("target_event_index")
            if isinstance(idx, int) and 0 <= idx < len(messages):
                pre_messages = messages[:idx]
            else:
                # Drop the target permission_prompt itself from the tail.
                pre_messages = [m for m in messages if m is not target]

            transcript = render_transcript(pre_messages)
            shell = _shell_for(c, command)

            rows.append({
                "case_name":         c.get("name", ""),
                "transcript":        _redact_secrets(transcript),
                "proposed_command":  _redact_secrets(command),
                "shell":             shell,
                "intention":         _redact_secrets(intention),
                "decision":          decision_3,
                "raw_decision":      (expected.get("decision") or "").lower(),
                "report_bucket":     c.get("report_bucket", ""),
                "tags":              c.get("tags", []),
                "source":            "d3",
            })

    # Stratified split by decision.
    by_class: dict[str, list[dict]] = {"allow": [], "maybe": [], "block": []}
    for r in rows:
        by_class[r["decision"]].append(r)
    rng = random.Random(SEED)
    for k in by_class:
        rng.shuffle(by_class[k])
    train, test = [], []
    for k, lst in by_class.items():
        n_te = max(1, round(len(lst) * TEST_FRAC)) if len(lst) >= 2 else 0
        for r in lst[:n_te]:
            r2 = dict(r); r2["split"] = "test";  test.append(r2)
        for r in lst[n_te:]:
            r2 = dict(r); r2["split"] = "train"; train.append(r2)
    rng2 = random.Random(SEED + 1); rng2.shuffle(train); rng2.shuffle(test)
    all_rows = train + test

    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ----- Summary -----
    print(f"Scanned {n_files} D3 case files.")
    print(f"  emitted {len(rows)} rows (after variant expansion + filtering).")
    if skipped_reasons:
        print("  skip reasons:")
        for k, v in skipped_reasons.most_common():
            print(f"    {v:4d}  {k}")

    print(f"\nWrote {len(all_rows)} rows -> {OUT_PATH}")
    by_d = Counter(r["decision"] for r in all_rows)
    by_split = Counter(r["split"] for r in all_rows)
    print(f"  by decision : {dict(by_d)}")
    print(f"  by split    : {dict(by_split)}")
    print(f"  train dist  : {dict(Counter(r['decision'] for r in all_rows if r['split']=='train'))}")
    print(f"  test  dist  : {dict(Counter(r['decision'] for r in all_rows if r['split']=='test'))}")
    print(f"  shell mix   : {dict(Counter(r['shell'] for r in all_rows))}")

    # Transcript length sanity.
    tlens = [len(r["transcript"]) for r in all_rows]
    cmdlens = [len(r["proposed_command"]) for r in all_rows]
    import statistics
    print(f"\n  transcript len chars : mean={statistics.mean(tlens):.0f} "
          f"median={int(statistics.median(tlens))} max={max(tlens)} "
          f"p95={sorted(tlens)[int(0.95*len(tlens))]}")
    print(f"  command    len chars : mean={statistics.mean(cmdlens):.0f} "
          f"median={int(statistics.median(cmdlens))} max={max(cmdlens)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
