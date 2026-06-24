"""Extract labeled (command, shell, verdict) tuples from WITS test files.

Sources:
  - C:/dev/copilot-agent-runtime-final/test/hooks/auto-approve/wits/**/*.test.ts
  - C:/dev/what-in-the-shell-fresh/tests/**/*.test.ts

Both repos drive their unit tests off arrays of TS case objects of the form

    { command: "ls -la", verdict: "safe", mustFire: [...] }

This script harvests those literals (only the ones carrying an explicit
`verdict:` field) into a single JSONL of labeled examples for downstream
GNN training. We deliberately do NOT infer labels from filenames (e.g.
`known-safe.test.ts`) — only explicit `verdict:` values are kept.

Heuristics for parsing TS source:
  - String literals: "...", '...', `...` (template). Escapes via backslash.
  - Line comments (`// ...`) and block comments (`/* ... */`) are ignored.
  - We walk the source tracking brace depth; for every top-level `{...}`
    block we test whether it contains a `command:` *and* `verdict:` key.
  - `shell: "powershell"` inside the same object overrides the default.
  - Default shell defaults to `powershell` for files named
    `windows-powershell.test.ts` / `windows.test.ts` / inside `/powershell/`,
    else `bash`.

Output schema (JSONL, one record per line):
    {"command": str, "shell": "bash"|"powershell",
     "verdict": "safe"|"maybe_safe"|"unsafe"|"extremely_unsafe",
     "source": str  # e.g. "car:known-safe", "wits:domains/git"
    }

Run:  python data/extract_wits_cases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

VALID_VERDICTS = {"safe", "maybe_safe", "unsafe", "extremely_unsafe"}
VALID_SHELLS = {"bash", "powershell"}

# ---------------------------------------------------------------------------
# Tokenizer + brace walker
# ---------------------------------------------------------------------------

def _strip_strings_and_comments(src: str) -> str:
    """Return a copy of `src` with string literals and comments replaced by
    spaces of the same length. This lets us run trivial substring/regex
    searches on the result without false positives from the contents of
    strings/comments. Newlines are preserved so line numbers still line up.
    """
    out = []
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        # Line comment
        if ch == "/" and nxt == "/":
            j = src.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
            continue
        # Block comment
        if ch == "/" and nxt == "*":
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            block = src[i:j]
            out.append("".join(" " if c != "\n" else "\n" for c in block))
            i = j
            continue
        # String literal: ", ', `
        if ch in ('"', "'", "`"):
            quote = ch
            j = i + 1
            while j < n:
                c = src[j]
                if c == "\\":
                    j += 2
                    continue
                if c == quote:
                    j += 1
                    break
                # template strings can span newlines; preserve them
                j += 1
            block = src[i:j]
            out.append("".join(" " if c != "\n" else "\n" for c in block))
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _iter_case_object_spans(src: str) -> Iterator[tuple[int, int]]:
    """Yield (start, end) char offsets of every `{ ... }` block in `src` that
    sits at the *innermost* nesting around `command:` keys. Strategy:
    find each `command` identifier at depth>=1, then walk outward to the
    enclosing `{` / `}` braces.

    Implementation: we precompute a depth-stack of `{` positions; for any
    `command:` occurrence at depth d we use the top of the stack at that
    point as the enclosing object's `{`. The matching `}` is the next `}`
    that pops back to d-1.
    """
    masked = _strip_strings_and_comments(src)
    n = len(masked)

    # Precompute matching-brace map.
    stack: list[int] = []
    pair: dict[int, int] = {}
    for i, c in enumerate(masked):
        if c == "{":
            stack.append(i)
        elif c == "}" and stack:
            open_i = stack.pop()
            pair[open_i] = i

    # Find `command:` identifiers (word boundary).
    seen: set[int] = set()
    i = 0
    while True:
        j = masked.find("command", i)
        if j == -1:
            break
        i = j + 1
        # Word boundary checks (prev char non-ident, next chars are ":" possibly with whitespace).
        if j > 0 and (masked[j - 1].isalnum() or masked[j - 1] == "_"):
            continue
        k = j + len("command")
        # skip spaces
        while k < n and masked[k] in " \t":
            k += 1
        if k >= n or masked[k] != ":":
            continue
        # Find enclosing `{` by scanning backwards through the brace map.
        # Easier: walk back tracking depth.
        depth = 0
        m = j - 1
        while m >= 0:
            ch = masked[m]
            if ch == "}":
                depth += 1
            elif ch == "{":
                if depth == 0:
                    break
                depth -= 1
            m -= 1
        if m < 0 or m not in pair:
            continue
        span = (m, pair[m])
        if span[0] in seen:
            continue
        seen.add(span[0])
        yield span


def _extract_string_value(src: str, key: str, lo: int, hi: int) -> str | None:
    """Inside `src[lo:hi]`, find `<key>:` at any *direct* depth and return the
    immediately-following string literal value. Only matches plain string
    literals (", ', `) — not concatenations, identifiers, or expressions.
    Uses depth tracking to skip into-and-out-of nested object/array literals.
    """
    masked = _strip_strings_and_comments(src[lo:hi])
    # Walk masked looking for `<key>` at depth 0 of the body (i.e. depth 1
    # from the original file's perspective, but the {..} wrapper is at 0/end).
    depth = 0
    i = 0
    n = len(masked)
    klen = len(key)
    while i < n:
        ch = masked[i]
        if ch == "{":
            depth += 1
            i += 1
            continue
        if ch == "}":
            depth -= 1
            i += 1
            continue
        if ch == "[":
            depth += 1
            i += 1
            continue
        if ch == "]":
            depth -= 1
            i += 1
            continue
        if depth != 1:
            i += 1
            continue
        # Try matching `key` at this position with word boundary.
        if masked.startswith(key, i):
            before_ok = i == 0 or not (masked[i - 1].isalnum() or masked[i - 1] == "_")
            after_pos = i + klen
            if before_ok:
                k = after_pos
                while k < n and masked[k] in " \t":
                    k += 1
                if k < n and masked[k] == ":":
                    # Walk past whitespace using the ORIGINAL source — `masked`
                    # has strings/comments replaced with spaces, so its
                    # whitespace is not trustworthy for locating the value
                    # boundary.
                    abs_pos = lo + k + 1
                    while abs_pos < hi and src[abs_pos] in " \t\n\r":
                        abs_pos += 1
                    if abs_pos < hi and src[abs_pos] in ('"', "'", "`"):
                        return _read_string_at(src, abs_pos)
                    # Otherwise: not a plain string literal value, skip.
                    i = (abs_pos - lo) if abs_pos > lo + k else k + 1
                    continue
        i += 1
    return None


def _read_string_at(src: str, pos: int) -> str:
    """Read a TS/JS string literal starting at `src[pos]` (which must be a
    quote). Returns the decoded value. Supports `, ', `."""
    quote = src[pos]
    assert quote in ('"', "'", "`"), f"not a quote at pos={pos}: {src[pos]!r}"
    out = []
    i = pos + 1
    n = len(src)
    while i < n:
        c = src[i]
        if c == "\\" and i + 1 < n:
            nxt = src[i + 1]
            mapping = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"', "`": "`", "0": "\0"}
            out.append(mapping.get(nxt, nxt))
            i += 2
            continue
        if c == quote:
            return "".join(out)
        out.append(c)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Per-file extraction
# ---------------------------------------------------------------------------

def _default_shell_for(path: Path) -> str:
    p = str(path).replace("\\", "/").lower()
    if "/powershell/" in p:
        return "powershell"
    name = path.name.lower()
    if "windows" in name and "powershell" in name:
        return "powershell"
    if name == "windows.test.ts":
        return "powershell"
    if "powershell" in name:
        return "powershell"
    return "bash"


def _source_label(root: Path, repo_tag: str, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    stem = rel.with_suffix("").as_posix()
    if stem.endswith(".test"):
        stem = stem[: -len(".test")]
    return f"{repo_tag}:{stem}"


def extract_from_file(path: Path, repo_root: Path, repo_tag: str) -> list[dict]:
    src = path.read_text(encoding="utf-8")
    default_shell = _default_shell_for(path)
    src_label = _source_label(repo_root, repo_tag, path)
    out: list[dict] = []
    for lo, hi in _iter_case_object_spans(src):
        command = _extract_string_value(src, "command", lo, hi)
        verdict = _extract_string_value(src, "verdict", lo, hi)
        if command is None or verdict is None:
            continue
        if verdict not in VALID_VERDICTS:
            continue
        if not command.strip():
            continue
        shell = _extract_string_value(src, "shell", lo, hi) or default_shell
        if shell not in VALID_SHELLS:
            shell = default_shell
        out.append({
            "command": command,
            "shell": shell,
            "verdict": verdict,
            "source": src_label,
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _gather(repo_root: Path, repo_tag: str, glob: str) -> list[dict]:
    records: list[dict] = []
    if not repo_root.exists():
        print(f"  [skip] {repo_root} does not exist")
        return records
    files = sorted(repo_root.glob(glob))
    print(f"  scanning {len(files)} test files under {repo_root} (glob={glob})")
    for f in files:
        recs = extract_from_file(f, repo_root, repo_tag)
        records.extend(recs)
    print(f"  extracted {len(records)} labeled cases from {repo_tag}")
    return records


def main() -> int:
    repo_dir = Path(__file__).resolve().parents[1]
    out_path = repo_dir / "data" / "wits_eval_cases.jsonl"

    sources = [
        (
            Path(r"C:/dev/copilot-agent-runtime-final"),
            "car",
            "test/hooks/auto-approve/wits/**/*.test.ts",
        ),
        (
            Path(r"C:/dev/what-in-the-shell-fresh"),
            "wits",
            "tests/**/*.test.ts",
        ),
    ]

    all_records: list[dict] = []
    for root, tag, glob in sources:
        all_records.extend(_gather(root, tag, glob))

    # Merge the reviewer-curated companion file. It carries Phase-1
    # hand-picked cases, Phase-2 programmatic augmentation, and Phase-3
    # hard negatives. Records from this file WIN ties against the raw
    # WITS-test extraction (Phase-0 decontamination of polluted "safe"
    # labels). Source bucket starts with "reviewer:".
    reviewer_path = repo_dir / "data" / "wits_eval_cases_reviewer.jsonl"
    reviewer_records: list[dict] = []
    if reviewer_path.exists():
        for line in reviewer_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("verdict") not in VALID_VERDICTS:
                continue
            if r.get("shell") not in VALID_SHELLS:
                continue
            reviewer_records.append(r)
        print(f"  loaded {len(reviewer_records)} reviewer-curated cases from {reviewer_path.name}")
    else:
        print(f"  [skip] no reviewer-curated cases at {reviewer_path}")

    # Merge the Phase-4 gap-fill companion file. Source bucket starts
    # with "gap:". Same priority as reviewer (overrides raw WITS-test
    # extraction on ties). Targets the audit-flagged holes: PowerShell
    # attacks, extreme disk destruction, network attacks, git destructive
    # ops, persistence/evasion, maybe_safe diversity, long safe commands.
    gap_path = repo_dir / "data" / "wits_eval_cases_gap_fill.jsonl"
    gap_records: list[dict] = []
    if gap_path.exists():
        for line in gap_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("verdict") not in VALID_VERDICTS:
                continue
            if r.get("shell") not in VALID_SHELLS:
                continue
            gap_records.append(r)
        print(f"  loaded {len(gap_records)} gap-fill cases from {gap_path.name}")
    else:
        print(f"  [skip] no gap-fill cases at {gap_path}")

    # Merge the Phase-5 agent-gating companion file. Source bucket starts
    # with "agent-gating:". Highest priority — targets the agent-context
    # surface (cloud-metadata SSRF, /proc/self/environ, IDE confused-
    # deputy via .vscode/.devcontainer, secret-file reads, untrusted ./
    # scripts), the common agent-workflow `safe` commands the corpus was
    # missing (pytest, eslint, mypy, cargo, docker images, kubectl logs,
    # …), and contrastive same-argv0 safes to keep the GNN from
    # collapsing on first-token features (more plain `curl GET`, `echo`,
    # `find`, `awk` safes against the attack majorities).
    agent_path = repo_dir / "data" / "wits_eval_cases_agent_gating.jsonl"
    agent_records: list[dict] = []
    if agent_path.exists():
        for line in agent_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("verdict") not in VALID_VERDICTS:
                continue
            if r.get("shell") not in VALID_SHELLS:
                continue
            agent_records.append(r)
        print(f"  loaded {len(agent_records)} agent-gating cases from {agent_path.name}")
    else:
        print(f"  [skip] no agent-gating cases at {agent_path}")

    # Merge the Phase-6 diversity-polish companion file. Source bucket
    # starts with "diversity:". Highest priority. Targets ML-failure
    # modes surfaced by data/_audit_ml.py:
    #  - 6a: payload diversification (rewrites attacks with realistic
    #    domains/paths so the model can't memorize "the literal string
    #    `evil`" or `evil.example`),
    #  - 6b: `maybe_safe` Schelling-point expansion (terraform plan,
    #    kubectl apply, helm install, git push to feature/..., chmod +x
    #    ./local.sh, …) so the "fall through to judge" class actually
    #    represents the gray area,
    #  - 6c: BENIGN env-prefix cases (NODE_OPTIONS=…, RUST_BACKTRACE=1)
    #    so the model has to learn dangerous-env-var-name discrimination
    #    rather than treating every `VAR=… cmd` form as unsafe,
    #  - 6d: contrastive examples for tokens (`ssh`, `DELETE`, `fetch`,
    #    `/etc/`, `system`) that were 100 % predictive of unsafe.
    diversity_path = repo_dir / "data" / "wits_eval_cases_diversity.jsonl"
    diversity_records: list[dict] = []
    if diversity_path.exists():
        for line in diversity_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("verdict") not in VALID_VERDICTS:
                continue
            if r.get("shell") not in VALID_SHELLS:
                continue
            diversity_records.append(r)
        print(f"  loaded {len(diversity_records)} diversity-polish cases from {diversity_path.name}")
    else:
        print(f"  [skip] no diversity-polish cases at {diversity_path}")

    # Dedupe: WITS-test cases first (baseline), then reviewer file, then
    # gap-fill file, then agent-gating file, then diversity-polish file.
    # Each subsequent source OVERWRITES earlier records on (command,
    # shell) key collision. Curated files reflect the actual verdict
    # definitions; raw WITS-test labels reflect what the analyzer
    # CURRENTLY emits, which the review comment showed is sometimes
    # wrong. Curated wins.
    seen: dict[tuple[str, str], dict] = {}
    overwritten = 0
    for r in all_records:
        seen[(r["command"], r["shell"])] = r
    for r in reviewer_records:
        key = (r["command"], r["shell"])
        if key in seen and seen[key].get("verdict") != r["verdict"]:
            overwritten += 1
        seen[key] = r
    for r in gap_records:
        key = (r["command"], r["shell"])
        if key in seen and seen[key].get("verdict") != r["verdict"]:
            overwritten += 1
        seen[key] = r
    for r in agent_records:
        key = (r["command"], r["shell"])
        if key in seen and seen[key].get("verdict") != r["verdict"]:
            overwritten += 1
        seen[key] = r
    for r in diversity_records:
        key = (r["command"], r["shell"])
        if key in seen and seen[key].get("verdict") != r["verdict"]:
            overwritten += 1
        seen[key] = r
    deduped = list(seen.values())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    print(f"\nWrote {len(deduped)} records -> {out_path}")
    from collections import Counter
    by_v = Counter(r["verdict"] for r in deduped)
    by_s = Counter(r["shell"] for r in deduped)
    by_src = Counter(r["source"].split(":", 1)[0] for r in deduped)
    print("  by verdict:", dict(by_v))
    print("  by shell  :", dict(by_s))
    print("  by repo   :", dict(by_src))
    total_input = (
        len(all_records)
        + len(reviewer_records)
        + len(gap_records)
        + len(agent_records)
        + len(diversity_records)
    )
    if total_input != len(deduped):
        print(f"  ({total_input - len(deduped)} duplicates collapsed)")
    if overwritten:
        print(f"  ({overwritten} verdicts overwritten by curated relabels)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
