/**
 * WITS-static-analyzer scoring shim.
 *
 * Reads JSONL records from stdin, one per line:
 *   {"command": str, "shell": "bash"|"powershell", ...passthrough}
 *
 * For each record, invokes `whatInTheShell.isThis(command, { shell,
 * workspace: Workspace.test(), network: false })` and writes a single JSONL
 * line to stdout containing:
 *   {
 *     "command":            <original>,
 *     "shell":              <original>,
 *     "verdict":            <wits-predicted-verdict>,
 *     "rule_ids":           [<rule.ruleId>, ...],
 *     "elapsed_ms":         <wall-clock>,
 *     "error":              <error message OR omitted>
 *   }
 *
 * `network: false` is deliberate — keeps the run deterministic (no OSV
 * fetches, no curl|sh body re-analysis). Matches what
 * `eval/verdict_snapshot.ts` does.
 *
 * Run:
 *   node data/_wits_score_shim.cjs < input.jsonl > predictions.jsonl
 */
"use strict";

const readline = require("node:readline");

const WITS_DIST = process.env.WITS_DIST || "c:/dev/what-in-the-shell-fresh/dist/index.cjs";
const { whatInTheShell, Workspace, prewarmPowerShellParser } = require(WITS_DIST);

async function main() {
    // Pre-warm the PowerShell parser so PS cases don't pay the cold start.
    try {
        await prewarmPowerShellParser();
    } catch (e) {
        // Non-fatal — bash cases will still work.
        process.stderr.write(`prewarm failed: ${e.message}\n`);
    }

    const rl = readline.createInterface({ input: process.stdin });
    const ws = Workspace.test();

    let count = 0;
    let errors = 0;
    for await (const line of rl) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        let rec;
        try {
            rec = JSON.parse(trimmed);
        } catch (e) {
            process.stderr.write(`bad json on line ${count + 1}: ${e.message}\n`);
            continue;
        }
        const command = rec.command;
        const shell = rec.shell === "powershell" ? "powershell" : "bash";
        const t0 = Date.now();
        let out = {
            command,
            shell,
            verdict: null,
            rule_ids: [],
            elapsed_ms: 0,
        };
        try {
            const result = await whatInTheShell.isThis(command, {
                workspace: ws,
                shell,
                network: false,
            });
            out.verdict = result.verdict;
            out.rule_ids = (result.ruleHits || []).map((h) => h.ruleId);
        } catch (e) {
            out.error = e && e.message ? e.message : String(e);
            errors++;
        }
        out.elapsed_ms = Date.now() - t0;
        process.stdout.write(JSON.stringify(out) + "\n");
        count++;
        if (count % 50 === 0) {
            process.stderr.write(`  ${count} scored (${errors} errors)\n`);
        }
    }
    process.stderr.write(`done. ${count} records scored, ${errors} errors.\n`);
}

main().catch((e) => {
    process.stderr.write(`fatal: ${e && e.stack ? e.stack : e}\n`);
    process.exit(1);
});
