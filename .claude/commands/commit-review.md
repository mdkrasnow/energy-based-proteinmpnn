---
description: Review uncommitted changes through multi-agent debate, produce actionable fix plan, and execute fixes safely
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
model: claude-sonnet-4-5
---

# Code Review with Multi-Agent Debate

You are the orchestrator for a debate-driven code review system.

## Overview

FOUR phases:
1. **Setup & Static Analysis** - Get diff, run lint/tsc, establish baseline
2. **Multi-Agent Debate** - 4 reviewers × 3 rounds (parallel)
3. **Fix Plan Synthesis** - Produce executable fix plan
4. **Execution & Validation** - Apply fixes, re-check, iterate

---

## Phase 1: Setup & Static Analysis

### Step 1.1: Assess the Diff
```bash
mkdir -p .claude/work/review-debate
git diff --stat
git diff --name-only
git diff > .claude/work/review-debate/current.diff
```

Determine strategy: small (1-5 files), medium (6-15 files), large (16+ files).

Write `.claude/work/review-debate/diff-assessment.json`:
```json
{
  "timestamp": "ISO-8601",
  "strategy": "small|medium|large",
  "files_changed": 0,
  "lines_inserted": 0,
  "lines_deleted": 0,
  "high_risk_files": [],
  "changed_files": []
}
```

### Step 1.2: Run Static Checks
```bash
cd eval/frontend && npm run lint 2>&1 | tee .claude/work/review-debate/lint-baseline.txt
cd eval/frontend && npx tsc 2>&1 | tee .claude/work/review-debate/tsc-baseline.txt
```

Parse and write `.claude/work/review-debate/static-checks-baseline.json`:
```json
{
  "timestamp": "ISO-8601",
  "lint": {
    "errors": [],
    "warnings": [],
    "error_count": 0,
    "warning_count": 0,
    "clean": true/false
  },
  "typescript": {
    "errors": [],
    "error_count": 0,
    "clean": true/false
  },
  "baseline_has_errors": true/false
}
```

### Step 1.3: Create Workspace
```bash
mkdir -p .claude/work/review-debate/round-{1,2,3}
```

---

## Phase 2: Multi-Agent Debate

Coordinate 4 specialized reviewers:
- security-reviewer
- correctness-reviewer
- performance-reviewer
- maintainability-reviewer

### Round 1: Independent Analysis (Parallel)

Invoke `code-reviewer` 4 times in parallel:

**For each reviewer:**
- reviewer_id: "[security|correctness|performance|maintainability]-reviewer"
- focus: "[security|correctness|performance|maintainability]"
- round: 1
- model: sonnet
- critique_mode: "none"
- diff_path: ".claude/work/review-debate/current.diff"
- diff_assessment: ".claude/work/review-debate/diff-assessment.json"
- static_checks: ".claude/work/review-debate/static-checks-baseline.json"
- output_path: ".claude/work/review-debate/round-1/[reviewer_id].json"

Wait for all 4, create `.claude/work/review-debate/round-1/summary.md`.

### Round 2: Cross-Examination (Parallel)

Invoke `code-reviewer` 4 times in parallel:

**For each reviewer:**
- reviewer_id: "[reviewer]"
- focus: "[focus]"
- round: 2
- model: sonnet
- critique_mode: "cross-examine"
- diff_path: ".claude/work/review-debate/current.diff"
- own_previous: ".claude/work/review-debate/round-1/[reviewer_id].json"
- others_previous: [other 3 reviewers' round-1 outputs]
- output_path: ".claude/work/review-debate/round-2/[reviewer_id].json"

Wait for all 4, create `.claude/work/review-debate/round-2/summary.md`.

### Round 3: Prioritization (Parallel)

Invoke `code-reviewer` 4 times in parallel:

**For each reviewer:**
- reviewer_id: "[reviewer]"
- focus: "[focus]"
- round: 3
- model: sonnet
- critique_mode: "prioritize"
- diff_path: ".claude/work/review-debate/current.diff"
- own_previous: ".claude/work/review-debate/round-2/[reviewer_id].json"
- others_previous: [other 3 reviewers' round-2 outputs]
- output_path: ".claude/work/review-debate/round-3/[reviewer_id].json"

Wait for all 4, create `.claude/work/review-debate/round-3/summary.md`.

---

## Phase 3: Fix Plan Synthesis

Invoke `fix-plan-synthesizer` once:

- diff_path: ".claude/work/review-debate/current.diff"
- diff_assessment: ".claude/work/review-debate/diff-assessment.json"
- static_checks_baseline: ".claude/work/review-debate/static-checks-baseline.json"
- round_1_reviews: [all 4 reviewer outputs from round-1/]
- round_2_reviews: [all 4 reviewer outputs from round-2/]
- round_3_reviews: [all 4 reviewer outputs from round-3/]
- output_path: ".claude/work/review-debate/fix-plan.json"

---

## Phase 4: Execution & Validation

### Step 4.1: Read Fix Plan

Read `.claude/work/review-debate/fix-plan.json`.

If no fixes needed → skip to Step 4.6.

### Step 4.2: Apply Fixes in Priority Order

For each fix in `fix_plan.fixes`:
1. Read the file
2. Locate exact `old_code`
3. Apply change to produce `new_code`
4. Use Edit tool
5. Track what changed

Keep fix execution log in memory.

### Step 4.3: Re-run Static Checks
```bash
cd eval/frontend && npm run lint 2>&1 | tee .claude/work/review-debate/lint-after.txt
cd eval/frontend && npx tsc 2>&1 | tee .claude/work/review-debate/tsc-after.txt
```

Parse and write `.claude/work/review-debate/static-checks-after.json`.

### Step 4.4: Compare Before/After

Classify outcome:
- **IMPROVED**: errors decreased, no new errors
- **CLEAN**: error_count now 0
- **DEGRADED**: errors increased or new errors
- **UNCHANGED**: same error_count

### Step 4.5: Handle Degradation

If DEGRADED:
1. Identify problematic fix
2. Revert it
3. Re-run checks
4. Report partial success

### Step 4.6: Final Report
```
=== CODE REVIEW DEBATE & FIX SUMMARY ===

DIFF SCOPE: [strategy] - X files, Y lines

DEBATE PROCESS:
  ✓ Round 1: Independent analysis
  ✓ Round 2: Cross-examination
  ✓ Round 3: Prioritization

FINDINGS: Critical X, High Y, Medium Z, Low W

FIXES APPLIED: X changes across Y files
  ✓ file:line - Fixed [issue]

STATIC CHECKS:
  BEFORE: lint X, tsc Y
  AFTER:  lint A (-B), tsc C (-D)

OUTCOME: [CLEAN|IMPROVED|DEGRADED|UNCHANGED]

NEXT STEPS: [context-appropriate guidance]
```

---

## Safety & Error Handling

- Never modify files outside the diff
- Revert if checks worsen
- One fix at a time
- Preserve all artifacts
- Stop on phase failure with clear error message

## Exit Criteria

Complete when:
- All 3 debate rounds done
- Fix plan synthesized
- Fixes applied/skipped
- Checks re-run
- Final report presented