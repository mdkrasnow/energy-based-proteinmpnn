---
description: Build with review, then run multi-agent debate review, and apply fixes via subagent
argument-hint: <user request>
allowed-tools: Read, Write, Edit, Grep, Glob, Bash(git *), TodoManager, Bash(cat*), Touch, Cat, Echo, Bash
model: claude-sonnet-4-5
---

# Build with Review + Multi-Agent Debate + Auto-Fix

You are the orchestrator for a comprehensive build-review-fix workflow.

## Workflow Overview

1. **Phase 1: Analyze & Plan** - Investigate codebase, create implementation todos
2. **Phase 2: Execute Implementation** - Build with alternating implementation/review
3. **Phase 3: Commit Self-Review** - Multi-agent debate on all changes
4. **Phase 4: Fix Plan Synthesis** - Generate actionable fix plan from debate
5. **Phase 5: Apply Fixes** - Use subagent to implement fixes safely
6. **Phase 6: Final Validation** - Verify complete solution

User's request: "$ARGUMENTS"

---

# Phase 1: Analyze & Plan

## 1.1 Deep Analysis & Investigation

**COMPLETE ALL INVESTIGATION NOW** - Do not defer investigation to later todos.

First, thoroughly analyze the request:
- What is the core goal and why?
- What files/components will be affected?
- What are the technical requirements and constraints?
- What edge cases or failure modes should we anticipate?
- What could go wrong (regressions, breaking changes, performance)?

**Then, INVESTIGATE THE CODEBASE:**
- Locate all relevant files and read them
- Understand the current implementation
- Use Grep/Glob to find related code
- Identify existing patterns and architecture
- Discover dependencies and integration points
- **Find the actual problems** - don't just speculate

**CRITICAL**: By the end of this phase, you must have:
- ✓ Located all relevant files
- ✓ Understood the current implementation
- ✓ Identified the specific issues to fix
- ✓ Determined the best approach

Only after completing investigation should you create the todo list.

## 1.2 Create Structured Todo List

Now create a todo list with **ONLY IMPLEMENTATION TASKS** (actual code changes):

**RULES FOR TODO TASKS:**
- ❌ NO analysis tasks ("Locate files", "Identify issues", "Examine component")
- ❌ NO investigation tasks ("Research", "Explore", "Understand")
- ✅ ONLY implementation tasks ("Fix X", "Add Y", "Refactor Z", "Update W")
- ✅ Each implementation task gets a review task after it (unless trivial)

For each logical implementation task:
1. **Implementation task**: "Fix/Add/Refactor [specific change]"
   - Be specific about what files to modify
   - Keep scope small and reversible
   - Must be an actual code change, not investigation
2. **Review task**: "Review: [what was just implemented]"
   - This forces a fresh evaluation after each implementation
   - Review ONLY the changes from the previous task
   - **Exception**: Skip review for trivial changes (judgment call)

**Critical**: The review tasks must use these unbiased techniques:

### Review Technique: Critical Distance with Perspective Shift

When you reach a review task, **PUT ON A DIFFERENT HAT**:

**You are now a senior code reviewer who:**
- Did NOT write this code and has no emotional attachment to it
- Is seeing these changes for the first time
- Has a reputation for catching subtle bugs
- Gets bonuses for finding issues, not for approving code
- Is slightly paranoid about regressions

**Your mindset shift:**
1. **Read ONLY the git diff** from the previous implementation task
2. Actively question every change: "Why this way? What could break?"
3. Apply this checklist systematically:

**Correctness Checklist:**
- Does it actually solve what the previous todo asked for?
- Are there any logical errors or typos?
- Does it handle the expected inputs correctly?
- **Think first**: What would "correct" look like here? Then check if the implementation matches.

**Edge Cases Checklist:**
- What happens with null/undefined/empty values?
- What about boundary conditions (0, 1, max values)?
- What if network requests fail or APIs return unexpected data?
- What about race conditions or timing issues?
- **Think first**: What could users do that might break this? What environmental conditions could cause problems?

**Regression Checklist:**
- Could this break existing functionality?
- Are there dependencies that might be affected?
- Did we maintain backward compatibility where needed?
- **Think first**: What was working before? Could this change that?

**Integration Checklist:**
- Does this fit with the surrounding code?
- Are variable names and patterns consistent?
- Does it follow the existing architecture?
- **Think first**: How does this piece connect to the rest of the system?

**STEP 4: DOCUMENT AND ACT**:

4. If you find issues: **fix them immediately** before moving to the next implementation task
5. Document what you checked and what you found (or didn't find) in the todo completion note

## 1.3 Add Final Validation Tasks

After all implementation/review pairs, add these final tasks:

1. **"Final Review: Integration"** - Check that all changes work together cohesively
2. **"Final Review: Requirements"** - Verify the complete solution addresses the original request
3. **"Final Review: Safety"** - Run tests, check for any remaining risks
4. **"Fix any final issues"** - Address anything found in final reviews

---

# Phase 2: Execute Implementation

**REMINDER**: Your todo list should contain ONLY implementation tasks (actual code changes), not investigation tasks. If you find yourself with "Locate files" or "Identify issues" todos, stop and complete that investigation now before proceeding.

Work through the todo list sequentially.

**For implementation tasks:**

**STEP 1: PLAN THE SOLUTION** (do this before writing any code):
- **Approach**: What's the most concise and elegant way to address this specific task?
- **Pitfall avoidance**: What changes could cause regressions or introduce bugs? How will you actively avoid them?
- **Best practices**: What coding patterns and principles apply here for maximum elegance and maintainability?
- **Keep it brief**: This should be a quick mental model, not a design document

**STEP 2: IMPLEMENT**:
- Execute the plan you just made
- Focus on clean, minimal changes
- Write clear code with good naming
- Keep diffs small and reviewable

**IMPORTANT - Adapting the plan as you learn:**
- As you research files, explore dependencies, and understand the codebase, you will discover new information
- **When discoveries warrant it** (not preemptively), update the todo list:
  - Found unexpected dependencies? Add todos to handle them
  - Discovered a better architectural approach? Refactor the task breakdown
  - Uncovered edge cases or technical debt? Add targeted todos
- This should be **reactive to what you learn**, not speculative planning
- Update todos naturally when you think "Oh, I need to also handle X" or "This approach won't work because Y"

**For review tasks (critical!):**

**PUT ON A DIFFERENT HAT** - You are now a skeptical code reviewer who didn't write this code:
- You're seeing these changes for the first time
- Your job is to find problems, not confirm success
- Be actively critical and look for what could go wrong

Now review:
- Get the diff: `git diff` or `git diff --cached`
- Apply the 4 checklists above systematically
- Challenge every assumption in the implementation
- Don't just confirm it looks okay - try to break it mentally
- If you find issues, fix them before proceeding

**For final review tasks:**
- Review ALL changes together: `git diff main` or equivalent
- Check integration points and data flow
- Verify the solution is complete and correct
- Run any relevant tests

---

# Phase 3: Multi-Agent Debate Review (Commit Self-Review)

After Phase 2 completes, run a comprehensive multi-agent debate on all changes.

## 3.1 Setup & Static Analysis

### Step 3.1.1: Assess the Diff
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

### Step 3.1.2: Run Static Checks
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
    "clean": true
  },
  "typescript": {
    "errors": [],
    "error_count": 0,
    "clean": true
  },
  "baseline_has_errors": false
}
```

### Step 3.1.3: Create Workspace
```bash
mkdir -p .claude/work/review-debate/round-{1,2,3}
```

## 3.2 Multi-Agent Debate (4 Reviewers × 3 Rounds)

Coordinate 4 specialized reviewers:
- security-reviewer
- correctness-reviewer
- performance-reviewer
- maintainability-reviewer

### Round 1: Independent Analysis (Parallel)

Invoke `code-reviewer` subagent 4 times in parallel:

**For each reviewer:**
```
reviewer_id: "[security|correctness|performance|maintainability]-reviewer"
focus: "[security|correctness|performance|maintainability]"
round: 1
model: sonnet
critique_mode: "none"
diff_path: ".claude/work/review-debate/current.diff"
diff_assessment: ".claude/work/review-debate/diff-assessment.json"
static_checks: ".claude/work/review-debate/static-checks-baseline.json"
output_path: ".claude/work/review-debate/round-1/[reviewer_id].json"
```

Wait for all 4 to complete, then create `.claude/work/review-debate/round-1/summary.md`.

### Round 2: Cross-Examination (Parallel)

Invoke `code-reviewer` subagent 4 times in parallel:

**For each reviewer:**
```
reviewer_id: "[reviewer]"
focus: "[focus]"
round: 2
model: sonnet
critique_mode: "cross-examine"
diff_path: ".claude/work/review-debate/current.diff"
own_previous: ".claude/work/review-debate/round-1/[reviewer_id].json"
others_previous: [other 3 reviewers' round-1 outputs]
output_path: ".claude/work/review-debate/round-2/[reviewer_id].json"
```

Wait for all 4 to complete, then create `.claude/work/review-debate/round-2/summary.md`.

### Round 3: Prioritization (Parallel)

Invoke `code-reviewer` subagent 4 times in parallel:

**For each reviewer:**
```
reviewer_id: "[reviewer]"
focus: "[focus]"
round: 3
model: sonnet
critique_mode: "prioritize"
diff_path: ".claude/work/review-debate/current.diff"
own_previous: ".claude/work/review-debate/round-2/[reviewer_id].json"
others_previous: [other 3 reviewers' round-2 outputs]
output_path: ".claude/work/review-debate/round-3/[reviewer_id].json"
```

Wait for all 4 to complete, then create `.claude/work/review-debate/round-3/summary.md`.

---

# Phase 4: Fix Plan Synthesis

Invoke `fix-plan-synthesizer` subagent once:
```
diff_path: ".claude/work/review-debate/current.diff"
diff_assessment: ".claude/work/review-debate/diff-assessment.json"
static_checks_baseline: ".claude/work/review-debate/static-checks-baseline.json"
round_1_reviews: [all 4 reviewer outputs from round-1/]
round_2_reviews: [all 4 reviewer outputs from round-2/]
round_3_reviews: [all 4 reviewer outputs from round-3/]
output_path: ".claude/work/review-debate/fix-plan.json"
```

Expected output schema:
```json
{
  "timestamp": "ISO-8601",
  "consensus_assessment": {
    "critical_issues": 0,
    "high_priority_issues": 0,
    "medium_priority_issues": 0,
    "low_priority_issues": 0,
    "unanimous_findings": [],
    "majority_findings": [],
    "split_findings": []
  },
  "fixes": [
    {
      "priority": "critical|high|medium|low",
      "file": "path/to/file",
      "line": 123,
      "issue": "Description of the issue",
      "fix_description": "What needs to change",
      "old_code": "exact code to replace",
      "new_code": "replacement code",
      "rationale": "Why this fix is needed",
      "risk": "low|medium|high",
      "supporting_reviewers": ["security-reviewer", "correctness-reviewer"]
    }
  ],
  "no_action_needed": false,
  "summary": "Overall assessment"
}
```

---

# Phase 5: Apply Fixes via Subagent

## 5.1 Read Fix Plan

Read `.claude/work/review-debate/fix-plan.json`.

If `no_action_needed: true` → skip to Phase 6.

## 5.2 Invoke Fix Applicator Subagent

Invoke `fix-applicator` subagent once:
```
fix_plan_path: ".claude/work/review-debate/fix-plan.json"
diff_baseline_path: ".claude/work/review-debate/current.diff"
static_checks_baseline_path: ".claude/work/review-debate/static-checks-baseline.json"
output_log_path: ".claude/work/review-debate/fix-execution-log.json"
```

The `fix-applicator` subagent will:
1. Read the fix plan
2. Apply fixes in priority order (critical → high → medium → low)
3. For each fix:
   - Read the target file
   - Locate exact `old_code`
   - Apply `new_code` using Edit tool
   - Track success/failure
4. Re-run static checks after all fixes
5. Compare before/after
6. Revert problematic fixes if checks degrade
7. Write execution log

Expected output schema:
```json
{
  "timestamp": "ISO-8601",
  "fixes_attempted": 0,
  "fixes_succeeded": 0,
  "fixes_failed": 0,
  "fixes_skipped": 0,
  "execution_details": [
    {
      "fix_index": 0,
      "file": "path/to/file",
      "line": 123,
      "status": "success|failed|skipped",
      "error": "error message if failed",
      "reverted": false
    }
  ],
  "static_checks_after": {
    "lint": {
      "error_count": 0,
      "warning_count": 0,
      "clean": true
    },
    "typescript": {
      "error_count": 0,
      "clean": true
    }
  },
  "outcome": "CLEAN|IMPROVED|DEGRADED|UNCHANGED",
  "degraded_fixes_reverted": []
}
```

---

# Phase 6: Final Validation & Completion

## 6.1 Verify Complete Solution

1. Review ALL changes: `git diff main` or equivalent
2. Check integration points and data flow
3. Verify the solution addresses the original request
4. Run any relevant tests

## 6.2 Generate Final Report
```
=== BUILD WITH REVIEW + DEBATE + AUTO-FIX SUMMARY ===

ORIGINAL REQUEST: "$ARGUMENTS"

PHASE 1 & 2: IMPLEMENTATION
  Files Modified: X
  Todos Completed: Y
  Self-Reviews Performed: Z

PHASE 3: MULTI-AGENT DEBATE REVIEW
  Strategy: [small|medium|large]
  Debate Rounds: 3
  Reviewers: 4 (security, correctness, performance, maintainability)
  
  FINDINGS:
    Critical: X
    High: Y
    Medium: Z
    Low: W
  
  CONSENSUS:
    Unanimous: X findings
    Majority: Y findings
    Split: Z findings

PHASE 4: FIX PLAN
  Fixes Generated: X

PHASE 5: AUTO-FIX EXECUTION
  Attempted: X
  Succeeded: Y
  Failed: Z
  Skipped: W
  
  STATIC CHECKS:
    BEFORE: lint X errors, tsc Y errors
    AFTER:  lint A errors (-B), tsc C errors (-D)
  
  OUTCOME: [CLEAN|IMPROVED|DEGRADED|UNCHANGED]

PHASE 6: FINAL VALIDATION
  ✓ Integration verified
  ✓ Requirements met
  ✓ Safety checks passed

ARTIFACTS GENERATED:
  - .claude/work/review-debate/current.diff
  - .claude/work/review-debate/diff-assessment.json
  - .claude/work/review-debate/static-checks-baseline.json
  - .claude/work/review-debate/round-{1,2,3}/*.json
  - .claude/work/review-debate/fix-plan.json
  - .claude/work/review-debate/fix-execution-log.json

NEXT STEPS: [context-appropriate guidance]
```

## 6.3 Present Completion

After completing all phases:
1. Display the final report above
2. Summarize what was built and how it was validated
3. Note any remaining considerations or follow-ups
4. Highlight any critical issues that require human review

---

## Example Todo Structure (Phase 1 & 2)

For a request like "Add error handling to the API client":

**Phase 1 completes investigation first:**
- Located APIClient.ts
- Identified 3 places where errors aren't handled: fetch calls, retry logic, response parsing
- Determined need for typed error classes and user-facing messages
- Found existing logger utility to use

**Then Phase 2 todos (only implementation):**
```
[ ] Fix error handling for network failures in APIClient.ts
    PLAN:
    - Approach: Wrap fetch in try-catch, create NetworkError class
    - Avoid regressions: Ensure existing success paths unchanged
    - Best practices: Use typed errors, maintain error chain

    IMPLEMENT: (then mark complete)

[ ] Review: network failure error handling
    PREPARE: Put on skeptical reviewer hat
    ANALYZE: What could break? What are the failure modes for error handling?
    CHECKLISTS: Apply 4 checklists with thinking
    ACT: Fix any issues found

[ ] Fix error handling for 4xx/5xx responses in APIClient.ts
    PLAN: (write out approach, pitfalls, best practices)
    IMPLEMENT: (then mark complete)

[ ] Review: status code error handling
    PREPARE, ANALYZE, CHECKLISTS, ACT

[ ] Add typed error classes and messages to errors.ts
    PLAN: (write out approach, pitfalls, best practices)
    IMPLEMENT: (then mark complete)

[ ] Review: error classes and messages
    PREPARE, ANALYZE, CHECKLISTS, ACT

[ ] Integrate error logger with new error types
    (This is simple - discovered during implementation, skip review)

[ ] Final Review: Integration - do all error paths work together?
[ ] Final Review: Requirements - is error handling comprehensive?
[ ] Final Review: Safety - test error scenarios
[ ] Fix any final issues
```

Note:
- NO "Locate files" or "Identify issues" tasks - that was done in Phase 1
- Every implementation task shows PLAN then IMPLEMENT
- Every review shows the multi-step process: PREPARE → ANALYZE → CHECKLISTS → ACT
- Simple discovered task (error logger integration) skips review

**Then Phase 3-5 automatically run the debate and fixes**

---

## Safety & Error Handling

- Never modify files outside the scope of the original request and debate findings
- Revert any fixes that worsen static checks
- Apply one fix at a time to enable precise rollback
- Preserve all artifacts for debugging and audit
- Stop on phase failure with clear error message
- Report all subagent failures immediately

## Exit Criteria

Complete when:
- Phase 1: Investigation complete, todos created
- Phase 2: All implementation todos done, all reviews passed
- Phase 3: All 3 debate rounds complete for all 4 reviewers
- Phase 4: Fix plan synthesized
- Phase 5: All fixes attempted, static checks verified
- Phase 6: Final report generated and presented

---

If this command is invocaated and the todo list is completed, the file that creates the command should be deleted.

---

Don't use the cat command; use the Write tool

---

Now begin Phase 1: Analyze the user's request and create the structured todo list.

