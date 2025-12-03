---
description: Review tentatively completed todos via multi-round debate, then fix identified issues with rigorous validation
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, SlashCommand
---

# Review and Fix Orchestrator

You are orchestrating a rigorous multi-stage review and fix process for tentatively completed work.

## Overview

This system:
1. **Identifies** todos marked `[Tentatively completed]`
2. **Reviews** each via multi-round internal debate (critical-reviewer subagent)
3. **Prioritizes** issues found across all reviews
4. **Fixes** high-priority issues (issue-fixer subagent for each item)
5. **Updates** todos with final status and confidence

## Phase 1: Identify and Prepare

### 1.1 Read Planning Context

Read both files to understand the project:
- `documentation/implementation-plan.md` - overall architecture and constraints
- `documentation/implementation-todo.md` - tasks and their current status

### 1.2 Extract Tentatively Completed Items

Scan `implementation-todo.md` for entries marked:
- `[Tentatively completed]`

For each, extract:
- Original task description
- Current confidence level
- What was implemented
- What's functional
- What needs work
- Known issues
- Areas flagged for review

**Create a review manifest:**

```json
{
  "review_batch_id": "review-{timestamp}",
  "items_to_review": [
    {
      "item_id": "auth-error-handling",
      "description": "Add error handling to authentication flow",
      "current_confidence": 0.85,
      "implementation_summary": "Added try-catch, error classes, logging",
      "flagged_concerns": [
        "Error message security",
        "Integration with error boundaries"
      ],
      "files_modified": ["src/auth/authService.ts", "src/types/errors.ts"],
      "review_priority": "high"
    }
  ]
}
```

Write to: `.claude/work/review/manifest.json`

### 1.3 Create Review Assignments

For each item to review, create an assignment file:

```json
{
  "item_id": "auth-error-handling",
  "original_task": "Add comprehensive error handling to authentication flow",
  "implementation_details": {
    "confidence": 0.85,
    "completed_work": "Full description from todo",
    "functional_areas": ["List from todo"],
    "needs_work": ["List from todo"],
    "flagged_concerns": ["List from todo"],
    "files_modified": ["List"]
  },
  "review_context": {
    "architectural_notes": "Relevant notes from plan",
    "constraints": ["From plan"],
    "related_todos": ["Other relevant items"]
  },
  "output_path": ".claude/work/review/item-{id}-review-result.json"
}
```

Write to: `.claude/work/review/item-{id}-assignment.json`

## Phase 2: Multi-Round Reviews (Parallel)

### 2.1 Create Review Directory

```bash
mkdir -p .claude/work/review
```

### 2.2 Spawn Review Subagents

For each tentatively completed item, invoke the `critical-reviewer` subagent:

**Invocation pattern:**
```
Use the critical-reviewer subagent to perform a rigorous 4-round debate review
of the implementation described in .claude/work/review/item-{id}-assignment.json.

The subagent will:
1. Round 1: Find ALL potential issues (critical perspective)
2. Round 2: Validate/invalidate Round 1, find what was missed (devil's advocate)
3. Round 3: Rank issues by priority and impact (triage perspective)
4. Round 4: Final validation and confidence assessment (synthesis perspective)

Assignment: .claude/work/review/item-{id}-assignment.json
Output: .claude/work/review/item-{id}-review-result.json
```

**Launch all review subagents in parallel** (up to 5 at once).

### 2.3 Collect Review Results

After all reviews complete, read all result files:
- `.claude/work/review/item-*-review-result.json`

**Expected review result schema:**
```json
{
  "item_id": "string",
  "review_rounds_completed": 4,
  "final_assessment": {
    "overall_quality": "excellent|good|acceptable|poor",
    "confidence_in_implementation": 0.90,
    "confidence_change": +0.05,
    "confidence_reasoning": "Why this score changed or stayed same"
  },
  "issues_found": [
    {
      "issue_id": "auth-err-1",
      "severity": "critical|high|medium|low|negligible",
      "category": "correctness|security|performance|maintainability|style",
      "description": "Detailed issue description",
      "location": "file:line or component",
      "impact": "What breaks or degrades",
      "fix_priority": "must-fix|should-fix|nice-to-fix|wont-fix",
      "estimated_effort": "time estimate",
      "discovered_in_round": 1,
      "validated_in_rounds": [2, 3, 4],
      "false_positive": false,
      "reasoning": "Why this is/isn't a real issue"
    }
  ],
  "validation_summary": {
    "round_1_issues": 12,
    "round_2_confirmed": 8,
    "round_2_rejected": 4,
    "round_3_must_fix": 3,
    "round_3_should_fix": 3,
    "round_3_wont_fix": 2,
    "round_4_final_must_fix": 2,
    "round_4_final_should_fix": 4
  },
  "strengths_identified": [
    "What's working well that shouldn't be changed"
  ],
  "review_notes": "Overall assessment and recommendations"
}
```

## Phase 3: Consolidate and Prioritize

### 3.1 Cross-Review Analysis

Analyze all review results together:

**Look for patterns:**
- Issues appearing in multiple reviews (cross-cutting concerns)
- Common issue categories (security, error handling, etc.)
- Items with low final confidence (need more work)
- Items with no must-fix issues (ready to finalize)

**Create priority matrix:**

```json
{
  "critical_path_items": [
    {
      "item_id": "auth-error-handling",
      "must_fix_count": 2,
      "total_estimated_effort": "3 hours",
      "blocking_reason": "Security vulnerability"
    }
  ],
  "high_priority_items": [...],
  "acceptable_items": [
    {
      "item_id": "logging-middleware",
      "must_fix_count": 0,
      "should_fix_count": 1,
      "ready_to_finalize": true
    }
  ]
}
```

### 3.2 Create Fix Assignments

For each item with must-fix or should-fix issues:

```json
{
  "item_id": "auth-error-handling",
  "original_task": "Add error handling to auth",
  "issues_to_fix": [
    {
      "issue_id": "auth-err-1",
      "severity": "high",
      "description": "Error messages expose internal implementation details",
      "location": "src/auth/authService.ts:145-160",
      "fix_priority": "must-fix",
      "suggested_approach": "Use generic error messages for external consumption",
      "validation_criteria": "Error messages don't reveal stack traces or internal paths"
    }
  ],
  "context": {
    "files_to_modify": ["src/auth/authService.ts"],
    "constraints": ["Don't break existing error handling"],
    "architectural_notes": "Follow existing error pattern"
  },
  "output_path": ".claude/work/review/item-{id}-fix-result.json"
}
```

Write to: `.claude/work/review/item-{id}-fix-assignment.json`

## Phase 4: Apply Fixes (Parallel)

### 4.1 Spawn Fixer Subagents

For each item requiring fixes, invoke the `issue-fixer` subagent:

**Invocation pattern:**
```
Use the issue-fixer subagent to address the issues identified in review
for .claude/work/review/item-{id}-fix-assignment.json.

The subagent will:
1. Analyze issues in context
2. Plan fixes using build-with-review approach
3. Implement fixes with critical self-review
4. Validate fixes against original issues
5. Report detailed results

Assignment: .claude/work/review/item-{id}-fix-assignment.json
Output: .claude/work/review/item-{id}-fix-result.json
```

**Launch fixers in parallel** (up to 5 at once).

### 4.2 Collect Fix Results

Read all fix result files:
- `.claude/work/review/item-*-fix-result.json`

**Expected fix result schema:**
```json
{
  "item_id": "string",
  "fix_session_complete": true,
  "issues_addressed": [
    {
      "issue_id": "auth-err-1",
      "status": "fixed|partially-fixed|not-fixed|obsolete",
      "changes_made": "Description of fix",
      "files_modified": ["list"],
      "verification": "How we know it's fixed",
      "confidence": 0.95
    }
  ],
  "new_issues_discovered": [
    {
      "description": "Issue found while fixing",
      "severity": "high|medium|low",
      "addressed": false
    }
  ],
  "final_state": {
    "all_must_fix_resolved": true,
    "all_should_fix_resolved": false,
    "remaining_work": ["What's still needed"],
    "confidence": 0.92
  },
  "git_changes": {
    "files_modified": ["list"],
    "lines_changed": 45
  }
}
```

## Phase 5: Update Todos and Report

### 5.1 Update Todo Status

Read `documentation/implementation-todo.md` and update each reviewed item:

**For items with all must-fix issues resolved:**

```markdown
- [✓ Completed] {Original task description}
  **Status**: Completed after review with {confidence}% confidence
  **Implementation**: {original implementation summary}
  **Review process**: 4-round debate found {N} issues
  **Issues fixed**: {must-fix count} critical/high priority issues resolved
  **Final state**: {what's now working, what was improved}
  **Remaining minor items**: {should-fix items not addressed, if any}
  **Review confidence**: {final confidence from review}
  **Ready for**: Production deployment / Further testing / [specify]
```

**For items with unresolved must-fix issues:**

```markdown
- [⚠ Needs rework] {Original task description}
  **Status**: Issues found in review - requires additional work
  **Review process**: 4-round debate found {N} critical issues
  **Must fix**: {list of unresolved must-fix issues}
  **Current confidence**: {lowered confidence}
  **Next steps**: {specific actions needed}
  **Blocking**: {what this blocks}
```

**For items that passed review with no issues:**

```markdown
- [✓ Completed] {Original task description}
  **Status**: Completed - passed rigorous review with {confidence}% confidence
  **Implementation**: {original implementation summary}
  **Review process**: 4-round debate found no critical issues
  **Strengths**: {what reviewers praised}
  **Minor suggestions**: {nice-to-fix items for future}
  **Ready for**: Production deployment
```

### 5.2 Create Batch Review Report

Write `.claude/work/review/batch-report.md`:

```markdown
# Review and Fix Batch Report - {timestamp}

## Executive Summary

- Items reviewed: {N}
- Items passed review: {N}
- Items requiring fixes: {N}
- Items fixed successfully: {N}
- Items needing rework: {N}

## Success Metrics

- Average confidence before review: {X}%
- Average confidence after fixes: {Y}%
- Confidence improvement: {+Z}%
- Total issues found: {N}
- Must-fix issues: {N} (all resolved: {yes/no})
- Should-fix issues: {N} ({X} resolved)

## Review Process Quality

- Average rounds to identify all issues: {X}
- False positive rate: {Y}% (issues flagged then rejected)
- Issues missed in Round 1 but caught in Round 2: {N}
- Issues caught in final Round 4 validation: {N}

## Detailed Results

### Items Completed Successfully

#### {Item 1}
- Original confidence: {X}%
- Final confidence: {Y}%
- Issues found: {N} ({M} fixed)
- Status: ✓ Ready for production

### Items Requiring Rework

#### {Item 2}
- Original confidence: {X}%
- Current confidence: {Y}%
- Critical issues remaining: {N}
- Blocking: {what this blocks}
- Estimated rework: {time}

## Cross-Cutting Issues Discovered

1. **{Pattern 1}**: Found in {N} items
   - Impact: {description}
   - Recommendation: {what to do}

2. **{Pattern 2}**: Found in {N} items
   - Impact: {description}
   - Recommendation: {what to do}

## Strengths Observed

1. **{Strength 1}**: Consistent across items
2. **{Strength 2}**: Particularly strong in {items}

## Recommendations

### Immediate Actions
1. {Action for items needing rework}
2. {Action for blocking issues}

### Strategic Improvements
1. {Pattern to address in future work}
2. {Process improvement based on review findings}

### Ready to Proceed
- Items ready for production: {list}
- Items ready for integration: {list}
- Items requiring more work: {list}

## Appendix: Issue Details

### All Must-Fix Issues
{Table of all must-fix issues across all items}

### All Should-Fix Issues
{Table of all should-fix issues}

### All Issues by Category
- Security: {N}
- Correctness: {N}
- Performance: {N}
- Maintainability: {N}
```

### 5.3 Provide Strategic Guidance

Based on results, give recommendations:

**If all items passed or were fixed successfully:**
```
✓ Batch Review Complete - High Quality Achieved

All tentatively completed items have been rigorously reviewed and issues 
addressed. Average confidence improved from {X}% to {Y}%.

READY FOR:
- Integration testing
- Production deployment preparation
- Next implementation batch

NEXT STEPS:
1. Review batch report for patterns
2. Consider addressing nice-to-fix items
3. Proceed with next phase of implementation
```

**If critical issues remain:**
```
⚠ Attention Required - Critical Issues Found

Review process identified {N} critical issues that must be addressed 
before proceeding.

BLOCKING ITEMS:
- {Item 1}: {critical issue}
- {Item 2}: {critical issue}

IMMEDIATE ACTIONS:
1. Address must-fix issues in blocking items
2. Re-run review-and-fix on those items
3. Do not proceed to integration until resolved

ESTIMATED TIME: {hours/days}
```

**If patterns emerge:**
```
📊 Patterns Detected Across Reviews

The review process identified recurring issues:

1. {Pattern}: Found in {N} items
   → Recommendation: {architectural or process change}

2. {Pattern}: Found in {N} items
   → Recommendation: {standard or guideline to add}

Consider addressing these systematically before next implementation batch.
```

## Orchestrator Guidelines

### Review Parallelization

- **Up to 5 reviews in parallel** if that many items need review
- Reviews are independent - safe to parallelize
- Each reviewer runs its own 4-round internal debate

### Fix Parallelization

- **Up to 5 fixes in parallel** for items with non-overlapping files
- **Sequential fixes** if items modify the same files
- Check file overlap before parallelizing

### Quality Thresholds

**Confidence thresholds:**
- ≥0.90: Excellent, ready for production
- 0.80-0.90: Good, minor concerns addressed
- 0.70-0.80: Acceptable, some should-fix items remain
- <0.70: Needs rework, critical issues likely

**Issue severity mapping:**
- Critical: Blocks deployment, security risk, data loss
- High: Significant impact, should fix before release
- Medium: Notable improvement, fix when convenient
- Low: Nice to have, future improvement
- Negligible: Style/preference, not worth fixing

### Error Handling

**If review subagent fails:**
- Mark item for manual review
- Include error in batch report
- Don't block other items

**If fixer fails:**
- Mark issues as not-fixed
- Lower confidence score
- Flag for manual intervention

**If results are ambiguous:**
- Conservative approach: mark for manual review
- Include uncertainty in report
- Don't claim fixes that aren't verified

## Exit Criteria

You are done when:
1. ✓ All tentatively completed items have been reviewed
2. ✓ All review results collected and analyzed
3. ✓ Fixes attempted for all must-fix issues
4. ✓ Fix results collected and verified
5. ✓ `documentation/implementation-todo.md` updated with final status
6. ✓ Batch report written with detailed findings
7. ✓ Strategic recommendations provided

---

Now begin Phase 1: Identify tentatively completed items and prepare for review.