---
description: Analyze request, create alternating implementation/review todos, and validate all changes with critical self-review
argument-hint: <user request>
allowed-tools: Read, Write, Edit, Grep, Glob, Bash(git *), TodoManager
---

You are implementing a rigorous build-and-review workflow. The user's request is: "$ARGUMENTS"

**WORKFLOW OVERVIEW:**
1. **Investigate thoroughly upfront** (Phase 1) - locate files, understand code, identify issues
2. **Create implementation-only todos** (Phase 1) - no analysis tasks, only concrete code changes
3. **Plan before each implementation** (Phase 2) - explicit approach, pitfalls, best practices
4. **Review with critical analysis** (Phase 2) - think about what to look for, then apply checklists systematically
5. **Adapt as you discover** (Phase 2) - update todos when implementation reveals new needs

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
   - **Exception**: Skip review for trivial changes (judgment call - see Phase 2)

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

# Phase 2: Execute

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

# Phase 3: Completion

After all todos are done:
1. Run a final sanity check
2. Summarize what was built and how it was validated
3. Note any remaining considerations or follow-ups

---

## Example Todo Structure

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

---


Now begin Phase 1: Analyze the user's request and create the structured todo list.
