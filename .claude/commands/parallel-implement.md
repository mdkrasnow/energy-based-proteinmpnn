---
description: Execute parallelizable implementation tasks using up to 5 independent subagents, each following build-with-review workflow
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, SlashCommand
---

# Parallel Implementation Orchestrator

You are orchestrating parallel implementation of tasks from the project plan and todo list.

## Phase 1: Parse and Analyze

### 1.1 Read Planning Documents

Read both required files:
- `documentation/implementation-plan.md` - overall strategy and context
- `documentation/implementation-todo.md` - specific tasks with checkboxes

### 1.2 Identify Parallelizable Tasks

Analyze the todo list to identify tasks that can run in parallel:

**Independence Criteria:**
- ✅ Tasks that modify different files
- ✅ Tasks with no shared dependencies
- ✅ Tasks that don't require sequential ordering
- ❌ Tasks that depend on each other's outputs
- ❌ Tasks that modify overlapping code sections

**Categorize tasks:**
1. **Immediately parallelizable** - can start now, no dependencies
2. **Blocked** - waiting on other tasks
3. **Sequential** - must run after specific tasks complete

**Select up to 5 tasks** for this batch:
- Prioritize highest-value independent tasks
- Prefer simpler tasks when uncertainty is high
- Consider effort estimates and balance workload

### 1.3 Create Work Assignments

For each selected task, prepare a structured assignment:

```json
{
  "task_id": "unique identifier from todo",
  "task_description": "Full description from todo",
  "affected_files": ["list", "of", "files"],
  "dependencies_met": true,
  "context": {
    "related_todos": ["other relevant todos"],
    "architectural_notes": "key context from plan",
    "constraints": ["specific requirements"]
  },
  "output_path": ".claude/work/parallel/task-{id}-result.json"
}
```

Write each assignment to `.claude/work/parallel/task-{id}-assignment.json`

## Phase 2: Execute in Parallel

### 2.1 Create Work Directory

```bash
mkdir -p .claude/work/parallel
```

### 2.2 Spawn Subagents

For each of the selected tasks (up to 5), invoke the `implementation-worker` subagent:

**Invocation pattern:**
```
Use the implementation-worker subagent to complete the task defined in 
.claude/work/parallel/task-{id}-assignment.json. 

The subagent should:
1. Read its assignment file
2. Follow rigorous build-with-review internally
3. Report detailed results to .claude/work/parallel/task-{id}-result.json

Assignment file: .claude/work/parallel/task-{id}-assignment.json
Expected output: .claude/work/parallel/task-{id}-result.json
```

**Launch all subagents before waiting for results.** This enables true parallelization.

### 2.3 Monitor and Collect

After launching all subagents, wait for completion. Then read all result files:
- `.claude/work/parallel/task-*-result.json`

**Expected result schema:**
```json
{
  "task_id": "string",
  "status": "completed|partial|blocked|failed",
  "confidence": 0.85,
  "
_completed": [
    {
      "description": "what was accomplished",
      "files_modified": ["list"],
      "verification": "how we know it works"
    }
  ],
  "work_remaining": [
    {
      "description": "what still needs work",
      "why_incomplete": "reason",
      "estimated_effort": "effort estimate"
    }
  ],
  "issues_encountered": [
    {
      "severity": "critical|high|medium|low",
      "description": "issue description",
      "persistent": true,
      "workaround": "if any"
    }
  ],
  "potential_future_issues": [
    {
      "concern": "what might go wrong",
      "likelihood": "high|medium|low",
      "mitigation": "how to address"
    }
  ],
  "review_notes": {
    "needs_close_review": ["areas requiring careful review"],
    "functional_areas": ["what's working well"],
    "fragile_areas": ["what's brittle or risky"]
  },
  "git_changes": {
    "files_added": [],
    "files_modified": [],
    "files_deleted": []
  }
}
```

## Phase 3: Consolidate and Update

### 3.1 Synthesize Results

For each completed subagent task, extract:
- **Completion percentage**: Based on status and work_remaining
- **Confidence score**: From the subagent's self-assessment
- **Functional vs incomplete**: What works vs what needs more work
- **Issues to track**: Persistent problems or future concerns
- **Review priorities**: What needs human attention

### 3.2 Update Todo File

Read `documentation/implementation-todo.md` and update it:

**For each task that was worked on:**

1. **If fully completed with high confidence (>0.8):**
   ```
   - [Tentatively completed] Original task description
     **Status**: Completed with {confidence}% confidence
     **Implementation**: {brief summary of what was done}
     **Functional**: {what's working}
     **Verification**: {how it was tested/checked}
     **Review needed**: {areas for human review}
   ```

2. **If partially completed:**
   ```
   - [Partial - {percentage}%] Original task description
     **Status**: Partially completed - {confidence}% confidence
     **Completed**: {what was accomplished}
     **Remaining**: {what still needs work}
     **Issues**: {problems encountered}
     **Next steps**: {what to do next}
   ```

3. **If blocked or failed:**
   ```
   - [ ] Original task description
     **Status**: Blocked/Failed
     **Issue**: {why it couldn't be completed}
     **Required**: {what's needed to unblock}
   ```

**Add consolidated notes section** at the top or bottom of relevant task groups:

```markdown
## Batch Implementation Notes - {timestamp}

### Tasks Attempted ({count})
- Task 1: {status summary}
- Task 2: {status summary}
...

### Overall Success Metrics
- Tasks completed: {X}/{Y}
- Average confidence: {Z}%
- Files modified: {count}

### Persistent Issues Requiring Attention
1. {Issue from any task that's marked persistent}
2. ...

### Potential Future Issues
1. {Concern that came up during implementation}
2. ...

### High-Priority Review Items
1. {Areas marked for close review}
2. ...
```

### 3.3 Write Batch Summary

Create `.claude/work/parallel/batch-summary.md` with:

```markdown
# Parallel Implementation Batch - {timestamp}

## Tasks Executed
{List of tasks with one-line summaries}

## Success Metrics
- Completion rate: {X}/{Y} tasks
- Average confidence: {Z}%
- Files modified: {count}
- Lines changed: ~{estimate}

## Key Accomplishments
1. {Major win from task 1}
2. {Major win from task 2}
...

## Challenges Encountered
1. {Significant issue from any task}
2. ...

## Next Recommended Actions
1. {Based on what was learned}
2. {What should happen next}
3. {What to review first}

## Detailed Task Results

### Task 1: {description}
- Status: {completed/partial/failed}
- Confidence: {X}%
- Files: {list}
- Notes: {key points}

### Task 2: {description}
...
```

## Phase 4: Recommendations

Based on the results, provide strategic guidance:

**If most tasks completed successfully:**
- "Ready for next batch. Recommended: {suggest next 3-5 parallelizable tasks}"

**If many tasks blocked:**
- "Several dependencies need resolution. Recommended: {list blocking tasks to handle sequentially}"

**If confidence is low across tasks:**
- "Consider more investigation before next batch. Recommended: {suggest deeper analysis}"

**If critical issues emerged:**
- "ATTENTION NEEDED: {list critical issues}. Recommend addressing before continuing."

## Orchestrator Guidelines

**Parallelization Strategy:**
- Start with 3-4 tasks if uncertain about independence
- Scale to 5 for clearly independent tasks
- Drop to 1-2 if tasks are tightly coupled

**Assignment Quality:**
- Give subagents complete context from plan
- Specify exact files to modify
- Highlight constraints and gotchas
- Link to related documentation

**Result Interpretation:**
- Trust subagent confidence scores - they know what they struggled with
- "Partial" is acceptable and valuable - it's progress
- Persistent issues should bubble up to human attention
- Low confidence on "completed" tasks means "tentative" - needs review

**Communication:**
- Be explicit about what happened
- Don't sugarcoat failures or uncertainties
- Quantify progress (percentages, metrics)
- Provide actionable next steps

## Error Handling

**If a subagent fails to produce output:**
- Note the failure in the todo update
- Mark task as blocked
- Include the error in batch summary

**If result JSON is malformed:**
- Attempt to parse what you can
- Mark task status as uncertain
- Flag for human review

**If subagents modified overlapping code:**
- Note the conflict in batch summary
- Mark affected tasks for careful review
- Recommend manual conflict resolution

## Exit Criteria

You are done when:
1. ✓ All selected tasks have been attempted by subagents
2. ✓ All result files have been read and parsed
3. ✓ `documentation/implementation-todo.md` is updated with detailed status
4. ✓ Batch summary is written to `.claude/work/parallel/batch-summary.md`
5. ✓ Recommendations for next steps are provided

---

Now begin Phase 1: Read the planning documents and identify parallelizable tasks.