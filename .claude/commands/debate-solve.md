---
description: Solve a problem using multi-agent debate (MAD) to find the optimal solution through 3 rounds of parallel deliberation
argument-hint: <problem-description>
allowed-tools: Write, Read, SubAgent, Bash, Bash(cat:*), Touch, Echo
model: claude-sonnet-4-5
---

# Multi-Agent Debate Solver

You are the orchestrator for a Multi-Agent Debate (MAD) system. Your job is to coordinate 3 solution agents through 3 rounds of debate to find the optimal solution to a problem.

## Input
Problem to solve: "$ARGUMENTS"

## Process Overview

You will coordinate a **fixed 3-round debate** with **parallel agent execution**:
- **Round 1**: Independent generation (3 agents, Haiku, parallel)
- **Round 2**: Adversarial critique (3 agents, Sonnet, parallel)
- **Round 3**: Cooperative synthesis (3 agents, Sonnet, parallel)
- **Final**: Single synthesis agent produces the final solution (Sonnet)

## Execution Steps

### Step 1: Setup Workspace

Create the debate workspace:

```bash
mkdir -p .claude/work/debate/round-1
mkdir -p .claude/work/debate/round-2
mkdir -p .claude/work/debate/round-3
```

Write the problem statement to `.claude/work/debate/problem.md`:

```markdown
# Problem Statement

{user's problem description from $ARGUMENTS}

---
Generated: {timestamp}
Debate ID: {generate random ID}
```

### Step 2: Round 1 - Independent Generation (Parallel, Haiku)

Invoke the `solution-debater` sub-agent **3 times in parallel** with different configurations:

**Agent 1 Configuration:**
- agent_id: "agent-1"
- perspective: "simplicity"
- round: 1
- model: haiku
- critique_mode: "none"
- output_path: ".claude/work/debate/round-1/agent-1.json"

**Agent 2 Configuration:**
- agent_id: "agent-2"
- perspective: "robustness"
- round: 1
- model: haiku
- critique_mode: "none"
- output_path: ".claude/work/debate/round-1/agent-2.json"

**Agent 3 Configuration:**
- agent_id: "agent-3"
- perspective: "maintainability"
- round: 1
- model: haiku
- critique_mode: "none"
- output_path: ".claude/work/debate/round-1/agent-3.json"

**How to invoke in parallel:**

For each agent, create a prompt that includes:
1. The problem statement from `.claude/work/debate/problem.md`
2. The agent configuration (id, perspective, round, critique_mode)
3. Instruction to write output to the specified path

Invoke all 3 agents and **wait for all to complete** before proceeding to Round 2.

**After Round 1 completes:**
- Read all 3 output files
- Create `.claude/work/debate/round-1/summary.md` with a brief overview of the 3 proposed solutions
- Note key differences and commonalities

### Step 3: Round 2 - Adversarial Critique (Parallel, Sonnet)

Invoke the `solution-debater` sub-agent **3 times in parallel** with adversarial critique mode:

**Agent 1 Configuration:**
- agent_id: "agent-1"
- perspective: "simplicity"
- round: 2
- model: sonnet
- critique_mode: "adversarial"
- own_previous: ".claude/work/debate/round-1/agent-1.json"
- others_previous: [".claude/work/debate/round-1/agent-2.json", ".claude/work/debate/round-1/agent-3.json"]
- output_path: ".claude/work/debate/round-2/agent-1.json"

**Agent 2 Configuration:**
- agent_id: "agent-2"
- perspective: "robustness"
- round: 2
- model: sonnet
- critique_mode: "adversarial"
- own_previous: ".claude/work/debate/round-1/agent-2.json"
- others_previous: [".claude/work/debate/round-1/agent-1.json", ".claude/work/debate/round-1/agent-3.json"]
- output_path: ".claude/work/debate/round-2/agent-2.json"

**Agent 3 Configuration:**
- agent_id: "agent-3"
- perspective: "maintainability"
- round: 2
- model: sonnet
- critique_mode: "adversarial"
- own_previous: ".claude/work/debate/round-1/agent-3.json"
- others_previous: [".claude/work/debate/round-1/agent-1.json", ".claude/work/debate/round-1/agent-2.json"]
- output_path: ".claude/work/debate/round-2/agent-3.json"

Invoke all 3 agents and **wait for all to complete** before proceeding to Round 3.

**After Round 2 completes:**
- Read all 3 output files
- Create `.claude/work/debate/round-2/summary.md` noting:
  - What critiques emerged
  - How solutions evolved
  - Areas of growing consensus vs persistent disagreement

### Step 4: Round 3 - Cooperative Synthesis (Parallel, Sonnet)

Invoke the `solution-debater` sub-agent **3 times in parallel** with cooperative mode:

**Agent 1 Configuration:**
- agent_id: "agent-1"
- perspective: "simplicity"
- round: 3
- model: sonnet
- critique_mode: "cooperative"
- own_previous: ".claude/work/debate/round-2/agent-1.json"
- others_previous: [".claude/work/debate/round-2/agent-2.json", ".claude/work/debate/round-2/agent-3.json"]
- output_path: ".claude/work/debate/round-3/agent-1.json"

**Agent 2 Configuration:**
- agent_id: "agent-2"
- perspective: "robustness"
- round: 3
- model: sonnet
- critique_mode: "cooperative"
- own_previous: ".claude/work/debate/round-2/agent-2.json"
- others_previous: [".claude/work/debate/round-2/agent-1.json", ".claude/work/debate/round-2/agent-3.json"]
- output_path: ".claude/work/debate/round-3/agent-2.json"

**Agent 3 Configuration:**
- agent_id: "agent-3"
- perspective: "maintainability"
- round: 3
- model: sonnet
- critique_mode: "cooperative"
- own_previous: ".claude/work/debate/round-2/agent-3.json"
- others_previous: [".claude/work/debate/round-2/agent-1.json", ".claude/work/debate/round-2/agent-2.json"]
- output_path: ".claude/work/debate/round-3/agent-3.json"

Invoke all 3 agents and **wait for all to complete** before proceeding to synthesis.

**After Round 3 completes:**
- Read all 3 output files
- Create `.claude/work/debate/round-3/summary.md` noting final convergence

### Step 5: Final Synthesis

Invoke the `synthesis-agent` sub-agent **once** with the complete debate history:

**Synthesis Configuration:**
- problem: ".claude/work/debate/problem.md"
- round_1_outputs: all 3 agent outputs from round-1/
- round_2_outputs: all 3 agent outputs from round-2/
- round_3_outputs: all 3 agent outputs from round-3/
- output_path: ".claude/work/debate/final-synthesis.md"

The synthesis agent will produce the final, optimal solution.

### Step 6: Present Results

After synthesis completes:

1. **Read** `.claude/work/debate/final-synthesis.md`

2. **Present to user:**
   ```markdown
   # 🎯 Multi-Agent Debate Results

   After 3 rounds of debate with 3 solution agents, here is the optimal solution:

   {paste the Executive Summary section from final-synthesis.md}

   📄 **Full Solution Plan**: [View final-synthesis.md](computer:///.claude/work/debate/final-synthesis.md)

   ## Debate Process
   - ✅ Round 1: Independent generation (3 diverse perspectives)
   - ✅ Round 2: Adversarial critique (identified weaknesses)
   - ✅ Round 3: Cooperative synthesis (refined consensus)

   ## Debate Transcripts
   You can review the full debate history:
   - [Round 1 Summary](computer:///.claude/work/debate/round-1/summary.md)
   - [Round 2 Summary](computer:///.claude/work/debate/round-2/summary.md)
   - [Round 3 Summary](computer:///.claude/work/debate/round-3/summary.md)

   Would you like me to:
   1. Show the detailed debate transcript?
   2. Explain any specific decision?
   3. Proceed with implementing this solution?
   ```

## Implementation Notes

**Parallel Execution:**
When invoking agents "in parallel," you should:
1. Prepare prompts for all 3 agents
2. Invoke each agent via the SubAgent tool
3. Do NOT wait for one to finish before starting the next
4. After all 3 are invoked, gather results from their output files

**Model Selection:**
- Round 1: Use `model: claude-3-5-haiku` (fast, diverse)
- Round 2-3: Use `model: claude-sonnet-4-5` (deep reasoning)
- Synthesis: Use `model: claude-sonnet-4-5` (high quality)

**Error Handling:**
If any agent fails to write output:
- Note the failure in the summary
- Continue with available outputs
- Flag this in the final synthesis

## Exit Criteria

You are done when:
- All 3 rounds complete
- Synthesis produces `.claude/work/debate/final-synthesis.md`
- User is presented with the final solution

This is **planning only**. The output is a detailed solution plan, NOT implemented code.