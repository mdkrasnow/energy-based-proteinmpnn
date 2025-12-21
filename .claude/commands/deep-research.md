---
description: Conduct comprehensive, multi-perspective research of the codebase to deeply understand architecture, patterns, and relationships
argument-hint: <topic or question to research>
allowed-tools: Read, Grep, Glob, Bash, Cat, Echo, Touch, Git
model: claude-sonnet-4-5-20250929
---

# Deep Research Command

You are conducting **deep research** on a codebase topic or question. This is not a quick lookup - it's a comprehensive, multi-round investigation that synthesizes information from across the codebase to build a thorough understanding.

**User's Research Question:** "$ARGUMENTS"

## Research Protocol

Follow this structured, multi-phase approach to ensure comprehensive coverage:

### Phase 1: Initial Exploration (15% of effort)

**Goal:** Understand the scope and identify key areas to investigate.

1. **Parse the question**:
   - What is the core information need?
   - What files, components, or systems are relevant?
   - What specific aspects need investigation (architecture, patterns, behavior, history)?

2. **Initial reconnaissance**:
   - Use `tree` or `find` to understand directory structure
   - Use `Glob` to identify potentially relevant files
   - Use `git log` to understand recent activity in relevant areas
   - Scan for README files, architecture docs, or comments

3. **Define research scope**:
   - List 3-5 specific sub-questions to answer
   - Identify key files/directories to investigate
   - Note any dependencies or related systems

**Output:** Clear research plan with sub-questions and target areas.

---

### Phase 2: Deep Investigation (50% of effort)

**Goal:** Gather comprehensive data through systematic exploration.

For each sub-question identified in Phase 1:

1. **Locate all relevant code**:
   - Use `Grep` with multiple search terms (synonyms, related concepts)
   - Search for function/class definitions, imports, and usages
   - Find configuration files, tests, and documentation
   - Look for historical context in git history

2. **Read and analyze key files**:
   - Don't just skim - actually read the important files
   - Understand implementation details
   - Note patterns, idioms, and conventions
   - Identify dependencies and relationships

3. **Cross-reference findings**:
   - How do different parts connect?
   - Are there multiple implementations of similar functionality?
   - What are the data flows?
   - Where are the integration points?

4. **Collect evidence**:
   - Note specific file:line references
   - Capture key code snippets
   - Record git commit messages explaining why
   - Document configuration values

**Output:** Comprehensive notes with specific evidence for each sub-question.

---

### Phase 3: Pattern Analysis (20% of effort)

**Goal:** Identify patterns, antipatterns, and architectural insights.

1. **Find patterns**:
   - What design patterns are used? (Observer, Factory, Strategy, etc.)
   - What architectural patterns? (MVC, microservices, event-driven, etc.)
   - What coding conventions? (naming, structure, error handling)
   - What testing patterns?

2. **Identify antipatterns and tech debt**:
   - Code duplication
   - Tight coupling
   - Missing abstractions
   - Inconsistent patterns
   - Deprecated or outdated approaches

3. **Understand evolution**:
   - How has this code changed over time? (`git log`)
   - What were the major refactorings?
   - What issues have been fixed? (search for "fix", "bug", etc. in commits)
   - What future plans exist? (TODOs, FIXMEs, comments)

4. **Evaluate quality**:
   - Test coverage
   - Documentation completeness
   - Error handling robustness
   - Performance considerations

**Output:** Pattern analysis with supporting evidence.

---

### Phase 4: Synthesis (15% of effort)

**Goal:** Create a comprehensive, actionable understanding.

1. **Answer the original question**:
   - Provide a clear, direct answer
   - Support with specific evidence (file:line references)
   - Explain the "why" behind the "what"

2. **Provide context**:
   - Historical context (how we got here)
   - Architectural context (how it fits in the system)
   - Business context (why it matters)

3. **Identify knowledge gaps**:
   - What questions remain unanswered?
   - What areas need more investigation?
   - What assumptions are we making?

4. **Offer insights**:
   - Strengths of the current approach
   - Weaknesses or risks
   - Alternative approaches that exist or could exist
   - Recommendations for improvement

5. **Create visual aids** (if helpful):
   - ASCII diagrams of architecture
   - Call graphs
   - Dependency trees
   - Timeline of changes

**Output:** Comprehensive research report.

---

## Research Report Format

Structure your findings as follows:

```markdown
# Deep Research: [Topic]

## Executive Summary
2-3 paragraphs answering the core question with key findings.

## Research Scope
- Original question
- Sub-questions investigated
- Files/systems analyzed
- Time period examined (if relevant)

## Key Findings

### Finding 1: [Title]
**Evidence:** 
- `src/path/to/file.ts:123` - [description of what this shows]
- `another/file.py:45-67` - [description]

**Analysis:**
[What this means, why it matters, how it connects to other findings]

**Confidence:** High/Medium/Low
[Why this confidence level]

### Finding 2: [Title]
[Same structure...]

## Patterns Identified

### Design Patterns
- Pattern X used in [locations] for [purpose]
- Pattern Y appears in [contexts]

### Architectural Patterns
[Description with evidence]

### Antipatterns & Tech Debt
- Issue 1: [description with file references]
- Issue 2: [description]

## Timeline & Evolution
[If relevant - how this code has changed]

## Connections & Dependencies
[How this relates to other parts of the system]

## Knowledge Gaps & Uncertainties
- What we couldn't determine: [list]
- What needs more investigation: [list]
- Assumptions made: [list]

## Recommendations
1. **[Recommendation]**: [Rationale and specific actions]
2. **[Recommendation]**: [Rationale]

## Additional Context
[Any other relevant information]

## Sources Consulted
- Files read: [count] files across [count] directories
- Git history: [date range], [commit count] commits examined
- Lines of code analyzed: ~[estimate]
- Key directories: [list]

## Appendix (if needed)
[Additional code snippets, diagrams, or detailed evidence]
```

---

## Research Best Practices

**Be Thorough:**
- Don't stop at the first answer - investigate multiple perspectives
- Read actual code, don't just pattern match on names
- Consider edge cases and error paths
- Look at tests to understand intended behavior

**Be Systematic:**
- Follow the phases in order
- Document your search strategy (what terms, what tools)
- Keep notes as you go
- Revisit earlier findings with new context

**Be Evidence-Based:**
- Always include specific file:line references
- Quote relevant code snippets
- Link to git commits when relevant
- Distinguish between facts (what the code does) and interpretations (why/how)

**Be Honest:**
- Report confidence levels accurately
- Admit when you can't find something
- Note competing interpretations
- Flag speculation vs. certainty

**Think Like a Researcher:**
- Ask "why?" repeatedly
- Look for connections and patterns
- Consider historical context
- Think about stakeholder perspectives

---

## Example Research Strategies

**For Architecture Questions:**
```bash
# Find entry points
Glob "**/*main*.{ts,js,py}" "**/*index*.{ts,js,py}" "**/app.{ts,js,py}"

# Find dependencies
Grep "import.*from" --include="*.ts" --include="*.js"
Grep "^from .* import" --include="*.py"

# Understand structure
tree -L 3 -d

# Find initialization
Grep "constructor|__init__|initialize" -n
```

**For Behavior Questions:**
```bash
# Find function definitions
Grep "function functionName|def functionName|const functionName" -n

# Find all usages
Grep "functionName" --include="*.ts" --include="*.js" -n

# Find tests
Glob "**/*.test.{ts,js}" "**/*.spec.{ts,js}" "**/test_*.py"
Grep "describe.*functionName|test.*functionName" -n

# Find error handling
Grep "try|catch|except|error|Error" -A 2 -B 2
```

**For Pattern Questions:**
```bash
# Find similar implementations
Grep "class.*Strategy|class.*Factory|class.*Observer" -n

# Find configuration
Glob "**/*.config.{ts,js,json}" "**/*.yaml" "**/*.yml" "**/config/*"

# Find environment-specific code
Grep "process.env|os.environ|NODE_ENV|ENVIRONMENT" -n
```

**For History Questions:**
```bash
# Find recent changes
git log --oneline --since="6 months ago" -- path/to/area

# Find major refactorings
git log --all --oneline --grep="refactor" -- path/to/area

# Find when feature was added
git log --all --oneline --grep="keyword" 

# See file history
git log --follow -- path/to/file
```

---

## Cost & Time Considerations

This command is designed for **deep investigation**, not quick lookups:

- **Expected time:** 10-30 minutes depending on scope
- **Expected searches:** 20-50 grep/glob operations
- **Expected files read:** 10-100 files
- **Model:** Using Sonnet 4.5 for balanced performance and cost

For quick lookups, use standard chat. For deep research, invest the time to do it right.

---

## Output Reminder

At the end, write your comprehensive research report to a markdown file and provide the link:

`~/documentation/reports/[topic]-[date].md`

Now begin Phase 1: Initial Exploration of the user's research question.