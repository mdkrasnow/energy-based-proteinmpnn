## Command Usage

/build-with-review 

<task>
Based on the implementation plan and the summary of the proposal, I want you to carry out the following step of the implementation plan. Make sure you first understand what the main goal we are trying to do here is and then go ahead and implement.
</task>


<specifications>
Look at what we have completed so far in the @documentation/implementation-todo.md and the specifications of what should be completed in the @documentation/implementation-plan.md.md . You may also find the summary of what we are doing useful @documentation/summary_plan.md . Once you have completed this part and your review of it, update the implementation plan todo list with what was completed. if anything was completed that was not described in the implementation plan beforehand, update the implementation plan to include it as completed. note if there were sigificant challenges while implementing in the implementation plan.
</specifications>


<step-to-complete>


## Phase 4: Comprehensive Evaluation

### 4.1 Benchmark Dataset Preparation
- [ ] **Create evaluation datasets**:
  - [ ] Curate novel backbone designs (hallucinated structures)
  - [ ] Prepare multi-constraint problems (binding + stability)
  - [ ] Include length/complexity extrapolation test cases
  - [ ] Add challenging design targets from literature
  - [ ] Create ground truth labels for validation

- [ ] **Baseline comparisons**:
  - [ ] Implement ProteinMPNN baseline evaluation
  - [ ] Add RFdiffusion comparison (if available)
  - [ ] Include Rosetta design baseline
  - [ ] Create unified evaluation framework for fair comparison


</step-to-complete>

Small note: Tests you write will be run on macos, but the actual implementation will be run on our GPU cluster; don't worry about MPS issues running on macos


---


/commit-review

<task>
Check the changes made, which complete the following step of our implementation plan:

<step>

## Phase 4: Comprehensive Evaluation

### 4.1 Benchmark Dataset Preparation
- [ ] **Create evaluation datasets**:
  - [ ] Curate novel backbone designs (hallucinated structures)
  - [ ] Prepare multi-constraint problems (binding + stability)
  - [ ] Include length/complexity extrapolation test cases
  - [ ] Add challenging design targets from literature
  - [ ] Create ground truth labels for validation

- [ ] **Baseline comparisons**:
  - [ ] Implement ProteinMPNN baseline evaluation
  - [ ] Add RFdiffusion comparison (if available)
  - [ ] Include Rosetta design baseline
  - [ ] Create unified evaluation framework for fair comparison

</step>

@documentation/implementation-todo.md
@documentation/implementation-plan.md
</task>