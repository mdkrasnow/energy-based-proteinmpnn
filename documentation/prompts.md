## Command Usage

/build-with-review 

<task>
Based on the implementation plan and the summary of the proposal, I want you to carry out the following step of the implementation plan. Make sure you first understand what the main goal we are trying to do here is and then go ahead and implement.
</task>


<specifications>
Look at what we have completed so far in the @documentation/implementation-todo.md and the specifications of what should be completed in the @documentation/implementation-plan.md.md . You may also find the summary of what we are doing useful @documentation/summary_plan.md . Once you have completed this part and your review of it, update the implementation plan todo list with what was completed. if anything was completed that was not described in the implementation plan beforehand, update the implementation plan to include it as completed. note if there were sigificant challenges while implementing in the implementation plan.
</specifications>


<step-to-complete>


## Phase 3: Iterative Optimization (Weeks 9-12)

### 3.1 IRED-Style Optimization Engine
- [ ] **Implement `hybrid/inference/ired_optimizer.py`**:
  - [ ] Create `IREDSequenceOptimizer` class with multi-landscape support
  - [ ] Implement `optimize_sequence` with annealed energy landscapes
  - [ ] Add gradient-based optimization loop with Adam optimizer
  - [ ] Include convergence monitoring and early stopping
  - [ ] Add noise injection for exploration in early landscapes

- [ ] **Adaptive computation features**:
  - [ ] Implement `adaptive_optimization` with difficulty assessment
  - [ ] Add automatic step allocation based on convergence quality
  - [ ] Include multiple restart strategies for failed optimizations
  - [ ] Add trajectory logging and analysis capabilities

- [ ] **Test optimization engine**:
  - [ ] Verify optimization converges to discrete sequences
  - [ ] Test adaptive step allocation on different difficulty levels
  - [ ] Validate gradient flow and numerical stability
  - [ ] Check memory usage and computational efficiency


</step-to-complete>

Small note: Tests you write will be run on macos, but the actual implementation will be run on our GPU cluster; don't worry about MPS issues running on macos


---


/commit-review

<task>
Check the changes made, which complete the following step of our implementation plan:

<step>

### 3.1 IRED-Style Optimization Engine
- [ ] **Implement `hybrid/inference/ired_optimizer.py`**:
  - [ ] Create `IREDSequenceOptimizer` class with multi-landscape support
  - [ ] Implement `optimize_sequence` with annealed energy landscapes
  - [ ] Add gradient-based optimization loop with Adam optimizer
  - [ ] Include convergence monitoring and early stopping
  - [ ] Add noise injection for exploration in early landscapes

- [ ] **Adaptive computation features**:
  - [ ] Implement `adaptive_optimization` with difficulty assessment
  - [ ] Add automatic step allocation based on convergence quality
  - [ ] Include multiple restart strategies for failed optimizations
  - [ ] Add trajectory logging and analysis capabilities

- [ ] **Test optimization engine**:
  - [ ] Verify optimization converges to discrete sequences
  - [ ] Test adaptive step allocation on different difficulty levels
  - [ ] Validate gradient flow and numerical stability
  - [ ] Check memory usage and computational efficiency


</step>

@documentation/implementation-todo.md
@documentation/implementation-plan.md
</task>