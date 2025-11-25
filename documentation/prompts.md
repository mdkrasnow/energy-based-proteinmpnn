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

### 3.2 Multi-Landscape Training
- [ ] **Implement landscape training `hybrid/training/train_landscapes.py`**:
  - [ ] Create script for training multiple energy models E_1, ..., E_T
  - [ ] Implement progressive noise/smoothness annealing£
  - [ ] Add curriculum learning for landscape difficulty
  - [ ] Include cross-landscape consistency losses
  - [ ] Add landscape-specific evaluation metrics

- [ ] **Landscape validation**:
  - [ ] Test that landscapes form proper annealing sequence
  - [ ] Verify optimization progresses smoothly across landscapes
  - [ ] Check that final landscape produces sharp energy minima
  - [ ] Validate computational requirements for multi-landscape training


</step-to-complete>

Small note: Tests you write will be run on macos, but the actual implementation will be run on our GPU cluster; don't worry about MPS issues running on macos


---


/commit-review

<task>
Check the changes made, which complete the following step of our implementation plan:

<step>

### 3.2 Multi-Landscape Training
- [ ] **Implement landscape training `hybrid/training/train_landscapes.py`**:
  - [ ] Create script for training multiple energy models E_1, ..., E_T
  - [ ] Implement progressive noise/smoothness annealing
  - [ ] Add curriculum learning for landscape difficulty
  - [ ] Include cross-landscape consistency losses
  - [ ] Add landscape-specific evaluation metrics

- [ ] **Landscape validation**:
  - [ ] Test that landscapes form proper annealing sequence
  - [ ] Verify optimization progresses smoothly across landscapes
  - [ ] Check that final landscape produces sharp energy minima
  - [ ] Validate computational requirements for multi-landscape training

</step>

@documentation/implementation-todo.md
@documentation/implementation-plan.md
</task>