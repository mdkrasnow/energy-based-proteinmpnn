## Command Usage

/build-with-review 

<task>
Based on the implementation plan and the summary of the proposal, I want you to carry out the following step of the implementation plan. Make sure you first understand what the main goal we are trying to do here is and then go ahead and implement.
</task>


<specifications>
Look at what we have completed so far in the @documentation/implementation-todo.md and the specifications of what should be completed in the @documentation/implementation-plan.md.md . You may also find the summary of what we are doing useful @documentation/summary_plan.md . Once you have completed this part and your review of it, update the implementation plan todo list with what was completed. if anything was completed that was not described in the implementation plan beforehand, update the implementation plan to include it as completed. note if there were sigificant challenges while implementing in the implementation plan.
</specifications>


<step-to-complete>

### 2.2 Loss Functions & Training
- [ ] **Implement `hybrid/training/losses.py`**:
  - [ ] Create `ContrastiveLoss` with ranking and temperature terms
  - [ ] Add margin-based loss for energy ranking (E_pos + margin < E_neg)
  - [ ] Implement temperature-scaled contrastive loss
  - [ ] Add regularization terms (entropy, smoothness)
  - [ ] Include loss weighting for different negative types

- [ ] **Implement training loop `hybrid/training/train_energy.py`**:
  - [ ] Create training script with proper argument parsing
  - [ ] Add model initialization and checkpoint loading/saving
  - [ ] Implement training loop with validation monitoring
  - [ ] Add logging for loss curves and energy distributions
  - [ ] Include early stopping and learning rate scheduling

- [ ] **Training validation**:
  - [ ] Test training loop on small dataset subset
  - [ ] Verify loss decreases and energy rankings improve
  - [ ] Monitor for training instabilities or divergence
  - [ ] Validate checkpoint saving/loading functionality


</step-to-complete>

Small note: Tests you write will be run on macos, but the actual implementation will be run on our GPU cluster; don't worry about MPS issues running on macos


---



<task>
Check the changes made, which complete the following step of our implementation plan:

<step>

### 2.2 Loss Functions & Training
- [ ] **Implement `hybrid/training/losses.py`**:
  - [ ] Create `ContrastiveLoss` with ranking and temperature terms
  - [ ] Add margin-based loss for energy ranking (E_pos + margin < E_neg)
  - [ ] Implement temperature-scaled contrastive loss
  - [ ] Add regularization terms (entropy, smoothness)
  - [ ] Include loss weighting for different negative types

- [ ] **Implement training loop `hybrid/training/train_energy.py`**:
  - [ ] Create training script with proper argument parsing
  - [ ] Add model initialization and checkpoint loading/saving
  - [ ] Implement training loop with validation monitoring
  - [ ] Add logging for loss curves and energy distributions
  - [ ] Include early stopping and learning rate scheduling

- [ ] **Training validation**:
  - [ ] Test training loop on small dataset subset
  - [ ] Verify loss decreases and energy rankings improve
  - [ ] Monitor for training instabilities or divergence
  - [ ] Validate checkpoint saving/loading functionality


</step>

@documentation/implementation-todo.md
@documentation/implementation-plan.md
</task>