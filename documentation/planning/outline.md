## Phase 0 – Project skeleton

**Step 0.1 – Create a hybrid repo**

* New repo: `proteinmpnn_ired_hybrid/`
* Submodules or vendor code:

  * `third_party/proteinmpnn/` → clone the official ProteinMPNN repo.
  * `third_party/ired/` → clone the IRED repo.
* Create a top-level `hybrid/` package where your new code lives:

  * `hybrid/models/`
  * `hybrid/train/`
  * `hybrid/inference/`
  * `hybrid/data/`

*No special copying here; just basic project setup.*

---

## Phase 1 – Reuse ProteinMPNN as a backbone encoder

**Goal:** Get a callable `encode_backbone(B)` that outputs per-residue embeddings.

**Step 1.1 – Import ProteinMPNN model & config**

* From **ProteinMPNN** you can largely **reuse**:

  * The model definition (GNN + decoder) in their main `protein_mpnn.py`.
  * The config / hyperparameters used in the Science paper checkpoints.
* Write a thin wrapper class in `hybrid/models/mpnn_encoder.py`:

  ```python
  class ProteinMPNNBackboneEncoder(nn.Module):
      def __init__(self, pretrained_ckpt_path):
          # load ProteinMPNN model
          # keep only encoder modules
      def forward(self, backbone_batch):
          # returns per-residue embeddings: [B, L, d]
  ```

**What’s mostly copy-paste:**

* Graph construction (distance cutoffs, edge features) from ProteinMPNN’s `StructureDataset` and model.
* Encoder forward pass up until the point where they normally output logits.

---

## Phase 2 – Define the sequence representation

**Goal:** Simple, differentiable sequence representation for v1.

**Step 2.1 – Use softmax + straight-through**

* Representation per residue:

  * A logit vector `z[i] ∈ R^{20}`.
  * Softmax: `p[i] = softmax(z[i])`.
  * Discrete AA: `aa[i] = argmax(p[i])` but use straight-through in backprop.
* For **initialization**:

  * Reuse ProteinMPNN’s decoder to generate logits for each residue given the backbone.
  * This gives you a sane starting `z_0`.

**What’s reused from ProteinMPNN:**

* The **decoder** that outputs logits for each AA.
* Their masking / batching utilities for sequences if you want.

Implement this in `hybrid/models/sequence_repr.py`.

---

## Phase 3 – Energy head (v1 EBM)

**Goal:** A single scalar energy given backbone embeddings + sequence logits.

**Step 3.1 – Define a simple energy network**

In `hybrid/models/energy_head.py`:

* Input:

  * Backbone features: `[B, L, d_b]` from MPNN encoder.
  * Sequence softmax probs: `[B, L, 20]` (or logits).
* Combine:

  * Concatenate per-residue: `[B, L, d_b + 20]`.
  * Pass through a small per-residue MLP → hidden dim `h`.
  * Pool over residues (mean or sum).
  * Final linear → scalar energy `E(B, s)`.

Simple pseudocode:

```python
class EnergyHead(nn.Module):
    def __init__(self, d_backbone, hidden_dim=512):
        ...
    def forward(self, backbone_feats, seq_probs):
        x = torch.cat([backbone_feats, seq_probs], dim=-1)
        h = self.res_mlp(x)           # [B, L, h]
        h_pool = h.mean(dim=1)        # [B, h]
        E = self.out_linear(h_pool)   # [B, 1]
        return E.squeeze(-1)
```

**What’s reused from IRED:**

* Conceptually, the **API** of an energy function: `E(x)` returning a scalar.
* But code-wise, this head is mostly your own; you don’t need IRED internals yet.

---

## Phase 4 – Minimal dataset & labeling (positives/negatives)

**Goal:** Tiny but workable dataset to shape the energy.

**Step 4.1 – Backbone extraction**

* Reuse ProteinMPNN’s **StructureDataset** / PDB parsing:

  * Get backbones (coordinates, masks, etc.) + native sequences.

**Step 4.2 – Positive / negative construction (MVP, keep it dumb)**

* Positives:

  * Native PDB sequence for that backbone.
  * Optionally 1–2 MPNN-designed sequences per backbone (no AF filtering yet).
* Negatives:

  * For each backbone, generate:

    * 1 random sequence.
    * 1–2 mutated versions of the positive (e.g., 5–10 random mutations).

This can all live in `hybrid/data/dataset.py`, and reuse:

* ProteinMPNN’s data loading + batching + graph building.

No AF2/Rosetta in v1; you’re just learning “native > random” as a sanity check.

---

## Phase 5 – Training loop (contrastive EBM, single landscape)

**Goal:** Train E so positives have lower energy than negatives.

**Step 5.1 – Implement a simple contrastive loss**

In `hybrid/train/train_energy_v1.py`:

For each batch, you’ll have:

* Backbone `B`.
* Positive sequence logits `z_pos`.
* Negative sequence logits `z_neg`.

Compute:

```python
E_pos = E(B, softmax(z_pos))
E_neg = E(B, softmax(z_neg))
loss = F.softplus(E_pos - E_neg).mean()  # margin-free logistic ranking
```

**What you can copy from IRED:**

* **Optimizer / scheduler patterns**:

  * Adam with specific LR schedule.
* **Training script skeleton**:

  * Argument parsing.
  * Logging, checkpointing.
  * Device placement and mixed precision if they have it.

You don’t need IRED’s full multi-landscape or noise schedule yet; just repurpose their training harness style.

---

## Phase 6 – Iterative inference loop (IRED-style, but simple)

**Goal:** Given backbone B, iteratively refine a sequence using energy gradients.

**Step 6.1 – Implement gradient descent on z**

In `hybrid/inference/optimize_sequence_v1.py`:

1. Initialize:

   * `z_0` from ProteinMPNN decoder logits.

2. For `t = 1..T` (e.g., 50 steps):

   * Compute `E_t = E(B, softmax(z_t))`.
   * `grad = ∂E_t/∂z_t` via autograd.
   * Update: `z_{t+1} = z_t - η * grad`.
   * Optionally:

     * Clamp logits magnitude.
     * Add a bit of noise for exploration early on.

3. At the end, decode:

   * `aa_seq = argmax(softmax(z_T), dim=-1)`.

**What you can reuse from IRED:**

* Their **update loop template**:

  * They already implement repeated gradient steps with step size schedules.
  * You can copy their pattern (e.g., schedule for η, early stopping if energy stops improving).
* Any utilities for:

  * Saving optimization trajectories.
  * Logging energy vs iteration.

For v1, stick to a **single energy function** (no annealing schedule yet) and a fixed T.

---

## Phase 7 – Basic evaluation harness

**Goal:** Quick sanity checks that the system isn’t degenerate.

In `hybrid/eval/eval_v1.py`:

* For a hold-out set of backbones:

  * Generate:

    * Native sequence.
    * MPNN one-shot design.
    * Hybrid EBM-optimized sequence (start from MPNN).
  * Compare:

    * Energy E(B, s_native), E(B, s_mpnn), E(B, s_hybrid).
    * ProteinMPNN log-likelihood of each sequence (to see if hybrid moves off the natural manifold).
* Print some summaries:

  * How often is `E_native < E_random`?
  * How often is `E_hybrid < E_mpnn`?

No AF2/Rosetta required for v1; this is a purely internal consistency check.

---

## Phase 8 – Wiring pieces together

**Step 8.1 – Configs and CLI**

* Add simple YAML or argparse configs:

  * `train_energy_v1.py --data_root ... --mpnn_ckpt ...`
  * `optimize_sequence_v1.py --backbone_pdb ... --mpnn_ckpt ... --energy_ckpt ...`

**Step 8.2 – Document what’s reused**

Add a short `README_v1.md`:

* From **ProteinMPNN**, you directly reuse:

  * Backbone parsing & graph building.
  * Encoder architecture and weights.
  * Decoder to produce initial logits / sequences.
* From **IRED**, you directly reuse:

  * Training loop structure (optimizer, logging, checkpointing).
  * Gradient-based iterative optimization pattern (update schedule).
  * General “energy_function(x) → scalar” abstraction.
