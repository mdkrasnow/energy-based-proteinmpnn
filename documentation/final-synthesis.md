# Final Synthesis: Energy-Based ProteinMPNN Implementation Plan

## Executive Summary

This implementation plan synthesizes research insights to create a hybrid Energy-Based Model (EBM) that combines ProteinMPNN's structural understanding with IRED's iterative optimization framework. The approach aims to achieve superior protein design capabilities through learned energy landscapes and adaptive computation, enabling designs that are more stable and robust to distribution shifts than current one-shot methods.

**Core Innovation**: Replace ProteinMPNN's autoregressive sequence prediction with an energy-based optimization that iteratively refines sequences using gradient-based updates on learned stability landscapes.

**Key Advantages**:
- Adaptive computation allocation for harder design problems
- Direct optimization of stability metrics rather than likelihood approximation
- Better generalization to out-of-distribution backbones and novel constraints
- Unified framework for multi-objective design (stability + binding + solubility)

## Architecture Overview

The hybrid system consists of four main components:

1. **Backbone Encoder**: ProteinMPNN's pre-trained structure encoder (reused/fine-tuned)
2. **Sequence Representation**: Continuous relaxation with Gumbel-Softmax + straight-through estimator
3. **Energy Head**: Neural network mapping (backbone features, sequence) → scalar energy
4. **Iterative Optimizer**: IRED-style gradient descent with annealed energy landscapes

## Detailed Technical Specifications

### Phase 1: Backbone Encoder Integration

**Component**: `hybrid/models/mpnn_encoder.py`

```python
class ProteinMPNNBackboneEncoder(nn.Module):
    def __init__(self, pretrained_ckpt_path, freeze_layers=True):
        super().__init__()
        # Load pre-trained ProteinMPNN model
        self.mpnn_model = load_pretrained_mpnn(pretrained_ckpt_path)
        
        # Extract encoder layers (graph construction + message passing)
        self.graph_builder = self.mpnn_model.graph_builder
        self.encoder_layers = self.mpnn_model.encoder_layers
        
        if freeze_layers:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, batch):
        # Input: backbone coordinates, masks, chain info
        # Output: per-residue embeddings [B, L, d_model]
        edge_idx, edge_features = self.graph_builder(batch)
        node_features = self.encoder_layers(batch['X'], edge_idx, edge_features)
        return node_features
```

**Key Decisions**:
- **Frozen vs Fine-tuned**: Start frozen to preserve structural knowledge, optionally fine-tune higher layers for stability-specific features
- **Feature Extraction**: Extract embeddings after final encoder layer but before decoder
- **Dimensions**: Maintain 128-dim hidden features to match original architecture

### Phase 2: Sequence Representation

**Component**: `hybrid/models/sequence_repr.py`

```python
class ContinuousSequenceRepr(nn.Module):
    def __init__(self, vocab_size=20, temperature_schedule=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.temperature_schedule = temperature_schedule or [1.0, 0.1]  # anneal over landscapes
        
    def forward(self, logits, landscape_idx=0, training=True):
        # logits: [B, L, 20] raw sequence logits
        # Use Gumbel-Softmax with annealing temperature
        temp = self.get_temperature(landscape_idx)
        
        if training:
            # Gumbel-Softmax for differentiable sampling
            soft_seq = F.gumbel_softmax(logits, tau=temp, hard=False)
        else:
            # Straight-through estimator for inference
            soft_seq = F.softmax(logits / temp, dim=-1)
            hard_seq = F.one_hot(logits.argmax(dim=-1), self.vocab_size).float()
            soft_seq = hard_seq + (soft_seq - soft_seq.detach())  # straight-through
            
        return soft_seq
    
    def get_temperature(self, landscape_idx):
        # Anneal temperature across IRED landscapes
        progress = landscape_idx / (len(self.temperature_schedule) - 1)
        return self.temperature_schedule[0] * (1 - progress) + self.temperature_schedule[1] * progress
```

**Key Decisions**:
- **Gumbel-Softmax**: Enables differentiable discrete sampling with temperature annealing
- **Straight-through**: Maintains discrete sequences during inference while allowing gradients
- **Annealing Schedule**: Temperature decreases across IRED landscapes (1.0 → 0.1)
- **Initialization**: Use ProteinMPNN decoder outputs as initial logits

### Phase 3: Energy Head Architecture

**Component**: `hybrid/models/energy_head.py`

```python
class EnergyHead(nn.Module):
    def __init__(self, backbone_dim=128, seq_dim=20, hidden_dim=512, num_layers=3):
        super().__init__()
        
        # Per-residue feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(backbone_dim + seq_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Per-residue processing layers
        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # Global pooling and energy prediction
        self.pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.energy_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, backbone_features, sequence_probs, mask):
        # backbone_features: [B, L, backbone_dim]
        # sequence_probs: [B, L, 20]  
        # mask: [B, L] sequence mask
        
        # Fuse per-residue features
        x = torch.cat([backbone_features, sequence_probs], dim=-1)
        x = self.feature_fusion(x)
        
        # Per-residue processing
        for layer in self.res_layers:
            x = x + layer(x)  # residual connections
            
        # Masked global pooling
        x = self.pooling(x)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        
        # Energy prediction
        energy = self.energy_head(pooled).squeeze(-1)
        return energy
```

**Key Decisions**:
- **Multi-scale Processing**: Combine local per-residue and global pooled features
- **Residual Connections**: Prevent gradient vanishing in deeper networks
- **Masked Pooling**: Handle variable sequence lengths appropriately
- **Single Energy**: Unified scalar representing overall sequence stability

### Phase 4: Training Data Pipeline

**Component**: `hybrid/data/stability_dataset.py`

```python
class StabilityDataset(Dataset):
    def __init__(self, data_root, split='train', augment_negatives=True):
        super().__init__()
        self.structure_dataset = StructureDatasetPDB(data_root)
        self.augment_negatives = augment_negatives
        
        # Load stability annotations
        self.positive_pairs = self.load_positive_pairs()  # (pdb_id, sequence, label=1)
        self.negative_pairs = self.generate_negative_pairs()  # (pdb_id, sequence, label=0)
        
    def load_positive_pairs(self):
        positives = []
        
        # Native PDB sequences
        for pdb_id in self.structure_dataset.pdb_ids:
            backbone, native_seq = self.structure_dataset[pdb_id]
            positives.append((pdb_id, native_seq, 'native'))
            
        # High-confidence designed sequences (if available)
        designed_seqs = self.load_validated_designs()  # AF2/Rosetta filtered
        positives.extend(designed_seqs)
        
        # Stability-enhanced mutants (from literature/databases)
        stable_mutants = self.load_stable_mutants()
        positives.extend(stable_mutants)
        
        return positives
        
    def generate_negative_pairs(self):
        negatives = []
        
        for pdb_id, pos_seq, _ in self.positive_pairs:
            backbone = self.structure_dataset.get_backbone(pdb_id)
            
            # Random sequences with realistic AA composition
            random_seq = self.generate_random_sequence(len(pos_seq))
            negatives.append((pdb_id, random_seq, 'random'))
            
            # Destabilizing mutations of positive sequences
            if self.augment_negatives:
                mutated_seqs = self.generate_destabilizing_mutations(pos_seq, backbone)
                negatives.extend([(pdb_id, seq, 'destabilizing') for seq in mutated_seqs])
                
            # Failed designs (low AF2 confidence)
            failed_designs = self.generate_failed_designs(backbone)
            negatives.extend([(pdb_id, seq, 'failed_design') for seq in failed_designs])
            
        return negatives
```

**Training Objectives**:

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, energy_pos, energy_neg):
        # Ranking loss: E_pos + margin < E_neg
        ranking_loss = F.softplus(energy_pos - energy_neg + self.margin).mean()
        
        # Contrastive loss for better separation
        pos_term = torch.exp(-energy_pos / self.temperature)
        neg_term = torch.exp(-energy_neg / self.temperature)
        contrastive_loss = -torch.log(pos_term / (pos_term + neg_term)).mean()
        
        return ranking_loss + 0.1 * contrastive_loss
```

**Key Decisions**:
- **Multi-source Positives**: Native, designed, and experimentally-validated sequences
- **Diverse Negatives**: Random, mutational, and design-failure examples
- **Hard Negative Mining**: Generate challenging negatives that look plausible but are unstable
- **Balanced Training**: Equal numbers of positive/negative pairs per backbone

### Phase 5: Iterative Optimization Engine

**Component**: `hybrid/inference/ired_optimizer.py`

```python
class IREDSequenceOptimizer:
    def __init__(self, energy_models, sequence_repr, num_landscapes=5):
        self.energy_models = energy_models  # List of E_1, E_2, ..., E_T
        self.sequence_repr = sequence_repr
        self.num_landscapes = num_landscapes
        
    def optimize_sequence(self, backbone_features, initial_logits, max_steps=50):
        """
        Iteratively optimize sequence using annealed energy landscapes
        """
        current_logits = initial_logits.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([current_logits], lr=0.01)
        
        trajectory = []
        
        for landscape_idx in range(self.num_landscapes):
            energy_model = self.energy_models[landscape_idx]
            steps_per_landscape = max_steps // self.num_landscapes
            
            for step in range(steps_per_landscape):
                optimizer.zero_grad()
                
                # Convert logits to soft sequence representation
                soft_seq = self.sequence_repr(current_logits, landscape_idx)
                
                # Compute energy
                energy = energy_model(backbone_features, soft_seq)
                
                # Gradient step (minimize energy)
                energy.backward()
                optimizer.step()
                
                # Optional: add noise for exploration in early landscapes
                if landscape_idx < self.num_landscapes // 2:
                    noise = torch.randn_like(current_logits) * 0.01
                    current_logits.data += noise
                
                # Log trajectory
                trajectory.append({
                    'landscape': landscape_idx,
                    'step': step,
                    'energy': energy.item(),
                    'sequence': soft_seq.argmax(dim=-1)
                })
                
                # Early stopping if converged
                if len(trajectory) > 5:
                    recent_energies = [t['energy'] for t in trajectory[-5:]]
                    if max(recent_energies) - min(recent_energies) < 1e-4:
                        break
        
        # Final sequence
        final_soft_seq = self.sequence_repr(current_logits, self.num_landscapes-1, training=False)
        final_sequence = final_soft_seq.argmax(dim=-1)
        
        return final_sequence, trajectory
        
    def adaptive_optimization(self, backbone_features, initial_logits, 
                            difficulty_threshold=0.1, max_total_steps=200):
        """
        Adaptively allocate more steps for harder problems
        """
        # Initial attempt with standard steps
        sequence, trajectory = self.optimize_sequence(
            backbone_features, initial_logits, max_steps=50
        )
        
        # Assess convergence quality
        final_energy = trajectory[-1]['energy']
        energy_variance = np.var([t['energy'] for t in trajectory[-10:]])
        
        # If not well-converged, allocate more steps
        if energy_variance > difficulty_threshold and len(trajectory) < max_total_steps:
            extended_steps = min(100, max_total_steps - len(trajectory))
            
            # Continue optimization with tighter convergence
            extended_seq, extended_traj = self.optimize_sequence(
                backbone_features, current_logits, max_steps=extended_steps
            )
            
            trajectory.extend(extended_traj)
            sequence = extended_seq
            
        return sequence, trajectory
```

**Key Decisions**:
- **Annealed Landscapes**: Train multiple energy models E_1, ..., E_T with increasing sharpness
- **Adaptive Steps**: Allocate more computation to harder design problems
- **Gradient-based**: Use standard optimizers (Adam) for smooth optimization
- **Early Stopping**: Monitor convergence to avoid unnecessary computation

## Implementation Phases and Timeline

### Phase 1: Foundation (Weeks 1-4)
**Milestone**: Working backbone encoder and basic sequence representation

**Tasks**:
1. Set up hybrid repository structure
2. Extract and wrap ProteinMPNN encoder
3. Implement continuous sequence representation
4. Basic training loop with simple energy model

**Success Criteria**:
- ProteinMPNN encoder produces embeddings for arbitrary backbones
- Sequence representation converts between logits and soft/hard sequences
- Basic energy model can be trained to distinguish native vs random sequences

### Phase 2: Energy Model Training (Weeks 5-8)
**Milestone**: Trained energy model that correlates with stability

**Tasks**:
1. Build comprehensive training dataset (positive/negative pairs)
2. Implement and train energy head architecture  
3. Add hard negative mining and data augmentation
4. Validate energy model on hold-out structures

**Success Criteria**:
- Energy model assigns lower energy to native sequences than random
- Model generalizes to unseen backbone topologies
- Energy scores correlate with Rosetta/AlphaFold stability metrics

### Phase 3: Iterative Optimization (Weeks 9-12)
**Milestone**: Full IRED-style optimization pipeline

**Tasks**:
1. Train sequence of annealed energy landscapes
2. Implement iterative optimization engine
3. Add adaptive step allocation based on problem difficulty
4. Integrate all components into unified design pipeline

**Success Criteria**:
- Iterative optimization converges to valid sequences
- Adaptive computation improves success rate on hard problems
- End-to-end pipeline from backbone → optimized sequence

### Phase 4: Comprehensive Evaluation (Weeks 13-16)
**Milestone**: Validated performance improvements over baselines

**Tasks**:
1. Systematic evaluation on benchmark tasks:
   - Novel backbone designs
   - Multi-constraint problems (binding + stability)
   - Length/complexity extrapolation
2. Comparison against ProteinMPNN, RFdiffusion, Rosetta
3. In silico validation with AlphaFold, Rosetta scoring

**Success Criteria**:
- Higher success rates on out-of-distribution designs
- Better stability scores for generated sequences
- Successful design of challenging multi-objective problems

### Phase 5: Optimization and Production (Weeks 17-20)
**Milestone**: Production-ready implementation

**Tasks**:
1. Performance optimization and memory efficiency
2. Comprehensive documentation and examples
3. Integration with existing design workflows
4. Preparation for experimental validation

**Success Criteria**:
- Scalable to large proteins (>500 residues)
- User-friendly API and CLI tools
- Ready for experimental collaboration

## Risk Mitigation Strategies

### Technical Risks

**Risk**: Sequence representation doesn't optimize stably
- **Mitigation**: Implement multiple representation schemes (Gumbel-Softmax, straight-through, discrete mutations)
- **Fallback**: Use discrete mutation proposals guided by continuous gradients

**Risk**: Energy model doesn't capture true stability
- **Mitigation**: Extensive validation against experimental data and physics-based models
- **Fallback**: Incorporate explicit physics terms (Rosetta energy, clash detection)

**Risk**: Iterative optimization gets stuck in local minima
- **Mitigation**: Multiple random restarts, noise injection, curriculum learning
- **Fallback**: Hybrid approach combining iterative refinement with MPNN initialization

### Data/Evaluation Risks

**Risk**: Limited experimental validation data
- **Mitigation**: Focus on computational metrics initially, collaborate for experimental validation
- **Fallback**: Use existing design success databases and literature validation

**Risk**: Evaluation metrics don't reflect real performance
- **Mitigation**: Multi-metric evaluation including structure prediction, physics simulation
- **Fallback**: Conservative experimental testing on subset of designs

## Success Metrics

### Primary Metrics
1. **Design Success Rate**: Fraction of designs that fold correctly (AlphaFold pLDDT > 80)
2. **Stability Improvement**: Rosetta energy scores vs baselines
3. **OOD Generalization**: Performance on novel backbone topologies

### Secondary Metrics  
1. **Sequence Diversity**: Shannon entropy of generated sequences
2. **Computational Efficiency**: Energy evaluations needed for convergence
3. **Multi-objective Performance**: Success on constrained design problems

### Experimental Validation
1. **Expression Success**: Soluble expression in E. coli
2. **Structural Validation**: NMR/X-ray confirmation of designed folds
3. **Stability Measurements**: Thermal denaturation curves

## Resource Requirements

### Computational Resources
- **Training**: 4x A100 GPUs for 2-3 weeks (energy model training)
- **Evaluation**: 1x A100 GPU for benchmarking
- **Storage**: 500GB for datasets and model checkpoints

### Human Resources
- **Lead Developer**: Full-time implementation and optimization
- **Research Scientist**: Algorithm development and evaluation design
- **Experimental Collaborator**: Validation and testing (part-time)

### Infrastructure
- **MLOps**: Model versioning, experiment tracking (Weights & Biases)
- **Compute**: Cloud GPU cluster (AWS/GCP) or institutional HPC
- **Software**: PyTorch, AlphaFold, Rosetta, structure analysis tools

## Conclusion

This synthesis represents a practical, phased approach to implementing an energy-based ProteinMPNN that leverages the best insights from both structure-conditioned design and iterative optimization. The plan balances technical ambition with implementation feasibility, providing clear milestones and risk mitigation strategies.

The hybrid approach promises significant advantages over current methods:
- **Superior Stability**: Direct optimization of energy rather than likelihood approximation
- **Adaptive Computation**: More steps for harder problems, like IRED's reasoning
- **Better Generalization**: Less bias toward evolutionary patterns, more physics-driven
- **Multi-objective Capability**: Unified framework for complex design constraints

Success of this implementation would represent a significant advance in computational protein design, enabling more stable and robust designs for therapeutic, industrial, and research applications.

## Appendix: Key Technical Decisions Rationale

### Sequence Representation Choice
After analyzing the trade-offs in the research documentation, I selected **Gumbel-Softmax with straight-through estimation** as the optimal sequence representation. This choice provides:

1. **Differentiability**: Enables gradient-based optimization while maintaining discrete sequences
2. **Temperature Annealing**: Natural integration with IRED's landscape progression
3. **Stability**: Avoids the oscillation issues of pure discrete updates
4. **Flexibility**: Can adapt temperature schedule based on convergence behavior

### Energy Architecture Design
The energy head architecture synthesizes insights from both codebases:

1. **Per-residue Processing**: Captures local stability constraints (hydrophobicity, secondary structure)
2. **Global Pooling**: Models inter-residue cooperativity and overall fold stability
3. **Residual Connections**: Prevents gradient vanishing in deeper networks
4. **Multi-scale Features**: Combines ProteinMPNN's structural embeddings with sequence information

### Training Strategy
The contrastive training approach addresses key challenges identified in the research:

1. **Diverse Negatives**: Prevents overfitting to simple random vs native distinctions
2. **Hard Negative Mining**: Forces model to learn subtle stability differences
3. **Multi-source Positives**: Reduces bias toward natural sequences only
4. **Balanced Sampling**: Ensures stable training dynamics

This comprehensive plan represents the synthesis of all available research and provides a clear pathway to implementing a superior protein design system.