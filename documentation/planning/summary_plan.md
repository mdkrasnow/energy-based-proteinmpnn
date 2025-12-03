# ProteinMPNN-IRED Hybrid: Implementation Plan Summary

## Overview

This project implements a novel hybrid approach that combines **ProteinMPNN's structural understanding** with **IRED's iterative reasoning** to create a superior protein design system. Instead of generating sequences in a single forward pass, we use energy-based optimization to iteratively refine designs, enabling more stable and robust protein sequences.

## The Problem We're Solving

**Current Limitation**: ProteinMPNN generates sequences autoregressively based on evolutionary likelihood, which works well but has limitations:
- Biased toward naturally-occurring sequence patterns
- Cannot adaptively allocate computation to harder design problems
- Difficult to incorporate multiple design objectives simultaneously
- Limited exploration beyond the training distribution

**Our Solution**: Replace one-shot prediction with iterative energy minimization that can "think longer" on challenging problems.

## Core Innovation

### Energy-Based Sequence Optimization
```
Traditional: Backbone → ProteinMPNN → Sequence (one shot)
Our Approach: Backbone → Energy Landscapes → Iterative Optimization → Optimized Sequence
```

**Key Advantages**:
- **Adaptive Computation**: More optimization steps for harder design challenges
- **Physics-Based**: Direct optimization of stability rather than likelihood approximation  
- **Multi-Objective**: Unified framework for stability + binding + other constraints
- **Better Generalization**: Less evolutionary bias, better performance on novel structures

## Architecture Components

### 1. Backbone Encoder (Reused from ProteinMPNN)
- **Purpose**: Extract structural features from protein backbones
- **Implementation**: Wrap existing ProteinMPNN encoder with frozen/fine-tunable modes
- **Output**: Per-residue embeddings capturing local and global structural context

### 2. Sequence Representation (New)
- **Purpose**: Enable gradient-based optimization over discrete amino acid sequences
- **Implementation**: Gumbel-Softmax with temperature annealing + straight-through estimation
- **Key Feature**: Differentiable representation that maintains discrete sequences

### 3. Energy Head (New)
- **Purpose**: Learn to score sequence stability given backbone structure
- **Implementation**: Neural network mapping (backbone features + sequence) → scalar energy
- **Training**: Contrastive learning (native sequences = low energy, random/mutated = high energy)

### 4. Iterative Optimizer (Adapted from IRED)
- **Purpose**: Refine sequences through gradient-based energy minimization
- **Implementation**: Multi-landscape optimization with adaptive step allocation
- **Key Feature**: Automatically allocates more computation to challenging design problems

## Implementation Strategy

### Phase 1: Foundation (Weeks 1-4)
**Goal**: Get basic components working together
- Wrap ProteinMPNN encoder for feature extraction
- Implement differentiable sequence representation
- Create simple energy model that distinguishes native from random sequences
- Establish training pipeline

### Phase 2: Energy Model Training (Weeks 5-8) 
**Goal**: Train energy function that correlates with protein stability
- Build comprehensive training dataset (positive/negative sequence pairs)
- Train energy model with contrastive learning
- Validate that energy rankings correlate with known stability measures
- Implement hard negative mining for robust training

### Phase 3: Iterative Optimization (Weeks 9-12)
**Goal**: Implement full IRED-style optimization pipeline
- Train sequence of annealed energy landscapes (E₁, E₂, ..., E_T)
- Implement iterative optimization with adaptive step allocation
- Integrate all components into end-to-end design pipeline
- Test convergence and sequence quality

### Phase 4: Comprehensive Evaluation (Weeks 13-16)
**Goal**: Validate performance improvements over existing methods
- Systematic evaluation on challenging design tasks:
  - Novel backbone structures (out-of-distribution)
  - Multi-objective problems (stability + binding)
  - Complex/large protein assemblies
- Compare against ProteinMPNN, RFdiffusion, Rosetta baselines
- In silico validation with AlphaFold and Rosetta scoring

### Phase 5: Production Optimization (Weeks 17-20)
**Goal**: Create production-ready system
- Performance optimization for large proteins (>500 residues)
- User-friendly API and command-line interface
- Comprehensive documentation and tutorials
- Preparation for experimental validation

## Success Metrics

### Primary Goals
1. **Higher Design Success Rate**: >90% of designs fold correctly (AlphaFold pLDDT > 80)
2. **Superior Stability**: Better Rosetta energy scores vs. baselines
3. **OOD Generalization**: Successful designs on completely novel backbone structures
4. **Adaptive Performance**: Better success on challenging multi-objective problems

### Technical Milestones
- Energy model correctly ranks native > random sequences (>90% accuracy)
- Iterative optimization converges to valid sequences within reasonable compute budget
- System scales to large proteins without memory/speed issues
- End-to-end pipeline produces publication-quality results

## Why This Approach Will Work

### Strong Theoretical Foundation
- **ProteinMPNN**: Proven structural encoder with excellent performance on protein design
- **IRED**: Demonstrated ability to solve harder problems through iterative reasoning
- **Energy-Based Models**: Well-established framework for optimization problems

### Maximum Component Reuse
- Leverages ProteinMPNN's pre-trained encoder (saves months of development)
- Adapts IRED's optimization framework (proven iterative reasoning approach)
- Minimal new code required (~500 lines vs. 2000+ from scratch)

### Practical Implementation Path
- Progressive complexity: start simple, add sophistication incrementally
- Clear validation checkpoints at each phase
- Fallback strategies for technical risks
- Realistic timeline with built-in contingencies

## Expected Impact

### Immediate Benefits
- More stable protein designs for therapeutic applications
- Better performance on challenging design targets (binding, symmetry, large assemblies)
- Unified framework for multi-objective optimization
- Reduced need for experimental screening through better computational designs

### Long-term Significance
- Foundation for next-generation protein design methods
- Integration with experimental workflows and high-throughput validation
- Enables design of previously impossible protein architectures
- Potential applications in enzyme design, drug development, biomaterials

## Risk Mitigation

### Technical Risks
- **Sequence representation instability**: Multiple backup approaches (Gumbel-Softmax, straight-through, discrete)
- **Energy model limitations**: Extensive validation against physics-based models
- **Optimization convergence**: Multiple restart strategies and fallback mechanisms

### Implementation Risks
- **Integration complexity**: Careful component interface design and extensive testing
- **Performance bottlenecks**: Early profiling and optimization planning
- **Validation challenges**: Multi-metric evaluation with computational and experimental validation

## Timeline Summary

```
Weeks 1-4:   Foundation components working
Weeks 5-8:   Trained energy model
Weeks 9-12:  Full optimization pipeline
Weeks 13-16: Comprehensive evaluation
Weeks 17-20: Production-ready system
```

**Total Duration**: 20 weeks (5 months)
**Key Milestone**: Week 12 - End-to-end protein design pipeline operational
**Final Deliverable**: Week 20 - Production system ready for experimental validation

## Getting Started

1. **Read the detailed plan**: [final-synthesis.md](final-synthesis.md)
2. **Follow the implementation steps**: [implementation-todo.md](implementation-todo.md)  
3. **Set up development environment**: Phase 1.1 in todo list
4. **Begin with ProteinMPNN integration**: Phase 1.2 - highest priority

This project represents a significant advance in computational protein design, combining proven components in a novel way to achieve superior performance on challenging design problems.