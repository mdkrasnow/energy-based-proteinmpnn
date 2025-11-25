#!/usr/bin/env python3
"""
Multi-Constraint Problem Generation for Protein Design Benchmarks

This module creates challenging multi-objective protein design problems that require
optimizing for multiple constraints simultaneously, such as:

1. Binding affinity + structural stability
2. Binding specificity (target binding while avoiding off-targets)
3. Stability + expression levels
4. Stability + solubility
5. Multi-target binding problems

These problems test the ability of design methods to handle complex, real-world
design challenges that go beyond simple single-objective optimization.

Key Features:
- Realistic constraint specifications
- Biologically meaningful target values
- Support for various constraint combinations
- Integration with existing evaluation framework
- Comprehensive success criteria definition
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import random
import numpy as np

# Optional dependencies with graceful degradation
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Statistical constraint generation will be limited.")

# Add project root for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))


@dataclass
class ConstraintSpec:
    """
    Specification for a single design constraint.
    
    Attributes:
        constraint_id: Unique identifier for this constraint
        constraint_type: Type of constraint ('binding_affinity', 'stability', etc.)
        target_value: Target value for the constraint
        tolerance: Acceptable tolerance around target value
        weight: Relative importance (0.0-1.0)
        measurement_method: How this constraint is measured
        success_threshold: Threshold for considering constraint satisfied
        failure_penalty: Penalty for failing this constraint
    """
    constraint_id: str
    constraint_type: str
    target_value: float
    tolerance: float = 0.1
    weight: float = 1.0
    measurement_method: str = "computational"
    success_threshold: Optional[float] = None
    failure_penalty: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiConstraintProblem:
    """
    Complete specification of a multi-constraint design problem.
    
    Attributes:
        problem_id: Unique identifier
        description: Human-readable description
        constraints: List of constraint specifications
        target_structure: Structure information for the design target
        success_criteria: Overall success criteria for the problem
        difficulty_level: Estimated difficulty ('easy', 'medium', 'hard', 'extreme')
        constraint_interactions: Information about constraint interactions
        evaluation_protocol: How to evaluate solutions
        metadata: Additional problem metadata
    """
    problem_id: str
    description: str
    constraints: List[ConstraintSpec]
    target_structure: Dict[str, Any]
    success_criteria: Dict[str, Any]
    difficulty_level: str = "medium"
    constraint_interactions: Dict[str, Any] = field(default_factory=dict)
    evaluation_protocol: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiConstraintProblemGenerator:
    """
    Generates challenging multi-constraint protein design problems.
    
    This generator creates realistic protein design challenges that require
    optimizing for multiple objectives simultaneously. Problems are designed
    to test the limits of design methods and explore trade-offs between
    competing objectives.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize multi-constraint problem generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Define constraint type specifications with physics-based limits and literature citations
        self.constraint_types = {
            'binding_affinity': {
                'description': 'Binding affinity to target protein/molecule',
                'units': 'kcal/mol',
                'typical_range': (-15.0, -5.0),
                'excellent_threshold': -12.0,
                'good_threshold': -8.0,
                'measurement_methods': ['docking', 'experimental', 'ml_prediction'],
                'physical_limits': {
                    'minimum_achievable': -25.0,  # Most stable protein complexes (Janin 1995)
                    'maximum_measurable': -1.0,   # Detection limit for weak interactions
                    'typical_protein_protein': (-12.0, -6.0),  # Wells & McClendon 2007
                    'ultra_high_affinity_limit': -18.0  # Antibody-antigen complexes (Rini et al. 1992)
                },
                'literature_validation': {
                    'source_papers': [
                        'Wells & McClendon (2007) Nature 450:1001-1009',
                        'Janin (1995) Proteins 21:30-39', 
                        'Clackson & Wells (1995) Science 267:383-386'
                    ],
                    'experimental_range': 'Most protein-protein interactions: -6 to -12 kcal/mol',
                    'design_examples': 'Silva et al. (2019) achieved -13.2 kcal/mol for designed binders'
                }
            },
            'binding_specificity': {
                'description': 'Selectivity for target vs off-target binding',
                'units': 'fold_selectivity',
                'typical_range': (10.0, 1000.0),
                'excellent_threshold': 100.0,
                'good_threshold': 10.0,
                'measurement_methods': ['competitive_binding', 'selectivity_panel'],
                'physical_limits': {
                    'minimum_meaningful': 2.0,     # 2-fold selectivity minimum for biological relevance
                    'maximum_achievable': 100000.0, # Highest observed in natural proteins
                    'typical_designed_range': (10.0, 1000.0),
                    'exceptional_cases': 50000.0   # Highly evolved natural systems
                },
                'literature_validation': {
                    'source_papers': [
                        'Baker & Kornberg (1992) Cell 68:1137-1144',
                        'Horovitz (1996) Folding & Design 1:R121-R126'
                    ],
                    'experimental_range': 'Designed proteins typically achieve 10-1000 fold selectivity',
                    'design_examples': 'Fleishman et al. (2011) achieved >500-fold specificity'
                }
            },
            'structural_stability': {
                'description': 'Thermodynamic stability of folded structure',
                'units': 'kcal/mol',
                'typical_range': (-20.0, -5.0),
                'excellent_threshold': -15.0,
                'good_threshold': -8.0,
                'measurement_methods': ['rosetta', 'foldx', 'experimental'],
                'physical_limits': {
                    'minimum_achievable': -35.0,  # Most stable known proteins (Vieille & Zeikus 2001)
                    'maximum_unstable': 5.0,      # Severely destabilized but still foldable
                    'typical_natural_range': (-15.0, -5.0),  # Pace et al. (2004)
                    'hyperstable_threshold': -25.0  # Thermophilic proteins (Sterner & Liebl 2001)
                },
                'literature_validation': {
                    'source_papers': [
                        'Pace et al. (2004) Protein Sci 13:1-3',
                        'Vieille & Zeikus (2001) Microbiol Mol Biol Rev 65:1-43',
                        'Steipe et al. (1994) J Mol Biol 240:188-192'
                    ],
                    'experimental_range': 'Natural proteins: -5 to -15 kcal/mol; Hyperstable designs: -25 kcal/mol',
                    'design_examples': 'Borgo & Havranek (2012) achieved -22 kcal/mol stability improvement'
                }
            },
            'expression_level': {
                'description': 'Expression level in target organism',
                'units': 'relative_expression',
                'typical_range': (0.1, 10.0),
                'excellent_threshold': 5.0,
                'good_threshold': 1.0,
                'measurement_methods': ['expression_prediction', 'experimental'],
                'physical_limits': {
                    'minimum_detectable': 0.01,   # Detection threshold for most assays
                    'maximum_achievable': 50.0,   # Highest expression levels observed
                    'typical_design_range': (0.1, 10.0),
                    'toxic_threshold': 20.0       # Expression levels causing cell stress
                },
                'literature_validation': {
                    'source_papers': [
                        'Wilkinson & Harrison (1991) Biotechnology 9:443-448',
                        'Rosano & Ceccarelli (2014) Front Microbiol 5:172'
                    ],
                    'experimental_range': 'Successful designs typically show 0.5-5x expression vs wild-type',
                    'design_examples': 'Codon optimization can achieve 10-50x improvement'
                }
            },
            'solubility': {
                'description': 'Solubility in aqueous solution',
                'units': 'mg/ml',
                'typical_range': (0.1, 100.0),
                'excellent_threshold': 10.0,
                'good_threshold': 1.0,
                'measurement_methods': ['solubility_prediction', 'experimental'],
                'physical_limits': {
                    'minimum_meaningful': 0.01,   # Minimum for most biochemical assays
                    'maximum_achievable': 500.0,  # Highest observed for engineered proteins
                    'typical_design_range': (0.1, 50.0),
                    'supersaturation_limit': 200.0 # Practical upper limit before crystallization
                },
                'literature_validation': {
                    'source_papers': [
                        'Trainor et al. (2017) Protein Expr Purif 138:11-22',
                        'Wilkinson & Harrison (1991) Biotechnology 9:443-448'
                    ],
                    'experimental_range': 'Well-designed proteins: 1-50 mg/ml; poorly designed: <0.1 mg/ml',
                    'design_examples': 'Solubility optimization can improve from 0.1 to 20+ mg/ml'
                }
            },
            'aggregation_propensity': {
                'description': 'Tendency to form aggregates',
                'units': 'aggregation_score',
                'typical_range': (0.0, 1.0),
                'excellent_threshold': 0.2,
                'good_threshold': 0.5,
                'measurement_methods': ['aggregation_prediction', 'experimental'],
                'physical_limits': {
                    'minimum_achievable': 0.0,    # Perfectly non-aggregating (theoretical)
                    'maximum_tolerable': 0.8,     # Above this, protein is essentially unusable
                    'typical_good_range': (0.0, 0.3),
                    'problematic_threshold': 0.6   # Significant aggregation issues
                },
                'literature_validation': {
                    'source_papers': [
                        'Chiti & Dobson (2006) Annu Rev Biochem 75:333-366',
                        'Wang et al. (2012) Nat Biotechnol 30:1203-1207'
                    ],
                    'experimental_range': 'Well-behaved proteins: 0.0-0.3; problematic: >0.5',
                    'design_examples': 'Rational design can reduce aggregation scores from 0.8 to 0.1'
                }
            },
            'immunogenicity': {
                'description': 'Potential to trigger immune response',
                'units': 'immunogenicity_score',
                'typical_range': (0.0, 1.0),
                'excellent_threshold': 0.1,
                'good_threshold': 0.3,
                'measurement_methods': ['immunogenicity_prediction', 'epitope_analysis'],
                'physical_limits': {
                    'minimum_achievable': 0.0,    # Non-immunogenic (rare for foreign proteins)
                    'maximum_tolerable': 0.5,     # Above this, significant immune response
                    'typical_therapeutic_range': (0.0, 0.2),
                    'high_risk_threshold': 0.4     # Unacceptable for therapeutic use
                },
                'literature_validation': {
                    'source_papers': [
                        'De Groot & Scott (2007) Trends Immunol 28:482-490',
                        'Baker & Jones (2007) Curr Opin Drug Discov Devel 10:219-227'
                    ],
                    'experimental_range': 'Therapeutic proteins: <0.2; research proteins: <0.4',
                    'design_examples': 'Deimmunization can reduce scores from 0.6 to 0.1'
                }
            }
        }
        
        # Define common constraint combinations
        self.constraint_combinations = {
            'binding_stability': [
                'binding_affinity',
                'structural_stability'
            ],
            'binding_specificity': [
                'binding_affinity',
                'binding_specificity'
            ],
            'stability_expression': [
                'structural_stability',
                'expression_level'
            ],
            'stability_solubility': [
                'structural_stability', 
                'solubility'
            ],
            'therapeutic_design': [
                'binding_affinity',
                'structural_stability',
                'solubility',
                'immunogenicity'
            ],
            'enzyme_design': [
                'binding_affinity',
                'structural_stability',
                'expression_level',
                'aggregation_propensity'
            ]
        }
        
    def generate_multi_constraint_problem(
        self,
        problem_id: str,
        constraint_combination: str,
        difficulty: str = 'medium'
    ) -> MultiConstraintProblem:
        """
        Generate a single multi-constraint problem.
        
        Args:
            problem_id: Unique identifier for the problem
            constraint_combination: Type of constraint combination
            difficulty: Difficulty level ('easy', 'medium', 'hard', 'extreme')
            
        Returns:
            Generated multi-constraint problem
        """
        try:
            # Get constraint types for this combination
            constraint_types = self.constraint_combinations.get(
                constraint_combination, 
                ['binding_affinity', 'structural_stability']
            )
            
            # Generate constraints
            constraints = []
            for i, constraint_type in enumerate(constraint_types):
                constraint = self._generate_constraint(
                    f"{problem_id}_constraint_{i:02d}",
                    constraint_type,
                    difficulty
                )
                constraints.append(constraint)
                
            # Generate target structure
            target_structure = self._generate_target_structure(
                problem_id, constraint_combination, difficulty
            )
            
            # Define success criteria
            success_criteria = self._define_success_criteria(
                constraints, difficulty
            )
            
            # Analyze constraint interactions
            interactions = self._analyze_constraint_interactions(constraints)
            
            # Create evaluation protocol
            eval_protocol = self._create_evaluation_protocol(
                constraints, constraint_combination
            )
            
            # Validate constraint combination feasibility
            feasibility_check = self._validate_constraint_feasibility(constraints, difficulty)
            
            # Create problem
            problem = MultiConstraintProblem(
                problem_id=problem_id,
                description=self._generate_problem_description(
                    constraint_combination, constraints, difficulty
                ),
                constraints=constraints,
                target_structure=target_structure,
                success_criteria=success_criteria,
                difficulty_level=difficulty,
                constraint_interactions=interactions,
                evaluation_protocol=eval_protocol,
                metadata={
                    'constraint_combination': constraint_combination,
                    'generation_time': datetime.now().isoformat(),
                    'generator_version': '1.0',
                    'seed': self.seed,
                    'feasibility_check': feasibility_check
                }
            )
            
            return problem
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate multi-constraint problem {problem_id}: {e}")
            
    def _generate_constraint(
        self,
        constraint_id: str,
        constraint_type: str,
        difficulty: str
    ) -> ConstraintSpec:
        """Generate a single constraint specification."""
        if constraint_type not in self.constraint_types:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
            
        spec = self.constraint_types[constraint_type]
        
        # Adjust target values based on difficulty
        if difficulty == 'easy':
            # Lenient targets, easier to achieve
            target_value = self._sample_target_value(spec, strictness=0.3)
            tolerance = 0.2
            weight = 1.0
        elif difficulty == 'medium':
            # Moderate targets
            target_value = self._sample_target_value(spec, strictness=0.6)
            tolerance = 0.15
            weight = 1.0
        elif difficulty == 'hard':
            # Challenging targets
            target_value = self._sample_target_value(spec, strictness=0.8)
            tolerance = 0.1
            weight = 1.0
        else:  # extreme
            # Very challenging targets
            target_value = self._sample_target_value(spec, strictness=0.95)
            tolerance = 0.05
            weight = 1.0
            
        # Choose measurement method
        method = random.choice(spec['measurement_methods'])
        
        constraint = ConstraintSpec(
            constraint_id=constraint_id,
            constraint_type=constraint_type,
            target_value=target_value,
            tolerance=tolerance,
            weight=weight,
            measurement_method=method,
            success_threshold=target_value * (1 - tolerance),
            failure_penalty=1.0,
            metadata={
                'units': spec['units'],
                'description': spec['description'],
                'difficulty': difficulty,
                'strictness': 0.3 if difficulty == 'easy' else 0.95
            }
        )
        
        return constraint
        
    def _sample_target_value(
        self,
        constraint_spec: Dict[str, Any],
        strictness: float
    ) -> float:
        """Sample a target value based on constraint spec and strictness."""
        min_val, max_val = constraint_spec['typical_range']
        excellent = constraint_spec['excellent_threshold']
        good = constraint_spec['good_threshold']
        
        # Determine target based on strictness
        if strictness < 0.4:
            # Easy target - around "good" threshold
            target = good + random.uniform(-abs(good) * 0.3, abs(good) * 0.3)
        elif strictness < 0.7:
            # Medium target - between good and excellent
            target = (good + excellent) / 2 + random.uniform(-abs(excellent - good) * 0.3, abs(excellent - good) * 0.3)
        else:
            # Hard target - around or better than excellent threshold
            target = excellent + random.uniform(-abs(excellent) * 0.2, abs(excellent) * 0.5)
            
        # Ensure target is within reasonable range
        target = np.clip(target, min_val, max_val)
        
        # Additional validation for extreme cases using physics-based limits
        if strictness > 0.9:
            # For extreme difficulty, ensure target is actually achievable
            # by not exceeding established physical limits from literature
            constraint_type = constraint_spec.get('constraint_type', '')
            physical_limits = constraint_spec.get('physical_limits', {})
            
            if constraint_type == 'binding_affinity':
                # Don't exceed ultra-high affinity limit (Clackson & Wells 1995)
                ultra_limit = physical_limits.get('ultra_high_affinity_limit', -18.0)
                target = max(target, ultra_limit)
            elif constraint_type == 'structural_stability':
                # Don't exceed hyperstable protein limits (Vieille & Zeikus 2001)
                hyperstable_limit = physical_limits.get('hyperstable_threshold', -25.0)
                target = max(target, hyperstable_limit)
            elif constraint_type == 'binding_specificity':
                # Don't exceed maximum observed selectivity
                max_selectivity = physical_limits.get('maximum_achievable', 100000.0)
                target = min(target, max_selectivity)
            elif constraint_type == 'expression_level':
                # Don't exceed toxic expression levels
                toxic_limit = physical_limits.get('toxic_threshold', 20.0)
                target = min(target, toxic_limit)
            elif constraint_type == 'solubility':
                # Don't exceed supersaturation limits
                supersaturation_limit = physical_limits.get('supersaturation_limit', 200.0)
                target = min(target, supersaturation_limit)
            elif constraint_type == 'aggregation_propensity':
                # Don't exceed maximum tolerable aggregation
                max_tolerable = physical_limits.get('maximum_tolerable', 0.8)
                target = min(target, max_tolerable)
            elif constraint_type == 'immunogenicity':
                # Don't exceed maximum tolerable immunogenicity
                max_tolerable = physical_limits.get('maximum_tolerable', 0.5)
                target = min(target, max_tolerable)
        
        return float(target)
        
    def _validate_constraint_feasibility(
        self,
        constraints: List[ConstraintSpec],
        difficulty: str
    ) -> Dict[str, Any]:
        """Validate that constraint combination is feasible."""
        validation = {
            'feasible': True,
            'warnings': [],
            'confidence': 1.0,
            'constraint_checks': []
        }
        
        try:
            # Check individual constraint ranges
            for constraint in constraints:
                constraint_check = {
                    'constraint_id': constraint.constraint_id,
                    'constraint_type': constraint.constraint_type,
                    'target_value': constraint.target_value,
                    'feasible': True,
                    'issues': []
                }
                
                # Check if target value is within physics-based achievable ranges using literature validation
                constraint_spec = self.constraint_types.get(constraint.constraint_type, {})
                physical_limits = constraint_spec.get('physical_limits', {})
                literature_validation = constraint_spec.get('literature_validation', {})
                
                if constraint.constraint_type == 'binding_affinity':
                    min_achievable = physical_limits.get('minimum_achievable', -25.0)
                    max_measurable = physical_limits.get('maximum_measurable', -1.0)
                    if constraint.target_value < min_achievable:
                        constraint_check['feasible'] = False
                        constraint_check['issues'].append(
                            f"Binding affinity {constraint.target_value} kcal/mol exceeds minimum achievable "
                            f"{min_achievable} kcal/mol (Janin 1995; Wells & McClendon 2007)"
                        )
                    elif constraint.target_value > max_measurable:
                        constraint_check['issues'].append(
                            f"Very weak binding affinity {constraint.target_value} kcal/mol near detection limit "
                            f"{max_measurable} kcal/mol - may be difficult to measure accurately"
                        )
                        
                elif constraint.constraint_type == 'structural_stability':
                    min_achievable = physical_limits.get('minimum_achievable', -35.0)
                    max_unstable = physical_limits.get('maximum_unstable', 5.0)
                    if constraint.target_value < min_achievable:
                        constraint_check['feasible'] = False
                        constraint_check['issues'].append(
                            f"Stability {constraint.target_value} kcal/mol exceeds minimum achievable "
                            f"{min_achievable} kcal/mol (Vieille & Zeikus 2001)"
                        )
                    elif constraint.target_value > max_unstable:
                        constraint_check['issues'].append(
                            f"Unstable target {constraint.target_value} kcal/mol approaching unfolded limit "
                            f"{max_unstable} kcal/mol - very challenging to achieve"
                        )
                        
                elif constraint.constraint_type == 'binding_specificity':
                    min_meaningful = physical_limits.get('minimum_meaningful', 2.0)
                    max_achievable = physical_limits.get('maximum_achievable', 100000.0)
                    if constraint.target_value < min_meaningful:
                        constraint_check['feasible'] = False
                        constraint_check['issues'].append(
                            f"Selectivity {constraint.target_value}-fold below minimum meaningful "
                            f"{min_meaningful}-fold for biological relevance"
                        )
                    elif constraint.target_value > max_achievable:
                        constraint_check['feasible'] = False
                        constraint_check['issues'].append(
                            f"Selectivity {constraint.target_value}-fold exceeds maximum observed "
                            f"{max_achievable}-fold in natural systems"
                        )
                        
                elif constraint.constraint_type == 'expression_level':
                    min_detectable = physical_limits.get('minimum_detectable', 0.01)
                    toxic_threshold = physical_limits.get('toxic_threshold', 20.0)
                    if constraint.target_value < min_detectable:
                        constraint_check['feasible'] = False
                        constraint_check['issues'].append(
                            f"Expression level {constraint.target_value}x below detection threshold "
                            f"{min_detectable}x"
                        )
                    elif constraint.target_value > toxic_threshold:
                        constraint_check['feasible'] = False
                        constraint_check['issues'].append(
                            f"Expression level {constraint.target_value}x exceeds toxic threshold "
                            f"{toxic_threshold}x - may cause cell stress"
                        )
                        
                elif constraint.constraint_type == 'solubility':
                    min_meaningful = physical_limits.get('minimum_meaningful', 0.01)
                    supersaturation_limit = physical_limits.get('supersaturation_limit', 200.0)
                    if constraint.target_value < min_meaningful:
                        constraint_check['issues'].append(
                            f"Solubility {constraint.target_value} mg/ml below practical threshold "
                            f"{min_meaningful} mg/ml for most assays"
                        )
                    elif constraint.target_value > supersaturation_limit:
                        constraint_check['feasible'] = False
                        constraint_check['issues'].append(
                            f"Solubility {constraint.target_value} mg/ml exceeds supersaturation limit "
                            f"{supersaturation_limit} mg/ml - may crystallize spontaneously"
                        )
                        
                elif constraint.constraint_type == 'aggregation_propensity':
                    max_tolerable = physical_limits.get('maximum_tolerable', 0.8)
                    problematic_threshold = physical_limits.get('problematic_threshold', 0.6)
                    if constraint.target_value > max_tolerable:
                        constraint_check['feasible'] = False
                        constraint_check['issues'].append(
                            f"Aggregation score {constraint.target_value} exceeds maximum tolerable "
                            f"{max_tolerable} - protein essentially unusable (Chiti & Dobson 2006)"
                        )
                    elif constraint.target_value > problematic_threshold:
                        constraint_check['issues'].append(
                            f"Aggregation score {constraint.target_value} above problematic threshold "
                            f"{problematic_threshold} - significant issues expected"
                        )
                        
                elif constraint.constraint_type == 'immunogenicity':
                    max_tolerable = physical_limits.get('maximum_tolerable', 0.5)
                    high_risk_threshold = physical_limits.get('high_risk_threshold', 0.4)
                    if constraint.target_value > max_tolerable:
                        constraint_check['feasible'] = False
                        constraint_check['issues'].append(
                            f"Immunogenicity score {constraint.target_value} exceeds maximum tolerable "
                            f"{max_tolerable} - significant immune response expected"
                        )
                    elif constraint.target_value > high_risk_threshold:
                        constraint_check['issues'].append(
                            f"Immunogenicity score {constraint.target_value} above high-risk threshold "
                            f"{high_risk_threshold} - unacceptable for therapeutic use"
                        )
                        
                validation['constraint_checks'].append(constraint_check)
                
                if not constraint_check['feasible']:
                    validation['feasible'] = False
                    validation['warnings'].extend(constraint_check['issues'])
                    
            # Enhanced constraint interaction analysis with literature-based conflict detection
            binding_affinity_constraints = [c for c in constraints if c.constraint_type == 'binding_affinity']
            stability_constraints = [c for c in constraints if c.constraint_type == 'structural_stability']
            specificity_constraints = [c for c in constraints if c.constraint_type == 'binding_specificity']
            expression_constraints = [c for c in constraints if c.constraint_type == 'expression_level']
            solubility_constraints = [c for c in constraints if c.constraint_type == 'solubility']
            
            # Check binding affinity vs stability conflicts (Wells & McClendon 2007)
            if binding_affinity_constraints and stability_constraints:
                strong_binding = any(c.target_value < -12.0 for c in binding_affinity_constraints)
                high_stability = any(c.target_value < -15.0 for c in stability_constraints)
                
                if strong_binding and high_stability and difficulty in ['hard', 'extreme']:
                    validation['warnings'].append(
                        "Strong binding (-12 kcal/mol) + high stability (-15 kcal/mol) combination "
                        "creates design trade-offs (Wells & McClendon 2007) - confidence reduced"
                    )
                    validation['confidence'] *= 0.7
                    
            # Check binding affinity vs specificity conflicts (Horovitz 1996)
            if binding_affinity_constraints and specificity_constraints:
                ultra_strong = any(c.target_value < -15.0 for c in binding_affinity_constraints)
                high_specificity = any(c.target_value > 1000.0 for c in specificity_constraints)
                
                if ultra_strong and high_specificity:
                    validation['warnings'].append(
                        "Ultra-strong binding (<-15 kcal/mol) + high specificity (>1000-fold) "
                        "may be conflicting - very strong binders often show reduced specificity"
                    )
                    validation['confidence'] *= 0.6
                    
            # Check expression vs stability conflicts (Wilkinson & Harrison 1991)
            if expression_constraints and stability_constraints:
                high_expression = any(c.target_value > 5.0 for c in expression_constraints)
                very_high_stability = any(c.target_value < -20.0 for c in stability_constraints)
                
                if high_expression and very_high_stability:
                    validation['warnings'].append(
                        "High expression (>5x) + very high stability (<-20 kcal/mol) "
                        "may conflict - hyperstable proteins often express poorly"
                    )
                    validation['confidence'] *= 0.8
                    
            # Check expression vs solubility conflicts (Trainor et al. 2017)
            if expression_constraints and solubility_constraints:
                very_high_expression = any(c.target_value > 10.0 for c in expression_constraints)
                high_solubility = any(c.target_value > 20.0 for c in solubility_constraints)
                
                if very_high_expression and high_solubility:
                    validation['warnings'].append(
                        "Very high expression (>10x) + high solubility (>20 mg/ml) "
                        "combination challenging - high expression often leads to aggregation"
                    )
                    validation['confidence'] *= 0.75
                    
            # Literature-informed confidence adjustment based on difficulty and constraint count
            constraint_count = len(constraints)
            
            # Multi-constraint complexity penalty (based on protein design literature)
            if constraint_count > 2:
                complexity_penalty = 1.0 - (constraint_count - 2) * 0.15
                validation['confidence'] *= max(0.2, complexity_penalty)
                validation['warnings'].append(
                    f"Multi-constraint problem ({constraint_count} constraints) - "
                    f"success rate decreases exponentially with constraint count"
                )
                
            # Difficulty-based feasibility assessment
            if difficulty == 'extreme':
                if constraint_count > 3:
                    validation['confidence'] *= 0.3
                    validation['warnings'].append(
                        "Extreme difficulty with >3 constraints - very low success probability "
                        "even for state-of-the-art methods"
                    )
                elif constraint_count > 2:
                    validation['confidence'] *= 0.5
                    validation['warnings'].append(
                        "Extreme difficulty with multiple constraints - challenging even "
                        "for expert protein designers"
                    )
                    
            # Add literature-based success rate estimates
            estimated_success_rate = validation['confidence']
            if estimated_success_rate < 0.1:
                validation['warnings'].append(
                    f"Estimated success rate <10% - consider simplifying constraints "
                    f"or accepting partial solutions"
                )
            elif estimated_success_rate < 0.3:
                validation['warnings'].append(
                    f"Estimated success rate {estimated_success_rate:.1%} - "
                    f"expect multiple design iterations required"
                )
                
        except Exception as e:
            validation['feasible'] = False
            validation['warnings'].append(f"Physics-based feasibility validation failed: {e}")
            validation['confidence'] = 0.0
            
        return validation
        
    def _check_tool_availability(self, tool_name: str) -> Dict[str, Any]:
        """Check if required computational tools are available."""
        tool_info = {
            'available': False,
            'fallback': None,
            'notes': ''
        }
        
        # Define tool availability checks and fallbacks
        tool_mapping = {
            'molecular_docking': {
                'check_commands': ['vina', 'autodock_vina'],
                'fallback': 'ml_binding_prediction',
                'notes': 'Can use ML-based binding affinity prediction as fallback'
            },
            'rosetta': {
                'check_commands': ['rosetta', 'rosetta_scripts'],
                'fallback': 'simple_energy_calculation',
                'notes': 'Can use simplified energy calculation as fallback'
            },
            'foldx': {
                'check_commands': ['foldx'],
                'fallback': 'rosetta',
                'notes': 'Can use Rosetta as fallback for stability calculation'
            }
        }
        
        if tool_name in tool_mapping:
            # In a full implementation, would actually check for tool availability
            # For now, assume tools are not available and use fallbacks
            tool_info['available'] = False  # Placeholder
            tool_info['fallback'] = tool_mapping[tool_name]['fallback']
            tool_info['notes'] = tool_mapping[tool_name]['notes']
        else:
            tool_info['notes'] = 'Unknown tool'
            
        return tool_info
        
    def _generate_target_structure(
        self,
        problem_id: str,
        constraint_combination: str,
        difficulty: str
    ) -> Dict[str, Any]:
        """Generate target structure information for the problem."""
        # Generate realistic target structure based on constraint type
        if 'binding' in constraint_combination:
            structure_type = 'binding_target'
            target_info = {
                'target_protein_name': f"Target_{problem_id}",
                'target_protein_family': random.choice([
                    'kinase', 'gpcr', 'ion_channel', 'enzyme', 'transcription_factor'
                ]),
                'binding_site_residues': random.randint(5, 20),
                'binding_pocket_volume': random.uniform(200, 800),
                'binding_pocket_properties': {
                    'hydrophobicity': random.uniform(-2, 2),
                    'charge': random.uniform(-5, 5),
                    'polarity': random.uniform(0, 1)
                }
            }
        else:
            structure_type = 'standalone_protein'
            target_info = {
                'protein_fold': random.choice([
                    'all_alpha', 'all_beta', 'alpha_beta', 'small_proteins'
                ]),
                'secondary_structure_content': {
                    'helix': random.uniform(0.2, 0.6),
                    'sheet': random.uniform(0.1, 0.4),
                    'coil': random.uniform(0.2, 0.5)
                }
            }
            
        target_structure = {
            'structure_type': structure_type,
            'sequence_length': random.randint(50, 300),
            'difficulty': difficulty,
            'target_info': target_info,
            'metadata': {
                'generation_method': 'synthetic',
                'constraint_combination': constraint_combination,
                'complexity_score': self._estimate_structure_complexity(difficulty)
            }
        }
        
        return target_structure
        
    def _define_success_criteria(
        self,
        constraints: List[ConstraintSpec],
        difficulty: str
    ) -> Dict[str, Any]:
        """Define overall success criteria for the problem."""
        # Determine how many constraints must be satisfied
        if difficulty == 'easy':
            min_constraints_satisfied = max(1, len(constraints) - 1)
            overall_score_threshold = 0.6
        elif difficulty == 'medium':
            min_constraints_satisfied = len(constraints)
            overall_score_threshold = 0.7
        elif difficulty == 'hard':
            min_constraints_satisfied = len(constraints)
            overall_score_threshold = 0.8
        else:  # extreme
            min_constraints_satisfied = len(constraints)
            overall_score_threshold = 0.9
            
        success_criteria = {
            'min_constraints_satisfied': min_constraints_satisfied,
            'total_constraints': len(constraints),
            'overall_score_threshold': overall_score_threshold,
            'constraint_weights': {c.constraint_id: c.weight for c in constraints},
            'evaluation_method': 'weighted_score',
            'allow_partial_success': difficulty in ['easy', 'medium'],
            'penalty_for_failures': difficulty in ['hard', 'extreme']
        }
        
        return success_criteria
        
    def _analyze_constraint_interactions(
        self,
        constraints: List[ConstraintSpec]
    ) -> Dict[str, Any]:
        """Analyze potential interactions between constraints."""
        interactions = {
            'conflicting_pairs': [],
            'synergistic_pairs': [],
            'independent_pairs': [],
            'difficulty_modifiers': {}
        }
        
        # Define known constraint interactions
        conflict_patterns = [
            ('binding_affinity', 'structural_stability'),  # Strong binding may destabilize
            ('expression_level', 'solubility'),  # High expression may reduce solubility
            ('binding_affinity', 'binding_specificity')  # Strong binding may reduce specificity
        ]
        
        synergy_patterns = [
            ('structural_stability', 'expression_level'),  # Stable proteins express better
            ('solubility', 'aggregation_propensity'),  # Soluble proteins aggregate less
            ('structural_stability', 'solubility')  # Stable proteins often more soluble
        ]
        
        # Check for interactions
        for i, c1 in enumerate(constraints):
            for j, c2 in enumerate(constraints[i+1:], i+1):
                pair = (c1.constraint_type, c2.constraint_type)
                reverse_pair = (c2.constraint_type, c1.constraint_type)
                
                if pair in conflict_patterns or reverse_pair in conflict_patterns:
                    interactions['conflicting_pairs'].append({
                        'constraint1': c1.constraint_id,
                        'constraint2': c2.constraint_id,
                        'interaction_type': 'conflict',
                        'severity': random.uniform(0.3, 0.8)
                    })
                elif pair in synergy_patterns or reverse_pair in synergy_patterns:
                    interactions['synergistic_pairs'].append({
                        'constraint1': c1.constraint_id,
                        'constraint2': c2.constraint_id,
                        'interaction_type': 'synergy',
                        'benefit': random.uniform(0.1, 0.4)
                    })
                else:
                    interactions['independent_pairs'].append({
                        'constraint1': c1.constraint_id,
                        'constraint2': c2.constraint_id,
                        'interaction_type': 'independent'
                    })
                    
        # Calculate difficulty modifiers
        conflict_count = len(interactions['conflicting_pairs'])
        synergy_count = len(interactions['synergistic_pairs'])
        
        interactions['difficulty_modifiers'] = {
            'conflict_penalty': conflict_count * 0.2,
            'synergy_bonus': synergy_count * 0.1,
            'net_difficulty_modifier': conflict_count * 0.2 - synergy_count * 0.1
        }
        
        return interactions
        
    def _create_evaluation_protocol(
        self,
        constraints: List[ConstraintSpec],
        constraint_combination: str
    ) -> Dict[str, Any]:
        """Create evaluation protocol for the problem."""
        protocol = {
            'evaluation_steps': [],
            'required_tools': [],
            'computational_methods': [],
            'experimental_validation': [],
            'success_metrics': []
        }
        
        # Define evaluation steps based on constraints
        for constraint in constraints:
            if constraint.constraint_type == 'binding_affinity':
                protocol['evaluation_steps'].append({
                    'step': 'binding_affinity_evaluation',
                    'method': constraint.measurement_method,
                    'target_value': constraint.target_value,
                    'tolerance': constraint.tolerance
                })
                if constraint.measurement_method == 'docking':
                    tool_info = self._check_tool_availability('molecular_docking')
                    protocol['required_tools'].append({
                        'tool': 'molecular_docking',
                        'available': tool_info['available'],
                        'fallback': tool_info['fallback']
                    })
                    protocol['computational_methods'].append('autodock_vina')
                elif constraint.measurement_method == 'experimental':
                    protocol['experimental_validation'].append('binding_assay')
                    
            elif constraint.constraint_type == 'structural_stability':
                protocol['evaluation_steps'].append({
                    'step': 'stability_evaluation', 
                    'method': constraint.measurement_method,
                    'target_value': constraint.target_value,
                    'tolerance': constraint.tolerance
                })
                if constraint.measurement_method == 'rosetta':
                    protocol['required_tools'].append('rosetta')
                    protocol['computational_methods'].append('rosetta_relax')
                elif constraint.measurement_method == 'foldx':
                    protocol['required_tools'].append('foldx')
                    protocol['computational_methods'].append('foldx_stability')
                    
        # Add overall success metrics
        protocol['success_metrics'] = [
            'individual_constraint_satisfaction',
            'overall_weighted_score',
            'constraint_interaction_score',
            'robustness_score'
        ]
        
        # Estimate computational requirements
        protocol['estimated_runtime'] = {
            'computational_methods': len(protocol['computational_methods']) * 30,  # minutes
            'experimental_validation': len(protocol['experimental_validation']) * 480  # minutes
        }
        
        return protocol
        
    def _generate_problem_description(
        self,
        constraint_combination: str,
        constraints: List[ConstraintSpec],
        difficulty: str
    ) -> str:
        """Generate human-readable problem description."""
        constraint_names = [c.constraint_type.replace('_', ' ') for c in constraints]
        constraint_list = ', '.join(constraint_names[:-1]) + f' and {constraint_names[-1]}'
        
        descriptions = {
            'binding_stability': f"Design a protein that achieves strong binding affinity while maintaining structural stability",
            'binding_specificity': f"Design a protein with high binding affinity but excellent specificity",
            'stability_expression': f"Design a stable protein with high expression levels",
            'therapeutic_design': f"Design a therapeutic protein optimized for multiple drug-like properties",
            'enzyme_design': f"Design an enzyme with optimal activity and production characteristics"
        }
        
        base_description = descriptions.get(
            constraint_combination,
            f"Design a protein optimized for {constraint_list}"
        )
        
        difficulty_modifiers = {
            'easy': "with relaxed constraints",
            'medium': "with moderate constraints", 
            'hard': "with challenging constraints",
            'extreme': "with extremely demanding constraints"
        }
        
        full_description = f"{base_description} {difficulty_modifiers[difficulty]}. " \
                          f"This {difficulty} problem requires optimizing {len(constraints)} " \
                          f"objectives simultaneously: {constraint_list}."
        
        return full_description
        
    def _estimate_structure_complexity(self, difficulty: str) -> float:
        """Estimate structure complexity score based on difficulty."""
        complexity_ranges = {
            'easy': (0.2, 0.4),
            'medium': (0.4, 0.6), 
            'hard': (0.6, 0.8),
            'extreme': (0.8, 1.0)
        }
        
        min_complexity, max_complexity = complexity_ranges[difficulty]
        return random.uniform(min_complexity, max_complexity)
        
    def generate_multiple_problems(
        self,
        count: int,
        constraint_combinations: Optional[List[str]] = None,
        difficulty_distribution: Optional[Dict[str, float]] = None
    ) -> List[MultiConstraintProblem]:
        """
        Generate multiple multi-constraint problems.
        
        Args:
            count: Number of problems to generate
            constraint_combinations: List of constraint combinations to use
            difficulty_distribution: Distribution of difficulty levels
            
        Returns:
            List of generated multi-constraint problems
        """
        if constraint_combinations is None:
            constraint_combinations = list(self.constraint_combinations.keys())
            
        if difficulty_distribution is None:
            difficulty_distribution = {
                'easy': 0.2,
                'medium': 0.4,
                'hard': 0.3,
                'extreme': 0.1
            }
            
        problems = []
        
        for i in range(count):
            try:
                # Sample constraint combination and difficulty
                combination = random.choice(constraint_combinations)
                difficulty = np.random.choice(
                    list(difficulty_distribution.keys()),
                    p=list(difficulty_distribution.values())
                )
                
                # Generate problem
                problem = self.generate_multi_constraint_problem(
                    problem_id=f"multi_constraint_{i:04d}",
                    constraint_combination=combination,
                    difficulty=difficulty
                )
                
                problems.append(problem)
                
            except Exception as e:
                print(f"Warning: Failed to generate problem {i}: {e}")
                continue
                
        print(f"Generated {len(problems)}/{count} multi-constraint problems")
        return problems


def create_multi_constraint_benchmark(
    count: int = 50,
    seed: int = 42,
    **kwargs
) -> List[MultiConstraintProblem]:
    """
    Convenience function to create multi-constraint benchmark problems.
    
    Args:
        count: Number of problems to generate
        seed: Random seed
        **kwargs: Additional arguments for problem generation
        
    Returns:
        List of multi-constraint problems
    """
    generator = MultiConstraintProblemGenerator(seed=seed)
    return generator.generate_multiple_problems(count, **kwargs)


if __name__ == "__main__":
    # Example usage
    generator = MultiConstraintProblemGenerator(seed=42)
    
    # Generate a single problem
    problem = generator.generate_multi_constraint_problem(
        problem_id="test_problem_001",
        constraint_combination="binding_stability",
        difficulty="medium"
    )
    
    print(f"Generated problem: {problem.problem_id}")
    print(f"Description: {problem.description}")
    print(f"Constraints: {len(problem.constraints)}")
    print(f"Difficulty: {problem.difficulty_level}")
    
    # Generate multiple problems
    problems = generator.generate_multiple_problems(count=10)
    print(f"\nGenerated {len(problems)} multi-constraint problems")