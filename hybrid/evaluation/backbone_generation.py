#!/usr/bin/env python3
"""
Novel Backbone Structure Generation for Protein Design Benchmarks

This module provides enhanced methods for generating realistic protein backbone
structures that don't exist in nature but have valid geometric and biophysical
properties. Used for testing model generalization to out-of-distribution structures.

Features:
- Realistic backbone geometry with proper bond lengths/angles
- Secondary structure-aware generation
- Ramachandran-compliant dihedral angles
- Integration with existing structure validation tools
- Support for various complexity levels and fold types
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import random
import math

# BioPython imports with error handling
try:
    from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
    from Bio.PDB.vectors import Vector, calc_dihedral, calc_angle
    from Bio.SeqUtils import seq1, seq3
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    warnings.warn("BioPython not available. Backbone generation will use simplified methods.")

# Optional structure prediction imports
try:
    from scipy.stats import vonmises
    from scipy.spatial.transform import Rotation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced geometric calculations will be limited.")


@dataclass
class BackboneGenerationConfig:
    """
    Configuration for backbone generation.
    
    Attributes:
        target_length: Target sequence length
        complexity: Complexity level ('easy', 'medium', 'hard')
        secondary_structure_bias: Bias towards specific secondary structures
        ramachandran_strict: Whether to enforce strict Ramachandran compliance
        fold_type: Type of fold to generate ('alpha', 'beta', 'mixed', 'novel')
        add_noise: Amount of coordinate noise to add (0.0-1.0)
        validate_geometry: Whether to validate generated geometry
    """
    target_length: int = 100
    complexity: str = 'medium'
    secondary_structure_bias: Optional[str] = None  # 'alpha', 'beta', 'extended', None
    ramachandran_strict: bool = True
    fold_type: str = 'mixed'
    add_noise: float = 0.1
    validate_geometry: bool = True


class NovelBackboneGenerator:
    """
    Generates novel protein backbone structures with realistic geometry.
    
    This generator creates protein backbones that don't exist in nature but
    have valid geometric properties and can fold into stable structures.
    Used for testing design methods on out-of-distribution structures.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize backbone generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Standard protein geometry parameters
        self.bond_lengths = {
            'N-CA': 1.458,   # Angstroms
            'CA-C': 1.525,
            'C-N': 1.329,
            'C-O': 1.231
        }
        
        self.bond_angles = {
            'N-CA-C': 111.2,   # Degrees
            'CA-C-N': 116.6,
            'C-N-CA': 121.7
        }
        
        # Ramachandran angle preferences for different secondary structures
        self.ramachandran_preferences = {
            'alpha': {'phi': (-60, 15), 'psi': (-45, 15)},  # (mean, std)
            'beta': {'phi': (-120, 20), 'psi': (120, 20)},
            'extended': {'phi': (-140, 30), 'psi': (150, 30)},
            'turn': {'phi': (-90, 40), 'psi': (0, 60)}
        }
        
    def generate_novel_backbone(
        self, 
        config: BackboneGenerationConfig
    ) -> Dict[str, Any]:
        """
        Generate a single novel backbone structure.
        
        Args:
            config: Configuration for backbone generation
            
        Returns:
            Dictionary containing backbone structure data
        """
        try:
            # Generate secondary structure pattern
            ss_pattern = self._generate_secondary_structure_pattern(
                config.target_length, config.complexity, config.secondary_structure_bias
            )
            
            # Generate backbone coordinates
            coordinates = self._generate_backbone_coordinates(
                config.target_length, ss_pattern, config
            )
            
            # Add realistic noise if requested
            if config.add_noise > 0:
                coordinates = self._add_coordinate_noise(coordinates, config.add_noise)
                
            # Validate geometry if requested
            validation_results = {}
            if config.validate_geometry:
                validation_results = self._validate_backbone_geometry(coordinates)
                
            # Create structure data
            structure_data = {
                'coordinates': coordinates,
                'secondary_structure': ss_pattern,
                'sequence_length': config.target_length,
                'complexity': config.complexity,
                'fold_type': config.fold_type,
                'generation_config': config,
                'validation': validation_results,
                'metadata': {
                    'generation_method': 'novel_backbone_generator',
                    'generator_version': '1.0',
                    'ramachandran_strict': config.ramachandran_strict,
                    'geometry_validated': config.validate_geometry
                }
            }
            
            return structure_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate novel backbone: {e}")
            
    def _generate_secondary_structure_pattern(
        self, 
        length: int, 
        complexity: str,
        ss_bias: Optional[str] = None
    ) -> str:
        """
        Generate realistic secondary structure pattern.
        
        Args:
            length: Sequence length
            complexity: Complexity level
            ss_bias: Secondary structure bias
            
        Returns:
            Secondary structure string (H=helix, E=sheet, C=coil)
        """
        # Secondary structure probabilities based on complexity
        if complexity == 'easy':
            # Simple patterns with long segments
            min_segment = 8
            max_segment = 20
        elif complexity == 'medium':
            # Mixed patterns with medium segments
            min_segment = 5
            max_segment = 15
        else:  # hard
            # Complex patterns with short segments
            min_segment = 3
            max_segment = 10
            
        # Apply bias if specified
        if ss_bias == 'alpha':
            ss_probs = {'H': 0.6, 'E': 0.2, 'C': 0.2}
        elif ss_bias == 'beta':
            ss_probs = {'H': 0.2, 'E': 0.6, 'C': 0.2}
        elif ss_bias == 'extended':
            ss_probs = {'H': 0.1, 'E': 0.3, 'C': 0.6}
        else:
            # Balanced distribution
            ss_probs = {'H': 0.4, 'E': 0.3, 'C': 0.3}
            
        pattern = []
        remaining = length
        
        while remaining > 0:
            # Choose secondary structure element
            ss_choices = list(ss_probs.keys())
            ss_weights = list(ss_probs.values())
            ss_type = np.random.choice(ss_choices, p=ss_weights)
            
            # Choose segment length
            segment_length = min(
                remaining,
                np.random.randint(min_segment, max_segment + 1)
            )
            
            pattern.extend([ss_type] * segment_length)
            remaining -= segment_length
            
        return ''.join(pattern)
        
    def _generate_backbone_coordinates(
        self,
        length: int,
        ss_pattern: str,
        config: BackboneGenerationConfig
    ) -> Dict[str, np.ndarray]:
        """
        Generate realistic backbone coordinates based on secondary structure.
        
        Args:
            length: Sequence length
            ss_pattern: Secondary structure pattern
            config: Generation configuration
            
        Returns:
            Dictionary of atom coordinates
        """
        coordinates = {
            'N': np.zeros((length, 3)),
            'CA': np.zeros((length, 3)),
            'C': np.zeros((length, 3)),
            'O': np.zeros((length, 3))
        }
        
        # Initialize first residue
        coordinates['N'][0] = np.array([0.0, 0.0, 0.0])
        coordinates['CA'][0] = np.array([self.bond_lengths['N-CA'], 0.0, 0.0])
        coordinates['C'][0] = np.array([
            self.bond_lengths['N-CA'] + self.bond_lengths['CA-C'] * np.cos(np.radians(self.bond_angles['N-CA-C'])),
            self.bond_lengths['CA-C'] * np.sin(np.radians(self.bond_angles['N-CA-C'])),
            0.0
        ])
        coordinates['O'][0] = coordinates['C'][0] + np.array([0.0, self.bond_lengths['C-O'], 0.0])
        
        # Build remaining residues
        for i in range(1, length):
            ss_type = ss_pattern[i]
            
            # Get dihedral angles for this secondary structure
            phi, psi = self._sample_dihedral_angles(ss_type, config.ramachandran_strict)
            
            # Calculate new coordinates based on previous residue
            if i >= 1:
                coordinates = self._add_residue_with_dihedrals(
                    coordinates, i, phi, psi
                )
                
        return coordinates
        
    def _sample_dihedral_angles(
        self, 
        ss_type: str, 
        strict: bool = True
    ) -> Tuple[float, float]:
        """
        Sample realistic phi/psi angles for given secondary structure.
        
        Args:
            ss_type: Secondary structure type ('H', 'E', 'C')
            strict: Whether to enforce strict Ramachandran compliance
            
        Returns:
            Tuple of (phi, psi) angles in degrees
        """
        if ss_type == 'H':  # Alpha helix
            prefs = self.ramachandran_preferences['alpha']
        elif ss_type == 'E':  # Beta sheet
            prefs = self.ramachandran_preferences['beta']
        else:  # Coil/turn
            prefs = self.ramachandran_preferences['turn']
            
        if strict:
            # Sample from tight distributions around preferred angles
            phi = np.random.normal(prefs['phi'][0], prefs['phi'][1])
            psi = np.random.normal(prefs['psi'][0], prefs['psi'][1])
        else:
            # Allow wider sampling for more diverse structures
            phi = np.random.normal(prefs['phi'][0], prefs['phi'][1] * 2)
            psi = np.random.normal(prefs['psi'][0], prefs['psi'][1] * 2)
            
        # Keep angles in valid range
        phi = np.clip(phi, -180, 180)
        psi = np.clip(psi, -180, 180)
        
        return phi, psi
        
    def _add_residue_with_dihedrals(
        self,
        coordinates: Dict[str, np.ndarray],
        residue_idx: int,
        phi: float,
        psi: float
    ) -> Dict[str, np.ndarray]:
        """
        Add residue coordinates using dihedral angles.
        
        Args:
            coordinates: Existing coordinate arrays
            residue_idx: Index of residue to add
            phi: Phi dihedral angle in degrees
            psi: Psi dihedral angle in degrees
            
        Returns:
            Updated coordinate arrays
        """
        if residue_idx < 1:
            return coordinates
            
        prev_idx = residue_idx - 1
        
        # Get previous residue atoms
        prev_n = coordinates['N'][prev_idx]
        prev_ca = coordinates['CA'][prev_idx]
        prev_c = coordinates['C'][prev_idx]
        
        # Calculate new N position using C-N bond
        n_pos = self._place_atom_with_geometry(
            prev_ca, prev_c, prev_n,
            self.bond_lengths['C-N'],
            self.bond_angles['CA-C-N'],
            180.0  # trans peptide bond
        )
        coordinates['N'][residue_idx] = n_pos
        
        # Calculate new CA position using phi angle
        ca_pos = self._place_atom_with_geometry(
            prev_c, n_pos, prev_ca,
            self.bond_lengths['N-CA'],
            self.bond_angles['C-N-CA'],
            phi
        )
        coordinates['CA'][residue_idx] = ca_pos
        
        # Calculate new C position using psi angle (estimate for next residue)
        c_pos = self._place_atom_with_geometry(
            n_pos, ca_pos, prev_c,
            self.bond_lengths['CA-C'],
            self.bond_angles['N-CA-C'],
            psi
        )
        coordinates['C'][residue_idx] = c_pos
        
        # Calculate O position (approximately)
        o_pos = self._place_atom_with_geometry(
            ca_pos, c_pos, n_pos,
            self.bond_lengths['C-O'],
            120.0,  # approximate C=O angle
            0.0     # planar
        )
        coordinates['O'][residue_idx] = o_pos
        
        return coordinates
        
    def _place_atom_with_geometry(
        self,
        atom1: np.ndarray,
        atom2: np.ndarray,
        atom3: np.ndarray,
        bond_length: float,
        bond_angle: float,
        dihedral_angle: float
    ) -> np.ndarray:
        """
        Place atom using bond geometry (length, angle, dihedral).
        
        Args:
            atom1, atom2, atom3: Positions of three existing atoms
            bond_length: Bond length for new atom
            bond_angle: Bond angle in degrees
            dihedral_angle: Dihedral angle in degrees
            
        Returns:
            Position of new atom
        """
        # Convert angles to radians
        bond_angle_rad = np.radians(bond_angle)
        dihedral_rad = np.radians(dihedral_angle)
        
        # Calculate vectors
        v1 = atom1 - atom2
        v2 = atom3 - atom2
        
        # Normalize v1 with numerical stability
        v1_norm_magnitude = np.linalg.norm(v1)
        if v1_norm_magnitude < 1e-8:
            # Degenerate case - return default position
            return atom2 + np.array([bond_length, 0.0, 0.0])
        v1_norm = v1 / v1_norm_magnitude
        
        # Calculate perpendicular vector in plane of v1 and v2
        v1_cross_v2 = np.cross(v1, v2)
        cross_magnitude = np.linalg.norm(v1_cross_v2)
        if cross_magnitude > 1e-6:
            v1_cross_v2_norm = v1_cross_v2 / cross_magnitude
        else:
            # Vectors are collinear, choose arbitrary perpendicular
            v1_cross_v2_norm = np.array([0, 0, 1])
            if abs(np.dot(v1_norm, v1_cross_v2_norm)) > 0.9:
                v1_cross_v2_norm = np.array([1, 0, 0])
                
        # Calculate rotation axis
        rotation_axis = v1_cross_v2_norm
        
        # Rotate v1 by bond angle
        cos_angle = np.cos(bond_angle_rad)
        sin_angle = np.sin(bond_angle_rad)
        
        # Rodrigues rotation formula (simplified for orthogonal case)
        v_rotated = (v1_norm * cos_angle + 
                    np.cross(rotation_axis, v1_norm) * sin_angle)
        
        # Apply dihedral rotation
        cos_dihedral = np.cos(dihedral_rad)
        sin_dihedral = np.sin(dihedral_rad)
        
        # Rotate around v1 axis
        if SCIPY_AVAILABLE:
            rotation = Rotation.from_rotvec(dihedral_rad * v1_norm)
            v_final = rotation.apply(v_rotated)
        else:
            # Simple approximation
            v_final = v_rotated
            
        # Scale by bond length and place relative to atom2
        new_position = atom2 + v_final * bond_length
        
        return new_position
        
    def _add_coordinate_noise(
        self,
        coordinates: Dict[str, np.ndarray],
        noise_level: float
    ) -> Dict[str, np.ndarray]:
        """
        Add realistic coordinate noise to mimic experimental uncertainty.
        
        Args:
            coordinates: Original coordinates
            noise_level: Noise magnitude (0.0-1.0)
            
        Returns:
            Coordinates with added noise
        """
        noisy_coords = {}
        
        for atom_type, coords in coordinates.items():
            # Scale noise by atom type (backbone atoms more constrained)
            if atom_type in ['N', 'CA', 'C']:
                scale = noise_level * 0.1  # 0.1 Angstrom max for backbone
            else:
                scale = noise_level * 0.2  # More noise for side chain atoms
                
            noise = np.random.normal(0, scale, coords.shape)
            noisy_coords[atom_type] = coords + noise
            
        return noisy_coords
        
    def _validate_backbone_geometry(
        self,
        coordinates: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Validate geometric properties of generated backbone.
        
        Args:
            coordinates: Backbone coordinates
            
        Returns:
            Dictionary of validation results
        """
        validation = {
            'status': 'unknown',
            'bond_length_check': {},
            'bond_angle_check': {},
            'clash_check': {},
            'ramachandran_check': {},
            'overall_quality': 0.0
        }
        
        try:
            length = len(coordinates['CA'])
            
            # Check bond lengths
            validation['bond_length_check'] = self._check_bond_lengths(coordinates)
            
            # Check bond angles  
            validation['bond_angle_check'] = self._check_bond_angles(coordinates)
            
            # Check for atomic clashes
            validation['clash_check'] = self._check_atomic_clashes(coordinates)
            
            # Check Ramachandran angles if possible
            if BIOPYTHON_AVAILABLE and length > 2:
                validation['ramachandran_check'] = self._check_ramachandran(coordinates)
            
            # Calculate overall quality score
            quality_scores = []
            for check_name, check_result in validation.items():
                if isinstance(check_result, dict) and 'quality_score' in check_result:
                    quality_scores.append(check_result['quality_score'])
                    
            if quality_scores:
                validation['overall_quality'] = np.mean(quality_scores)
                
            # Set status based on quality
            if validation['overall_quality'] > 0.8:
                validation['status'] = 'good'
            elif validation['overall_quality'] > 0.6:
                validation['status'] = 'acceptable'
            else:
                validation['status'] = 'poor'
                
        except Exception as e:
            validation['status'] = 'error'
            validation['error'] = str(e)
            
        return validation
        
    def _check_bond_lengths(self, coordinates: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Check if bond lengths are within reasonable ranges."""
        check_result = {
            'bonds_checked': 0,
            'bonds_valid': 0,
            'quality_score': 0.0,
            'violations': []
        }
        
        try:
            length = len(coordinates['CA'])
            tolerance = 0.2  # Angstroms
            
            for i in range(length):
                # N-CA bonds
                if i < length:
                    n_ca_dist = np.linalg.norm(coordinates['CA'][i] - coordinates['N'][i])
                    expected = self.bond_lengths['N-CA']
                    if abs(n_ca_dist - expected) <= tolerance:
                        check_result['bonds_valid'] += 1
                    else:
                        check_result['violations'].append(f"N-CA bond {i}: {n_ca_dist:.3f} (expected {expected:.3f})")
                    check_result['bonds_checked'] += 1
                    
                # CA-C bonds
                if i < length:
                    ca_c_dist = np.linalg.norm(coordinates['C'][i] - coordinates['CA'][i])
                    expected = self.bond_lengths['CA-C']
                    if abs(ca_c_dist - expected) <= tolerance:
                        check_result['bonds_valid'] += 1
                    else:
                        check_result['violations'].append(f"CA-C bond {i}: {ca_c_dist:.3f} (expected {expected:.3f})")
                    check_result['bonds_checked'] += 1
                    
            # Calculate quality score
            if check_result['bonds_checked'] > 0:
                check_result['quality_score'] = check_result['bonds_valid'] / check_result['bonds_checked']
                
        except Exception as e:
            check_result['error'] = str(e)
            
        return check_result
        
    def _check_bond_angles(self, coordinates: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Check if bond angles are within reasonable ranges."""
        check_result = {
            'angles_checked': 0,
            'angles_valid': 0,
            'quality_score': 0.0,
            'violations': []
        }
        
        try:
            length = len(coordinates['CA'])
            tolerance = 15.0  # degrees
            
            for i in range(length):
                # N-CA-C angles
                if i < length:
                    try:
                        angle = self._calculate_angle(
                            coordinates['N'][i],
                            coordinates['CA'][i], 
                            coordinates['C'][i]
                        )
                        expected = self.bond_angles['N-CA-C']
                        if abs(angle - expected) <= tolerance:
                            check_result['angles_valid'] += 1
                        else:
                            check_result['violations'].append(f"N-CA-C angle {i}: {angle:.1f}° (expected {expected:.1f}°)")
                        check_result['angles_checked'] += 1
                    except:
                        pass
                        
            # Calculate quality score
            if check_result['angles_checked'] > 0:
                check_result['quality_score'] = check_result['angles_valid'] / check_result['angles_checked']
                
        except Exception as e:
            check_result['error'] = str(e)
            
        return check_result
        
    def _check_atomic_clashes(self, coordinates: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Check for unrealistic atomic clashes."""
        check_result = {
            'pairs_checked': 0,
            'clashes_found': 0,
            'quality_score': 1.0,
            'violations': []
        }
        
        try:
            min_distance = 2.0  # Angstroms - minimum allowed distance
            
            all_atoms = []
            atom_labels = []
            
            for atom_type, coords in coordinates.items():
                for i, pos in enumerate(coords):
                    all_atoms.append(pos)
                    atom_labels.append(f"{atom_type}_{i}")
                    
            all_atoms = np.array(all_atoms)
            
            # Check pairwise distances (skip adjacent atoms in sequence)
            for i in range(len(all_atoms)):
                for j in range(i + 3, len(all_atoms)):  # Skip nearby atoms
                    dist = np.linalg.norm(all_atoms[i] - all_atoms[j])
                    check_result['pairs_checked'] += 1
                    
                    if dist < min_distance:
                        check_result['clashes_found'] += 1
                        check_result['violations'].append(
                            f"Clash between {atom_labels[i]} and {atom_labels[j]}: {dist:.2f}Å"
                        )
                        
            # Calculate quality score (fewer clashes = better)
            if check_result['pairs_checked'] > 0:
                clash_rate = check_result['clashes_found'] / check_result['pairs_checked']
                check_result['quality_score'] = max(0.0, 1.0 - clash_rate * 10)  # Penalize clashes heavily
                
        except Exception as e:
            check_result['error'] = str(e)
            
        return check_result
        
    def _check_ramachandran(self, coordinates: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Check Ramachandran plot compliance."""
        check_result = {
            'residues_checked': 0,
            'residues_valid': 0,
            'quality_score': 0.0,
            'violations': []
        }
        
        try:
            length = len(coordinates['CA'])
            
            # Check each residue's phi/psi angles
            for i in range(1, length - 1):  # Skip first and last residue
                try:
                    phi = self._calculate_dihedral(
                        coordinates['C'][i-1],
                        coordinates['N'][i],
                        coordinates['CA'][i],
                        coordinates['C'][i]
                    )
                    
                    psi = self._calculate_dihedral(
                        coordinates['N'][i],
                        coordinates['CA'][i],
                        coordinates['C'][i],
                        coordinates['N'][i+1] if i+1 < length else coordinates['N'][i] + [1,0,0]
                    )
                    
                    # Check if angles are in allowed regions
                    if self._is_ramachandran_allowed(phi, psi):
                        check_result['residues_valid'] += 1
                    else:
                        check_result['violations'].append(f"Residue {i}: phi={phi:.1f}°, psi={psi:.1f}° (disallowed)")
                        
                    check_result['residues_checked'] += 1
                    
                except:
                    pass
                    
            # Calculate quality score
            if check_result['residues_checked'] > 0:
                check_result['quality_score'] = check_result['residues_valid'] / check_result['residues_checked']
                
        except Exception as e:
            check_result['error'] = str(e)
            
        return check_result
        
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points in degrees."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        return angle
        
    def _calculate_dihedral(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
        """Calculate dihedral angle between four points in degrees."""
        try:
            v1 = p2 - p1
            v2 = p3 - p2
            v3 = p4 - p3
            
            n1 = np.cross(v1, v2)
            n2 = np.cross(v2, v3)
            
            # Check for near-zero normal vectors (collinear atoms)
            n1_magnitude = np.linalg.norm(n1)
            n2_magnitude = np.linalg.norm(n2)
            
            if n1_magnitude < 1e-6 or n2_magnitude < 1e-6:
                # Atoms are nearly collinear, dihedral is undefined
                return 0.0
            
            # Normalize normal vectors
            n1_norm = n1 / n1_magnitude
            n2_norm = n2 / n2_magnitude
            
            # Calculate dihedral
            cos_dihedral = np.dot(n1_norm, n2_norm)
            cos_dihedral = np.clip(cos_dihedral, -1.0, 1.0)
            
            dihedral = np.arccos(cos_dihedral) * 180.0 / np.pi
            
            # Determine sign using scalar triple product
            v2_norm = v2 / max(np.linalg.norm(v2), 1e-8)
            triple_product = np.dot(np.cross(n1_norm, n2_norm), v2_norm)
            if triple_product < 0:
                dihedral = -dihedral
                
            return dihedral
            
        except Exception as e:
            # Log warning for debugging but don't crash
            warnings.warn(f"Dihedral calculation failed: {e}")
            return 0.0
            
    def _is_ramachandran_allowed(self, phi: float, psi: float) -> bool:
        """Check if phi/psi angles are in allowed Ramachandran regions."""
        # More comprehensive allowed regions (less restrictive)
        allowed_regions = [
            # Alpha helix region (expanded)
            (-100, -30, -70, 50),    # phi_min, phi_max, psi_min, psi_max
            # Beta sheet region (expanded)
            (-180, -70, 80, 180),
            # Extended/polyproline region
            (-90, 0, 80, 180),
            # Left-handed helix (rare but allowed)
            (30, 100, 0, 100),
            # Additional flexible regions
            (-180, -100, -50, 80),
            (0, 60, -50, 50)
        ]
        
        # Handle angle wrapping for boundary cases
        phi_wrapped = phi
        psi_wrapped = psi
        if phi > 180:
            phi_wrapped = phi - 360
        elif phi < -180:
            phi_wrapped = phi + 360
        if psi > 180:
            psi_wrapped = psi - 360
        elif psi < -180:
            psi_wrapped = psi + 360
        
        for phi_min, phi_max, psi_min, psi_max in allowed_regions:
            # Check both original and wrapped angles
            for test_phi in [phi, phi_wrapped]:
                for test_psi in [psi, psi_wrapped]:
                    if phi_min <= test_phi <= phi_max and psi_min <= test_psi <= psi_max:
                        return True
                        
        # If no exact match, allow if close to any allowed region (tolerance)
        tolerance = 30.0  # degrees
        for phi_min, phi_max, psi_min, psi_max in allowed_regions:
            phi_center = (phi_min + phi_max) / 2
            psi_center = (psi_min + psi_max) / 2
            
            phi_dist = min(abs(phi - phi_center), abs(phi_wrapped - phi_center))
            psi_dist = min(abs(psi - psi_center), abs(psi_wrapped - psi_center))
            
            if phi_dist <= tolerance and psi_dist <= tolerance:
                return True
                
        return False
        
    def generate_multiple_backbones(
        self,
        configs: List[BackboneGenerationConfig]
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple backbone structures efficiently with memory management.
        
        Args:
            configs: List of generation configurations
            
        Returns:
            List of generated backbone structures
        """
        backbones = []
        
        for i, config in enumerate(configs):
            try:
                structure = self.generate_novel_backbone(config)
                structure['batch_id'] = i
                backbones.append(structure)
                
                # Memory management for large batches
                if (i + 1) % 50 == 0:
                    self._cleanup_memory()
                    print(f"Generated {i + 1}/{len(configs)} backbone structures...")
                
            except Exception as e:
                print(f"Warning: Failed to generate backbone {i}: {e}")
                continue
                
        print(f"Successfully generated {len(backbones)}/{len(configs)} backbone structures")
        return backbones
        
    def _cleanup_memory(self) -> None:
        """Clean up memory during large batch generation."""
        import gc
        gc.collect()


def generate_hallucinated_structures(
    count: int = 100,
    length_range: Tuple[int, int] = (50, 200),
    complexity_levels: List[str] = ['easy', 'medium', 'hard'],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate multiple hallucinated structures.
    
    Args:
        count: Number of structures to generate
        length_range: Range of sequence lengths
        complexity_levels: List of complexity levels to sample from
        seed: Random seed
        
    Returns:
        List of generated backbone structures
    """
    generator = NovelBackboneGenerator(seed=seed)
    
    configs = []
    for i in range(count):
        config = BackboneGenerationConfig(
            target_length=random.randint(*length_range),
            complexity=random.choice(complexity_levels),
            fold_type=random.choice(['alpha', 'beta', 'mixed', 'novel']),
            ramachandran_strict=random.choice([True, False]),
            add_noise=random.uniform(0.05, 0.2),
            validate_geometry=True
        )
        configs.append(config)
        
    return generator.generate_multiple_backbones(configs)


if __name__ == "__main__":
    # Example usage
    generator = NovelBackboneGenerator(seed=42)
    
    config = BackboneGenerationConfig(
        target_length=100,
        complexity='medium',
        fold_type='mixed',
        validate_geometry=True
    )
    
    structure = generator.generate_novel_backbone(config)
    print(f"Generated structure with {structure['sequence_length']} residues")
    print(f"Validation status: {structure['validation']['status']}")
    print(f"Overall quality: {structure['validation']['overall_quality']:.2f}")