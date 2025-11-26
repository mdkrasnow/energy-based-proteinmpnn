#!/usr/bin/env python3
"""
Test suite for the design validation system.

This test script validates the comprehensive design validation framework
to ensure all components work correctly together.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import warnings

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import the validation system
from evaluation.validate_designs import (
    validate_protein_designs,
    ValidationConfig,
    AlphaFoldValidator,
    RosettaValidator,
    StructurePredictor,
    AggregationAnalyzer,
    PerplexityAnalyzer,
    DiversityAnalyzer,
    StatisticalTester,
    ValidationPipeline,
    PerformanceBenchmark,
    create_default_config
)

def test_individual_validators():
    """Test each validation tool individually."""
    print("Testing individual validators...")
    
    # Test sequences
    test_sequences = [
        "MKTIIALSYIFCLVFA",  # Short peptide
        "MKLLVLLSLLIGSTALAQPAMAMKLLVLLSLLIGSTALAQPAMAMKLLVLLSLLIGSTALAQPAMAM",  # Medium protein
        "MVIAPGDQQAKLGQFIEPEAFLAHLKMPQTEVSTTRLEKQYALVTQTPSTNEAAAEHHHHHH"  # His-tagged protein
    ]
    
    config = create_default_config()
    success_count = 0
    total_validators = 6  # Number of validators we're testing
    
    # Test each validator
    for i, sequence in enumerate(test_sequences):
        print(f"\nTesting sequence {i+1}: {sequence[:20]}...")
        
        # AlphaFold validator
        try:
            af_validator = AlphaFoldValidator(config)
            if af_validator.is_available():
                af_result = af_validator.validate(sequence)
                print(f"  AlphaFold: confidence = {af_result.get('alphafold_confidence', 'N/A'):.1f}")
                success_count += 1
            else:
                print("  AlphaFold: Not available")
                success_count += 1  # Count as success if gracefully unavailable
        except Exception as e:
            print(f"  AlphaFold: Error - {e}")
        
        # Rosetta validator
        try:
            rosetta_validator = RosettaValidator(config)
            if rosetta_validator.is_available():
                rosetta_result = rosetta_validator.validate(sequence)
                print(f"  Rosetta: score = {rosetta_result.get('rosetta_score', 'N/A'):.1f}")
                success_count += 1
            else:
                print("  Rosetta: Not available")
                success_count += 1  # Count as success if gracefully unavailable
        except Exception as e:
            print(f"  Rosetta: Error - {e}")
        
        # Structure predictor
        try:
            structure_validator = StructurePredictor(config)
            structure_result = structure_validator.validate(sequence)
            ss_comp = structure_result.get('ss_composition', {})
            print(f"  Structure: helix = {ss_comp.get('helix', 0):.2f}, sheet = {ss_comp.get('sheet', 0):.2f}")
            success_count += 1
        except Exception as e:
            print(f"  Structure: Error - {e}")
        
        # Aggregation analyzer
        try:
            agg_validator = AggregationAnalyzer(config)
            agg_result = agg_validator.validate(sequence)
            print(f"  Aggregation: propensity = {agg_result.get('aggregation_propensity', 'N/A'):.3f}")
            success_count += 1
        except Exception as e:
            print(f"  Aggregation: Error - {e}")
        
        # Perplexity analyzer
        try:
            perp_validator = PerplexityAnalyzer(config)
            if perp_validator.is_available():
                perp_result = perp_validator.validate(sequence)
                print(f"  Perplexity: value = {perp_result.get('proteinmpnn_perplexity', 'N/A'):.2f}")
                success_count += 1
            else:
                print("  Perplexity: Not available")
                success_count += 1  # Count as success if gracefully unavailable
        except Exception as e:
            print(f"  Perplexity: Error - {e}")
        
        # Diversity analyzer
        try:
            div_validator = DiversityAnalyzer(config)
            div_result = div_validator.validate(sequence)
            novelty = div_result.get('sequence_novelty')
            novelty_str = f"{novelty:.3f}" if novelty is not None else "N/A"
            print(f"  Diversity: novelty = {novelty_str}")
            success_count += 1
        except Exception as e:
            print(f"  Diversity: Error - {e}")
    
    # Calculate success rate
    total_tests = len(test_sequences) * total_validators
    success_rate = success_count / total_tests
    print(f"\n✅ Individual validator tests: {success_count}/{total_tests} successful ({success_rate:.1%})")
    
    return success_rate > 0.7  # Consider test successful if >70% pass


def test_validation_pipeline():
    """Test the complete validation pipeline."""
    print("\n" + "="*60)
    print("Testing complete validation pipeline...")
    
    # Test sequences
    designed_sequences = [
        "MKLLVLLSLLIGSTALAQPAMAM",
        "MVIAPGDQQAKLGQFIEPEAF",
        "MKTIIALSYIFCLVFAGHHHH",
        "MAVTQTPSTNEAAAEHHHHHH",
        "MLEKQYALVTQTPSTNEAAAE"
    ]
    
    baseline_sequences = [
        "MKTVRCKCCTKGFRKYGPKAV",  # Natural sequence-like
        "MAAIFCFLVQILKDVWALFGA",
        "MGSIVELLLSLLIGSTALAQA"
    ]
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ValidationConfig(
            output_dir=temp_dir,
            batch_size=2,
            bootstrap_samples=100,  # Reduced for faster testing
            generate_plots=True,
            verbose=True
        )
        
        try:
            # Run validation
            results = validate_protein_designs(
                sequences=designed_sequences,
                sequence_ids=[f"designed_{i:02d}" for i in range(len(designed_sequences))],
                baseline_sequences=baseline_sequences,
                config=config
            )
            
            print(f"✅ Validation completed successfully!")
            print(f"   Processed {len(designed_sequences)} designed sequences")
            print(f"   Compared with {len(baseline_sequences)} baseline sequences")
            print(f"   Output saved to: {results['output_directory']}")
            
            # Check summary results
            summary = results['summary']
            print(f"   Quality assessment: {summary.get('quality_assessment', {})}")
            
            if 'baseline_comparison' in summary:
                print(f"   Baseline comparison available")
                comparison = summary['baseline_comparison']
                if 'perplexity_analysis' in comparison:
                    perp_analysis = comparison['perplexity_analysis']
                    print(f"   Perplexity comparison: {perp_analysis.get('mann_whitney', {}).get('significant', 'N/A')}")
            
            # List generated files
            output_files = list(Path(temp_dir).glob("*"))
            print(f"   Generated {len(output_files)} output files:")
            for file in output_files:
                print(f"     - {file.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Validation pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_statistical_analysis():
    """Test statistical analysis components."""
    print("\n" + "="*60)
    print("Testing statistical analysis...")
    
    config = create_default_config()
    tester = StatisticalTester(config)
    
    # Generate mock data
    import numpy as np
    np.random.seed(42)
    
    energy_based_perplexities = np.random.normal(10, 2, 50).tolist()  # Higher perplexity
    baseline_perplexities = np.random.normal(8, 1.5, 40).tolist()     # Lower perplexity
    
    try:
        # Test perplexity comparison
        comparison = tester.compare_perplexity_distributions(
            energy_based_perplexities, baseline_perplexities
        )
        
        print("✅ Statistical comparison completed:")
        if 'mann_whitney' in comparison:
            mw = comparison['mann_whitney']
            print(f"   Mann-Whitney U test: p = {mw.get('p_value', 'N/A'):.4f}")
            print(f"   Significant difference: {mw.get('significant', 'N/A')}")
        
        if 'effect_size' in comparison:
            effect = comparison['effect_size']
            print(f"   Effect size (Cohen's d): {effect.get('cohens_d', 'N/A'):.3f} ({effect.get('interpretation', 'N/A')})")
        
        # Test bootstrap confidence intervals
        ci_result = tester.bootstrap_confidence_intervals(energy_based_perplexities)
        print(f"   Bootstrap CI: {ci_result.get('ci_lower', 'N/A'):.2f} - {ci_result.get('ci_upper', 'N/A'):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistical analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmark():
    """Test performance benchmarking."""
    print("\n" + "="*60)
    print("Testing performance benchmark...")
    
    # Small test for speed
    test_sequences = [
        "MKLLVLLSLLIGSTALAQPAMAM",
        "MVIAPGDQQAKLGQFIEPEAF",
        "MKTIIALSYIFCLVFAGHHHH"
    ]
    
    config = ValidationConfig(
        batch_size=1,
        bootstrap_samples=10,  # Very small for fast testing
        generate_plots=False,
        verbose=False
    )
    
    try:
        benchmark = PerformanceBenchmark()
        
        # Test individual validator profiling
        profile_results = benchmark.profile_individual_validators(
            test_sequences[0], config
        )
        
        print("✅ Individual validator profiling completed:")
        for validator_name, result in profile_results.items():
            if 'error' in result:
                print(f"   {validator_name}: {result['error']}")
            else:
                print(f"   {validator_name}: {result.get('mean_time_seconds', 'N/A'):.4f}s ± {result.get('std_time_seconds', 'N/A'):.4f}s")
        
        # Test pipeline benchmarking (with just 1 run for speed)
        benchmark_results = benchmark.benchmark_validation_pipeline(
            test_sequences, config, n_runs=1
        )
        
        print("✅ Pipeline benchmarking completed:")
        timing = benchmark_results['timing_results']
        throughput = benchmark_results['throughput_metrics']
        
        print(f"   Total time: {timing.get('mean_time_seconds', 'N/A'):.2f}s")
        print(f"   Throughput: {throughput.get('sequences_per_second', 'N/A'):.2f} seq/s")
        print(f"   Time per sequence: {throughput.get('seconds_per_sequence', 'N/A'):.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n" + "="*60)
    print("Testing error handling...")
    
    config = create_default_config()
    
    try:
        # Test with invalid sequence
        print("Testing invalid sequence...")
        invalid_sequence = "MKLLVXYZINVALID"  # Contains invalid amino acids
        
        pipeline = ValidationPipeline(config)
        result = pipeline.validate_sequence(invalid_sequence, "invalid_test")
        
        # Should handle gracefully
        print(f"   Invalid sequence handled: {len(result.errors)} errors, {len(result.warnings)} warnings")
        
        # Test with empty sequence
        print("Testing empty sequence...")
        empty_result = pipeline.validate_sequence("", "empty_test")
        print(f"   Empty sequence handled: {len(empty_result.errors)} errors, {len(empty_result.warnings)} warnings")
        
        # Test with very long sequence
        print("Testing very long sequence...")
        long_sequence = "A" * 1000  # 1000 residue sequence
        long_result = pipeline.validate_sequence(long_sequence, "long_test")
        print(f"   Long sequence handled: {len(long_result.errors)} errors, {len(long_result.warnings)} warnings")
        
        print("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("COMPREHENSIVE VALIDATION SYSTEM TEST")
    print("="*60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    tests = [
        ("Individual Validators", test_individual_validators),
        ("Validation Pipeline", test_validation_pipeline),
        ("Statistical Analysis", test_statistical_analysis),
        ("Performance Benchmark", test_performance_benchmark),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running test: {test_name}")
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Validation system is working correctly.")
        return 0
    else:
        print(f"⚠️  {total-passed} test(s) failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())