# Deep Research: `run_comprehensive_evaluation.py` Data Requirements

## Executive Summary
The `run_comprehensive_evaluation.py` script requires at least one of three specific data files to run successfully: `optimization_data_file`, `landscape_data_file`, or `benchmark_data_file`. The error message you're seeing is caused by the script not being able to find any of these files. To fix this, you need to provide the correct paths to your data files via a JSON configuration file or command-line arguments.

## Research Scope
- **Original question:** What evaluation data is needed for `run_comprehensive_evaluation.py` to work properly?
- **Files/systems analyzed:** `hybrid/evaluation/run_comprehensive_evaluation.py`
- **Time period examined:** Current version of the file.

## Key Findings

### Finding 1: Data File Requirements
The script explicitly looks for three types of data files:

1.  `optimization_data_file`: Used for convergence and adaptive computation analysis.
2.  `landscape_data_file`: Used for landscape quality analysis.
3.  `benchmark_data_file`: Used for performance analysis.

**Evidence:**
- `hybrid/evaluation/run_comprehensive_evaluation.py:636-655` - This block of code attempts to load each of the three data files.
- `hybrid/evaluation/run_comprehensive_evaluation.py:658-660` - This code checks if any of the data files were loaded successfully and raises an error if not.

**Analysis:**
The script is designed to be modular, allowing different analyses to be run independently based on the data provided. However, it requires at least one data file to be present to perform any analysis at all.

**Confidence:** High

### Finding 2: Data Schemas
The script includes a `DataValidator` class that defines the expected schema for each data file.

**Evidence:**
- `hybrid/evaluation/run_comprehensive_evaluation.py:82-97` - This section defines the `DATA_VALIDATION_SCHEMAS` dictionary, which outlines the required and optional fields for each data type.
- `hybrid/evaluation/run_comprehensive_evaluation.py:317-399` - This is the implementation of the `DataValidator` class, which uses the schemas to validate the input data.

**Analysis:**
The presence of a data validator indicates that the script is designed to be robust against malformed input data. To ensure that your data is processed correctly, you should make sure that it conforms to the schemas defined in the script.

**Confidence:** High

## Patterns Identified

### Design Patterns
- **Lazy Loading:** The script uses a `ComponentLoader` to lazy-load analysis components, which is a good practice for reducing startup time and memory usage.
- **Configuration-driven:** The script is designed to be configured via a JSON file or command-line arguments, which makes it flexible and easy to use in different environments.

### Architectural Patterns
- **Modular Architecture:** The script is divided into several components (data loading, validation, analysis, reporting), which makes it easy to understand and maintain.

## Recommendations
1.  **Provide Data Files:** To fix the error, you need to provide at least one of the three required data files. The easiest way to do this is to create a JSON configuration file and use the `--config` command-line argument to specify its path.

    ```json
    {
      "optimization_data_file": "/path/to/your/optimization_data.json",
      "landscape_data_file": "/path/to/your/landscape_data.json",
      "benchmark_data_file": "/path/to/your/benchmark_data.json",
      "output_directory": "./evaluation_results"
    }
    ```
2.  **Validate Your Data:** Before running the script, make sure that your data files conform to the schemas defined in the `DATA_VALIDATION_SCHEMAS` dictionary. This will help you avoid errors during the analysis.

## Sources Consulted
- **Files read:** 1 file (`hybrid/evaluation/run_comprehensive_evaluation.py`)
- **Lines of code analyzed:** ~1700
- **Key directories:** `hybrid/evaluation`
