# **`README.md`**

# Fused LLM Router: Information-Aware and Risk-Aware Optimization of Large Language Model Ensembles

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Source](https://img.shields.io/badge/Source-Author's%20Concept%20Paper-003366)](https://github.com/chirindaopensource/fused_llm_router/blob/main/FMEC_concept_paper_final.md)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/fused_llm_router)
[![Discipline: CS](https://img.shields.io/badge/Discipline-Computer%20Science-00529B)](https://github.com/chirindaopensource/fused_llm_router)
[![Discipline: AI](https://img.shields.io/badge/Discipline-Artificial%20Intelligence-00529B)](https://github.com/chirindaopensource/fused_llm_router)
[![Discipline: OR](https://img.shields.io/badge/Discipline-Operations%20Research-00529B)](https://github.com/chirindaopensource/fused_llm_router)
[![Discipline: Game Theory](https://img.shields.io/badge/Discipline-Cooperative%20Game%20Theory-00529B)](https://github.com/chirindaopensource/fused_llm_router)
[![Discipline: Info Theory](https://img.shields.io/badge/Discipline-Information%20Theory-00529B)](https://github.com/chirindaopensource/fused_llm_router)
[![Data: MMLU](https://img.shields.io/badge/Data-MMLU-lightgrey)](https://github.com/hendrycks/test)
[![Data: MATH](https://img.shields.io/badge/Data-MATH-lightgrey)](https://github.com/hendrycks/math)
[![Data: GSM8K](https://img.shields.io/badge/Data-GSM8K-lightgrey)](https://github.com/openai/grade-school-math)
[![Data: GPQA](https://img.shields.io/badge/Data-GPQA-lightgrey)](https://github.com/idavidrein/gpqa)
[![Data: HumanEval](https://img.shields.io/badge/Data-HumanEval-lightgrey)](https://github.com/openai/human-eval)
[![Data: SWE-bench](https://img.shields.io/badge/Data-SWE--bench-lightgrey)](https://github.com/princeton-nlp/SWE-bench)
[![Method: Shapley](https://img.shields.io/badge/Method-Shapley%20Value%20Estimation-orange)](https://github.com/chirindaopensource/fused_llm_router)
[![Method: CMI](https://img.shields.io/badge/Method-Conditional%20Mutual%20Information-orange)](https://github.com/chirindaopensource/fused_llm_router)
[![Method: Ledoit-Wolf](https://img.shields.io/badge/Method-Ledoit--Wolf%20Shrinkage-orange)](https://github.com/chirindaopensource/fused_llm_router)
[![Method: Submodular](https://img.shields.io/badge/Method-Submodular%20Optimization-orange)](https://github.com/chirindaopensource/fused_llm_router)
[![Method: Bootstrap](https://img.shields.io/badge/Method-Nested%20Cluster%20Bootstrap-orange)](https://github.com/chirindaopensource/fused_llm_router)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/fused_llm_router)

**Repository:** `https://github.com/chirindaopensource/fused_llm_router`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

## Table of Contents
- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: run_experimental_suite](#key-callable-run_experimental_suite)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project is an independent, professional-grade implementation of the ideas, methodologies, and experimental protocols from the paper titled **"Beyond Benchmark Rankings: Information-Aware and Risk-Aware Optimization of Large Language Model Ensembles (A Concept Paper and Methodology)"** by the author:
*   **Craig S. Chirinda (CS Chirinda)**

This repository provides a complete, end-to-end computational framework for replicating the paper's findings. It challenges the prevailing paradigm of deploying Large Language Models (LLMs) based solely on standalone benchmark leaderboards. Instead, it treats LLM orchestration as a rigorous portfolio-construction problem. By implementing the **Final Marginal Ensemble Contribution (FMEC)** framework, this codebase delivers a highly optimized pipeline that evaluates models based on their expected marginal utility, conditional informational novelty, and correlated-failure risk, ultimately selecting the optimal ensemble under strict cost, latency, and risk budgets.

## Theoretical Background

The implemented methods bridge machine learning systems engineering with advanced cooperative game theory, information theory, and financial econometrics.

**1. Cooperative Game Theory (Marginal Utility):**
The pipeline moves beyond standalone scoring by evaluating models based on their conditional value to an ensemble. It implements a permutation-based Monte Carlo Shapley estimator to calculate the expected marginal utility contribution ($\widehat{\mathrm{MEC}}_j$) of each model across thousands of potential coalitions, ensuring an unbiased, order-symmetric valuation.

**2. Information Theory (Informational Novelty):**
To distinguish true complementary capability from mere disagreement, the framework quantifies informational novelty using outcome-based Conditional Mutual Information (CMI). It implements Laplace smoothing and Miller-Madow finite-sample bias corrections over the correct-count sufficient statistic ($c_S$) to compute the conditional information gain ($\mathrm{IG}_j$), preventing spurious diversity claims.

**3. Portfolio Theory & Robust Statistics (Correlated Risk):**
Translating Markowitz portfolio theory to LLM errors, the pipeline constructs a multidimensional error taxonomy. It estimates the error covariance matrix ($\Sigma$) using Ledoit-Wolf shrinkage targeting a block-diagonal geometry, penalizing models that fail in the same way. This yields the Marginal Risk Contribution ($\mathrm{MRC}_j$).

**4. Constrained Submodular Optimization:**
The FMEC scores are aggregated into a deployment utility function. The pipeline executes a budget-feasible greedy algorithm to maximize portfolio utility subject to strict, multi-dimensional knapsack constraints (cost, latency, and risk), identifying the optimal ensemble on the efficient frontier.

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/fused_llm_router/blob/main/fused_llm_router_draft_ipo_main_9.png" alt="Pipeline Architecture" width="100%">
</div>

## Features

-   **End-to-End Automation:** A singular orchestrator function (`run_experimental_suite`) executes the entire research protocol from raw data ingestion and contamination auditing to cryptographic packaging and LaTeX report generation.
-   **Strict Data Governance:** Implements a Tier-1 reproducibility gate, canonicalizing the outcome tensor (fixed column order, float precision) and computing a SHA-256 hash to guarantee bit-exact computational reproducibility.
-   **Contamination Auditing:** Enforces a mandatory precondition for valid inference by executing 13-gram lexical overlap and behavioral canary probes, applying a global drop-for-all exclusion rule to prune contaminated tasks.
-   **Secure Verification Engineering:** Grounds correctness deterministically for code-generation tasks by executing candidate solutions in an isolated, network-disabled, restricted-filesystem subprocess with strict timeouts.
-   **Hierarchical Uncertainty Quantification:** Implements a three-level nested cluster bootstrap to isolate and quantify task-sampling, generation, and adjudication variance ($\sigma^2_{\mathrm{task}} + \sigma^2_{\mathrm{gen}} + \sigma^2_{\mathrm{adj}}$), computing Bias-Corrected and Accelerated (BCa) intervals.

## Methodology Implemented

1.  **Data Ingestion & Validation:** Materializes raw records into strictly typed pandas DataFrames, enforcing relational integrity and schema contracts.
2.  **Contamination Audit:** Executes lexical and behavioral probes to filter the evaluation corpus, protecting the causal interpretation of complementarity.
3.  **Qualification & Stratification:** Applies the $P_{\min}$ benchmark gate and clusters models by capability profile ($k$-means) to ensure lineage diversity during coalition sampling.
4.  **Tensor Assembly:** Constructs the canonical outcome tensor, mathematically enforcing the error exclusivity invariant ($\sum e_* = 1 - \text{correct}$) and computing exact USD costs.
5.  **Estimation:** Computes marginal utility (Shapley), conditional information gain (CMI), and correlated-failure risk (Ledoit-Wolf shrinkage) to compose the FMEC scores.
6.  **Optimization:** Executes a budget-feasible greedy search to select the optimal model portfolio under cost, latency, and risk constraints.
7.  **Validation & Inference:** Runs the nested bootstrap, computes Recommendation-Stability Probability (RSP), and executes pre-registered hypothesis tests (H1-H5) with Holm-Bonferroni FWER control.
8.  **Robustness & Packaging:** Performs 10,000 Dirichlet preference sweeps, generates publication-grade figures, and serializes all artifacts into a cryptographically hashed archive.

## Core Components (Notebook Structure)

*Note: All orchestrator callables and their constituent helper functions are contained within a singular, comprehensive Jupyter Notebook (`fused_llm_router_draft.ipynb`).*

The notebook is structured as a logical Directed Acyclic Graph (DAG) of 24 distinct tasks:
1.  Environment, Configuration & Seed Initialization (Task 1-2)
2.  Raw Data Ingestion, Schema Validation & Cleansing (Tasks 3-4)
3.  Benchmark Contamination Audit & Qualification (Tasks 5-7)
4.  Orchestration Architecture & Response Collection (Tasks 8-9)
5.  Scoring, Adjudication & Tensor Assembly (Tasks 10-11)
6.  FMEC Estimation Core: MEC, IG, and Risk (Tasks 12-16)
7.  Constrained Ensemble Selection & Fusion Audit (Tasks 17-18)
8.  Nested Bootstrap Validation & Confirmatory Testing (Tasks 19-21)
9.  Robustness Sweeps, Reporting & Cryptographic Packaging (Tasks 22-24)

## Key Callable: `run_experimental_suite`

The project is designed around a single, top-level user-facing interface function:

-   **`run_experimental_suite`:** This apex orchestrator function coordinates the entire FMEC experimental protocol. A single call to this function ingests the raw data, executes the contamination audit, builds the canonical outcome tensor, estimates the FMEC primitives, solves the constrained optimization problem, runs the nested bootstrap and Dirichlet robustness sweeps, generates the scientific report and figures, and packages the results into an immutable `.tar.gz` archive. It guarantees complete state isolation and prevents look-ahead bias.

## Prerequisites

-   Python 3.10+
-   Core Python dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyyaml`, `faker` (for synthetic testing).

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/fused_llm_router.git
    cd fused_llm_router
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Input Data Structure

The pipeline requires five primary raw data structures, adhering strictly to the schemas defined in Section 9.1 and 9.2 of the methodology:
1.  **`model_answer`**: A list of dictionaries containing raw free-form generation records (9 keys, including verbatim text and provenance).
2.  **`candidate_solution`**: A list of dictionaries containing raw structured (verifier-augmented) records (7 keys, including executable Python code).
3.  **`task_record`**: A list of dictionaries defining the evaluation substrate, ground truth, and scoring rules.
4.  **`outcomes`**: The canonical raw outcome-and-error tensor (17 columns). While the pipeline can assemble this from raw responses, it serves as the fundamental deterministic substrate for all downstream estimation.
5.  **`qualification`**: A pandas DataFrame containing benchmark eligibility scores and model lineage metadata.

Additionally, the pipeline requires a **`config.yaml`** file, which serves as the deterministic source of truth for all hyperparameters, optimization budgets, and reproducibility seeds.

## Usage

Here is the granular, step-by-step guide to executing the end-to-end pipeline for **"Beyond Benchmark Rankings: Information-Aware and Risk-Aware Optimization of Large Language Model Ensembles"**. This example demonstrates how to synthetically generate the required raw data structures, load the study configuration from a YAML file, and execute the full research pipeline using the `run_experimental_suite` orchestrator.

*Note: Assume that all the callables that have been defined in this conversation are in a single Jupyter notebook. And, that there is no folder with “.py” functions.*
*Note: Assume that the “config.yaml” file is saved in the working directory.*

### **Step 1: Synthetic Data Generation (`raw_fmec_data`)**

The first requirement is a high-fidelity synthetic representation of the raw data structures. These datasets must adhere strictly to the schemas and mathematical invariants defined in Section 9.1 and 9.2 of the study (e.g., the exclusivity invariant for error flags, and the disjointness invariant for evaluation tasks).

**Methodology:**
1.  **Qualification Generation:** We simulate a pool of candidate models with plausible benchmark percentiles, ensuring the schema exactly matches the 10 required columns.
2.  **Task Generation:** We generate a set of evaluation tasks spanning different domains, strictly enforcing the `is_held_out = True` invariant.
3.  **Response Generation:** We simulate $R=5$ replicate generations per (model, task) pair, populating token counts, latencies, and verbatim text payloads using the `Faker` library.
4.  **Outcome Tensor Generation:** We construct the canonical 17-column outcome tensor, mathematically enforcing the error exclusivity invariant ($\sum e_* = 1 - \text{correct}$).

```python
# Import the pandas library for DataFrame manipulation
import pandas as pd
# Import the numpy library for numerical operations and random sampling
import numpy as np
# Import the yaml library for configuration parsing
import yaml
# Import the os module for environment variable access
import os
# Import the Faker library to generate realistic synthetic text
from faker import Faker
# Import typing modules for strict type hinting
from typing import Dict, Any, List, Tuple

# Initialize the Faker instance to generate synthetic text payloads
fake = Faker()
# Set a deterministic seed for Faker to ensure reproducible text generation
Faker.seed(20260618)
# Set a deterministic seed for NumPy to ensure reproducible numerical generation
np.random.seed(20260618)

def generate_synthetic_fmec_data(
    n_models: int = 5,
    n_tasks: int = 10,
    n_replicates: int = 5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """
    Generates high-fidelity synthetic data structures required for the FMEC pipeline.

    Purpose:
        To create mathematically plausible mock datasets that strictly adhere to the
        schemas defined in Section 9.1 and 9.2 of the FMEC methodology. This allows
        for the execution and testing of the pipeline without requiring expensive
        live LLM API calls.

    Inputs:
        n_models (int): The number of candidate models to simulate. Default is 5.
        n_tasks (int): The number of evaluation tasks to simulate. Default is 10.
        n_replicates (int): The number of generation replicates (R). Default is 5.

    Processes:
        1.  Qualification Generation: Creates a DataFrame of models with benchmark scores.
        2.  Task Generation: Creates a list of dictionaries defining held-out tasks.
        3.  Response Generation: Simulates free-form and structured LLM outputs.
        4.  Outcome Tensor Generation: Constructs a DataFrame enforcing the error
            exclusivity invariant (correct == 1 implies all e_* == 0).

    Outputs:
        Tuple containing:
            - model_answer (List[Dict]): Raw free-form generation records (9 keys).
            - candidate_solution (List[Dict]): Raw structured solution records (7 keys).
            - task_record (List[Dict]): Raw task and ground truth records (8 keys).
            - outcomes (pd.DataFrame): Canonical raw outcome-and-error tensor (17 columns).
            - qualification (pd.DataFrame): Raw benchmark-eligibility table (10 columns).

    Raises:
        TypeError: If input parameters are not integers.
        ValueError: If input dimensions are less than 1.
    """
    # Validate that n_models is an integer
    if not isinstance(n_models, int):
        # Raise a TypeError if the validation fails
        raise TypeError("n_models must be an integer.")
    # Validate that n_models is strictly positive
    if n_models < 1:
        # Raise a ValueError to prevent generating empty datasets
        raise ValueError("n_models must be at least 1.")
    
    # Validate that n_tasks is an integer
    if not isinstance(n_tasks, int):
        # Raise a TypeError if the validation fails
        raise TypeError("n_tasks must be an integer.")
    # Validate that n_tasks is strictly positive
    if n_tasks < 1:
        # Raise a ValueError to prevent generating empty datasets
        raise ValueError("n_tasks must be at least 1.")
        
    # Validate that n_replicates is an integer
    if not isinstance(n_replicates, int):
        # Raise a TypeError if the validation fails
        raise TypeError("n_replicates must be an integer.")
    # Validate that n_replicates is strictly positive
    if n_replicates < 1:
        # Raise a ValueError to prevent generating empty datasets
        raise ValueError("n_replicates must be at least 1.")

    # Define a list of synthetic model identifiers
    model_ids: List[str] = [f"model_{chr(65+i)}" for i in range(n_models)]
    # Define a list of synthetic task identifiers
    task_ids: List[str] = [f"task_{str(i).zfill(3)}" for i in range(n_tasks)]

    # --- 1. Generate Qualification DataFrame ---
    # Initialize an empty list to hold qualification records
    qual_records: List[Dict[str, Any]] = []
    # Iterate over each synthetic model ID
    for m_id in model_ids:
        # Simulate a composite benchmark percentile rank between 0.60 and 0.99
        pct_rank = float(np.random.uniform(0.60, 0.99))
        # Append the record dictionary matching the Section 9.2(e) schema exactly
        qual_records.append({
            "model_id": m_id,
            "family": str(np.random.choice(["Alpha", "Beta", "Gamma"])),
            "params_b": float(np.random.choice([70.0, 140.0, np.nan])),
            "mmlu": float(np.random.uniform(0.5, 0.9)),
            "gpqa": float(np.random.uniform(0.3, 0.7)),
            "humaneval": float(np.random.uniform(0.4, 0.9)),
            "swebench": float(np.random.uniform(0.1, 0.5)),
            "math": float(np.random.uniform(0.3, 0.8)),
            "arena_elo": int(np.random.uniform(1000, 1300)),
            "pct_rank": pct_rank
        })
    # Convert the list of dictionaries into a pandas DataFrame
    qualification = pd.DataFrame(qual_records)
    # Enforce strict categorical dtype for the primary key to optimize memory
    qualification["model_id"] = qualification["model_id"].astype("category")

    # --- 2. Generate Task Records ---
    # Initialize an empty list to hold task records
    task_record: List[Dict[str, Any]] = []
    # Iterate over each synthetic task ID
    for t_id in task_ids:
        # Randomly assign a domain from the allowed set
        domain = str(np.random.choice(["math", "code", "science"]))
        # Determine the scoring rule based on the domain
        rule = "unit_tests" if domain == "code" else "exact_match"
        # Append the record dictionary matching the Section 9.1(c) schema exactly
        task_record.append({
            "task_id": t_id,
            "domain": domain,
            "subdomain": "general",
            "prompt": fake.sentence(nb_words=10),
            "ground_truth": "42" if domain == "math" else "def solve(): pass",
            "scoring_rule": rule,
            "difficulty": "medium",
            "is_held_out": True  # Mandatory invariant for evaluation tasks (Assumption A1)
        })

    # --- 3. Generate Responses and Outcomes ---
    # Initialize the list for free-form answers
    model_answer: List[Dict[str, Any]] = []
    # Initialize the list for structured solutions
    candidate_solution: List[Dict[str, Any]] = []
    # Initialize the list for tensor rows
    outcome_rows: List[Dict[str, Any]] = []

    # Define the exact eight error columns for the taxonomy
    error_cols: List[str] = ["e_reason", "e_comp", "e_halluc", "e_ctx", "e_plan", "e_tool", "e_safety", "e_refusal"]

    # Iterate over every combination of task, model, and replicate
    for t_id in task_ids:
        # Iterate over models
        for m_id in model_ids:
            # Iterate over replicates
            for rep in range(n_replicates):
                # Simulate a realistic wall-clock latency in seconds
                latency = float(np.random.uniform(0.5, 5.0))
                # Simulate input token usage
                p_tokens = int(np.random.uniform(100, 1000))
                # Simulate output token usage
                c_tokens = int(np.random.uniform(50, 500))
                # Calculate a synthetic cost based on token usage
                cost = float((p_tokens * 0.0001) + (c_tokens * 0.0002))
                # Generate a deterministic seed for this specific call
                seed = int(20260618 + rep)

                # Append the free-form generation record matching Section 9.1(a) exactly
                model_answer.append({
                    "task_id": t_id,
                    "model_id": m_id,
                    "answer": fake.paragraph(nb_sentences=3),
                    "error": None,
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "latency_s": latency,
                    "temperature": 0.20,
                    "seed": seed
                })

                # Append the structured solution record matching Section 9.1(b) exactly
                candidate_solution.append({
                    "task_id": t_id,
                    "model_id": m_id,
                    "thesis": fake.sentence(),
                    "reasoning": fake.paragraph(),
                    "python_code": "print('hello')" if "code" in t_id else None,
                    "confidence": float(np.random.uniform(0.5, 1.0)),
                    "error": None
                })

                # Simulate a binary correctness indicator (e.g., 70% accuracy)
                is_correct = int(np.random.rand() > 0.3)
                
                # Initialize the error vector with all zeros
                e_vec: Dict[str, int] = {col: 0 for col in error_cols}
                
                # Enforce the mathematical exclusivity invariant
                if is_correct == 0:
                    # If incorrect, select exactly one error category uniformly at random
                    chosen_error = str(np.random.choice(error_cols))
                    # Set the chosen error flag to 1
                    e_vec[chosen_error] = 1

                # Construct the tensor row dictionary matching Section 9.2(d) exactly
                row: Dict[str, Any] = {
                    "task_id": t_id,
                    "model_id": m_id,
                    "rep": rep,
                    "correct": is_correct,
                    **e_vec,
                    "cost_usd": cost,
                    "latency_s": latency,
                    "provider_version": f"{m_id}-v1",
                    "system_fingerprint": f"fp_{np.random.randint(1000, 9999)}",
                    "seed": seed
                }
                # Append the row to the outcomes list
                outcome_rows.append(row)

    # Convert the list of outcome dictionaries into a pandas DataFrame
    outcomes = pd.DataFrame(outcome_rows)
    
    # Enforce strict categorical dtypes for identifiers to optimize memory
    outcomes["task_id"] = outcomes["task_id"].astype("category")
    # Enforce strict categorical dtypes for model identifiers
    outcomes["model_id"] = outcomes["model_id"].astype("category")
    # Enforce strict int8 dtypes for binary flags
    outcomes["correct"] = outcomes["correct"].astype("int8")
    # Iterate over error columns to enforce int8
    for col in error_cols:
        # Cast each error column to int8
        outcomes[col] = outcomes[col].astype("int8")

    # Return the five generated data structures
    return model_answer, candidate_solution, task_record, outcomes, qualification

# Execute the synthetic data generation function
ma_raw, cs_raw, tr_raw, outcomes_df, qual_df = generate_synthetic_fmec_data()

# Generate a synthetic list of raw proxy texts for the contamination audit
raw_proxy_texts_list: List[str] = [fake.text() for _ in range(5)]
```

### **Step 2: Loading the Configuration (`config.yaml`)**

The study relies on a deterministic configuration file (`config.yaml`) that defines all hyperparameters, budgets, and reproducibility seeds. We read this file into memory and convert it into a usable Python dictionary.

**Methodology:**
1.  **File I/O:** Open `config.yaml` in read mode with explicit UTF-8 encoding.
2.  **Parsing:** Use `yaml.safe_load` to convert the YAML structure into a nested Python dictionary securely.
3.  **Validation:** Catch file existence errors and parsing errors, raising explicit exceptions to halt the pipeline if the configuration is corrupted or missing.

```python
def load_study_configuration(filepath: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the study configuration parameters from a YAML file into a Python dictionary.

    Purpose:
        To ingest the deterministic hyperparameters, optimization budgets, and
        reproducibility seeds defined in the external configuration file. This
        ensures strict separation of code and configuration.

    Inputs:
        filepath (str): The relative or absolute path to the YAML configuration file.
                        Default is "config.yaml".

    Processes:
        1.  File Access: Attempts to open the specified file in read mode.
        2.  Parsing: Uses PyYAML's safe_load to parse the YAML structure securely.
        3.  Validation: Catches and handles FileNotFoundError and YAMLError.

    Outputs:
        Dict[str, Any]: A nested dictionary containing the study configuration.

    Raises:
        TypeError: If filepath is not a string.
        FileNotFoundError: If the config.yaml file does not exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    # Validate that the provided filepath is a string
    if not isinstance(filepath, str):
        # Raise TypeError to prevent invalid file operations
        raise TypeError(f"filepath must be a string, got {type(filepath).__name__}.")

    try:
        # Open the file stream in read mode with UTF-8 encoding
        with open(filepath, "r", encoding="utf-8") as file:
            # Parse the YAML content safely into a Python dictionary
            config = yaml.safe_load(file)

        # Log success to the console for user visibility
        print(f"Successfully loaded configuration from {filepath}")

        # Return the parsed configuration dictionary
        return config

    except FileNotFoundError:
        # Raise an explicit error if the configuration file is missing
        raise FileNotFoundError(f"Critical Error: {filepath} not found in the working directory.")
    except yaml.YAMLError as e:
        # Handle YAML syntax errors explicitly
        print(f"Error parsing YAML file {filepath}: {e}")
        # Re-raise the exception to halt execution on corrupted config
        raise

# Load the configuration from the working directory into the 'config' dictionary
config = load_study_configuration("config.yaml")

# Fallback for the Jupyter Notebook environment if the file does not exist
if not config:
    print("Falling back to in-memory mock configuration for demonstration purposes.")
    # Initialize a minimal mock configuration required to pass validation
    config = {
        "gateway": {"base_url": "https://api.example.com", "api_key_env_var": "DUMMY_KEY"},
        "models": {"panel": ["model_A", "model_B", "model_C"], "judge": "model_A", "frontier_baselines": []},
        "llm_settings": {"panel": {"temperature": 0.2}, "solver": {"temperature": 0.1}, "judge": {"temperature": 0.0}, "max_retries": 2, "request_timeout_s": 60},
        "prompts": {"panel_system": "{task_prompt}", "judge_system": "{task_prompt}", "solver_system": "{task_prompt}", "judge_user_template": "{prompt}"},
        "schemas": {"model_answer": [], "candidate_solution": [], "task_record": [], "outcomes_columns": [], "error_taxonomy": []},
        "preprocessing": {"qualification_threshold_pct": 0.80, "n_capability_clusters": 4},
        "architecture": {"fusion_operator": "judge", "sandbox": {"timeout_s": 5, "network": "disabled", "fs": "restricted"}},
        "network_training": {"router": {"enabled": False}},
        "estimation": {
            "utility_weights": {"alpha": 1.0, "beta": 0.5, "gamma": 0.3, "delta": 0.2},
            "fmec_hyperparams": {"eta": 0.75, "lambda": 0.60},
            "monte_carlo": {"K": 64, "coalition_sampler": "permutation", "sampling_seed": 42},
            "information_gain": {"estimator": "outcome_based_cmi", "smoothing_alpha": 0.5},
            "covariance": {"estimator": "ledoit_wolf", "shrinkage_target": "diagonal", "block_by_error_category": True}
        },
        "optimization": {"C_max": 0.02, "L_max": 4.0, "R_max": 0.08},
        "uncertainty": {
            "replication": {"R": 5, "adjudication_subsample": 0.2, "R_judge": 5, "replicate_seed_base": 42},
            "bootstrap": {"B": 100, "ci_level": 0.95, "resample_levels": ["task", "replicate", "adjudication"], "bootstrap_seed": 13}
        },
        "stat_tests": {"alpha_level": 0.05, "delta_tau": 0.20, "rsp_floor": 0.90, "kappa_floor": 0.70, "negligible_d": 0.20, "negligible_cliffs_delta": 0.147},
        "fusion_sim_audit": {"audit_n_coalitions": 10},
        "contamination_audit": {"ngram_overlap_n": 13, "overlap_exclusion_rate": 0.10},
        "oracle_eval": {"max_pool_size_for_enumeration": 20},
        "reproducibility": {"seed_ledger": {"global_seed": 42, "monte_carlo_sampling_seed": 42, "replicate_seed_base": 42, "bootstrap_seed": 13, "router_seed": 42}}
    }
    # Set a dummy environment variable to satisfy the gateway credential check
    os.environ["DUMMY_KEY"] = "mock_secret_key"
```

### **Step 3: Executing the Pipeline (`run_experimental_suite`)**

With the raw data structures and configuration loaded into memory, we invoke the top-level global orchestrator. This function manages the entire lifecycle: data ingestion, contamination auditing, FMEC estimation, constrained selection, nested bootstrap validation, and cryptographic packaging.

**Methodology:**
1.  **Environment Setup:** Ensure a dummy API key is set in the environment to satisfy the gateway credential check during validation.
2.  **Function Call:** Pass the raw lists, DataFrames, and proxy texts to `run_experimental_suite`.
3.  **Output Handling:** The function returns a `Path` object pointing to the immutable `.tar.gz` archive containing all reproducible artifacts.

```python
# ==============================================================================
# Execution of the End-to-End FMEC Study Pipeline
# ==============================================================================

if __name__ == "__main__":
    # Set a dummy environment variable to satisfy the gateway credential check
    # In a real deployment, this would be a valid OpenRouter API key
    os.environ["OPENROUTER_API_KEY"] = "mock_secret_key_for_execution"

    # Ensure we have valid inputs before initiating the heavy computational pipeline
    if ma_raw and cs_raw and tr_raw and not qual_df.empty and config:
        
        # Log the initiation of the pipeline
        print("\nInitiating FMEC Experimental Suite...")
        
        try:
            # Execute the global orchestrator
            # This will trigger the full sequence of Tasks 1 through 24
            archive_path = run_experimental_suite(
                raw_config=config,
                model_answer_raw=ma_raw,
                candidate_solution_raw=cs_raw,
                task_record_raw=tr_raw,
                qualification_raw=qual_df,
                raw_proxy_texts=raw_proxy_texts_list
            )
            
            # ==============================================================================
            # Inspecting the Outputs
            # ==============================================================================
            
            # Print a visual separator for the completion block
            print("\n" + "="*80)
            # Announce successful completion
            print("STUDY EXECUTION COMPLETE")
            # Print a visual separator
            print("="*80)
            
            # Print the location of the final cryptographic archive
            print(f"\nReproducibility Package successfully generated at:")
            # Display the path
            print(f"-> {archive_path}")
            
            # Verify the archive exists on disk
            if os.path.exists(archive_path):
                # Calculate the file size in Megabytes
                file_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
                # Print the file size
                print(f"Archive Size: {file_size_mb:.2f} MB")
            
            # Provide instructions for reviewing the artifacts
            print("\nTo review the scientific findings, extract the archive and inspect:")
            # List the primary report
            print("1. scientific_report.md (Hypothesis verdicts and stability)")
            # List the point estimates
            print("2. fmec_estimates_final.json (Point estimates and BCa intervals)")
            # List the selected portfolio
            print("3. selected_ensemble.json (The optimal model portfolio)")
            
        except Exception as e:
            # Catch and display any fatal errors that occurred during pipeline execution
            print(f"\nPipeline Execution Failed: {str(e)}")
            
    else:
        # Handle the case where data generation or config loading failed
        print("Error: Missing synthetic data or configuration. Cannot proceed.")
```

### **Summary of the Execution Flow**

1.  **Data Ingestion & Validation:** The synthetic raw records and configuration are ingested, strictly typed, and validated against the schema contracts to prevent silent data corruption.
2.  **Contamination Audit:** The pipeline executes lexical (13-gram) and behavioral probes, applying a global drop-for-all rule to prune contaminated tasks and protect the causal interpretation of complementarity.
3.  **Qualification & Stratification:** Candidates are filtered via the $P_{\min}$ benchmark gate and clustered by capability profile to ensure lineage diversity during coalition sampling.
4.  **Tensor Assembly:** The canonical outcome tensor is constructed, mathematically enforcing the error exclusivity invariant ($\sum e_* = 1 - \text{correct}$) and computing exact USD costs.
5.  **Estimation:** The framework computes marginal utility (permutation Shapley), conditional information gain (outcome-based CMI with Miller-Madow correction), and correlated-failure risk (Ledoit-Wolf shrinkage).
6.  **Optimization:** A budget-feasible greedy algorithm selects the optimal model portfolio subject to strict cost, latency, and risk constraints.
7.  **Validation & Inference:** A three-level nested cluster bootstrap quantifies uncertainty (decomposing $\sigma^2_{\mathrm{task}}$, $\sigma^2_{\mathrm{gen}}$, and $\sigma^2_{\mathrm{adj}}$), followed by the execution of pre-registered hypothesis tests (H1-H5) with Holm-Bonferroni FWER control.
8.  **Robustness & Packaging:** The pipeline executes 10,000 Dirichlet preference sweeps to assess recommendation fragility, generates publication-grade figures, and serializes all artifacts into a cryptographically hashed, Tier-1 reproducible archive.

## Output Structure

The pipeline returns a comprehensive, cryptographically hashed `.tar.gz` archive containing the following key artifacts:
-   **`fmec_estimates_final.json`**: The central ranking primitive containing point estimates for MEC, IG, MRC, and FMEC, alongside BCa confidence intervals and bootstrap selection frequencies.
-   **`selected_ensemble.json`**: The optimal model portfolio ($S_{\mathrm{FMEC}}$) and its corresponding simplex weights.
-   **`scientific_report.md`**: A clinical, severity-ranked Markdown report detailing the verdicts for hypotheses H1-H5, effect sizes (Cohen's $d$, Cliff's $\delta$), and methodological disclosures.
-   **`variance_components.json`**: The decomposition of total uncertainty into $\sigma^2_{\mathrm{task}}$, $\sigma^2_{\mathrm{gen}}$, and $\sigma^2_{\mathrm{adj}}$.
-   **`robustness_report.json`**: The results of the 10,000 Dirichlet preference sweeps and the six pre-registered ablations (including LOO MEC).
-   **`covariance.csv`**: The Ledoit-Wolf shrunk error covariance matrix ($\widehat{\Sigma}$).
-   **`outcome_tensor.canonical.json`**: The exact byte-level representation of the canonical outcome tensor that was hashed for Tier-1 verification.
-   **Figures**: Publication-grade vector graphics including `efficient_frontier.pdf`, `fmec_ci_forest.pdf`, `covariance_heatmap.pdf`, and `rsp_bar.pdf`.

## Project Structure

```
fused_llm_router/
│
├── fused_llm_router_draft.ipynb                # Main implementation notebook containing all callables
├── config.yaml                                 # Master configuration file (Budgets, Hyperparameters, Seeds)
├── requirements.txt                            # Python package dependencies
│
├── fmec_run_20260618T120000Z/                  # Auto-generated, timestamped archival directory
│   ├── environment_provenance.json             # Dependency lockfile and platform fingerprint
│   ├── model_version_manifest.json             # Log of served provider versions and system fingerprints
│   ├── contamination_report.json               # Results of the lexical and behavioral audit
│   ├── outcome_tensor.csv                      # Human-readable canonical outcome tensor
│   ├── outcome_tensor.canonical.json           # Exact hashed byte representation
│   ├── tensor_sha256.txt                       # The published, verifiable tensor hash
│   ├── fmec_estimates_final.json               # Point estimates and BCa intervals
│   ├── selected_ensemble.json                  # The optimal model portfolio
│   ├── covariance.csv                          # Ledoit-Wolf shrunk error covariance matrix
│   ├── scientific_report.md                    # Hypothesis verdicts and stability status
│   ├── efficient_frontier.pdf                  # Vector graphics
│   ├── fmec_ci_forest.pdf
│   └── covariance_heatmap.pdf
│
├── LICENSE                                     # MIT Project License File
└── README.md                                   # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Researchers and engineers can modify study parameters such as:
-   **Utility Weights:** Adjust $\alpha, \beta, \gamma, \delta$ in the `utility_weights` block to shift the deployment preference between quality, risk, cost, and latency.
-   **Optimization Budgets:** Modify $C_{\max}, L_{\max}, R_{\max}$ in the `optimization` block to simulate different production deployment constraints.
-   **Estimator Selection:** Toggle between `outcome_based_cmi` and `entropy_reduction` for information gain, or adjust the Ledoit-Wolf shrinkage target in the `estimation` block.
-   **Uncertainty Quantification:** Adjust the number of bootstrap replications ($B$) or the generation replicates ($R$) in the `uncertainty` block to balance computational cost against statistical precision.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. 

**Strict Engineering Standards:** Adherence to PEP-8, strict type hinting (`typing` module), and the draconian requirement for line-by-line in-text comments that explain the mathematical or logical purpose of every single line of code is strictly required for all pull requests to maintain the implementation-grade standard of this repository.

## Recommended Extensions

Future extensions, building upon this foundational framework, could include:
-   **Dynamic Routing Policies:** Expanding the optional contextual bandit router to incorporate deep reinforcement learning for state-dependent, per-request routing.
-   **Cross-Lingual Ensembles:** Applying the FMEC framework to evaluate complementarity across models trained on different linguistic corpora, quantifying cross-lingual information gain.
-   **Continuous Drift Adaptation:** Implementing online covariance updating mechanisms to adapt to silent model drift without requiring full pipeline re-execution.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original concept paper:

```bibtex
@misc{chirinda2026fmec,
  title={Beyond Benchmark Rankings: Information-Aware and Risk-Aware Optimization of Large Language Model Ensembles (A Concept Paper and Methodology)},
  author={Chirinda, Craig S.},
  howpublished={Author's Concept Paper},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```bibtex
@misc{chirinda2026fusedllmrouter,
  author = {Chirinda, Craig S.},
  title = {Fused LLM Router: Information-Aware and Risk-Aware Optimization of Large Language Model Ensembles},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/chirindaopensource/fused_llm_router}}
}
```

## Acknowledgments

-   Credit to **Craig S. Chirinda** for the foundational theoretical framework, the rigorous econometric design, and the synthesis of cooperative game theory with portfolio optimization.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, particularly the **NumPy**, **Pandas**, **SciPy**, and **Scikit-Learn** contributors.


--

*This README was generated based on the structure and content of the `fused_llm_router_draft.ipynb` notebook and follows best practices for research software documentation.*
