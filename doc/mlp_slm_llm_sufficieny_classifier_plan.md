# Sufficiency Classifier Ablation Analysis: LLM vs MLP

The goal of this analysis is twofold:
1. **Interactive Run Behavior (LLM vs MLP):** Quantify and visualize the performance and behavioral differences between the MLP-based sufficiency classifier (old run) and the LLM-based sufficiency classifier (new run, which also incorporates Title and Abstract). Our main hypothesis is that the LLM classifier is **more optimistic** in its assessment of evidence, often determining sufficiency at the **first iteration**, leading to **faster convergence** (fewer search iterations). Note that this earlier termination is a behavioral difference and does not necessarily guarantee better downstream performance, as it may end the search process prematurely.
2. **Static Score Distributions (MLP vs SLM vs LLM):** Compare the distribution of sufficiency scores produced by the MLP, SLM (Qwen 3.5-9B), and LLM (Claude Haiku). Since the scatter plot revealed noticeable differences between the SLM and LLM, we will use violin plots to clearly illustrate the structural differences (e.g., score spread, optimism) among the three classifiers using the static ablation test results.

## Proposed Analysis and Metrics

We will extract the `evidence_state.json` data from both results directories (MLP: `signor_eval_20260330_205939` and LLM: `signor_eval_20260402_214154`). From this data, we will compute and visualize the following key statistical differences:

1. **Mean Iterations to Convergence:**
   - Compare the average number of search iterations required by the LLM vs MLP.
   - Calculate the percentage of claims that resolve early at **Iteration 1**.
2. **Sufficiency Score Trajectory:**
   - Track the average sufficiency score/confidence sequentially at Iteration 1, 2, 3, and 4.
   - We expect the LLM to start with a significantly higher score at Iteration 1 compared to the MLP.
3. **Paired Difference Analysis:**
   - On a per-claim basis, compute the difference in iterations. Evaluate how many individual claims had earlier termination under the LLM.

## Proposed Visualizations (NeurIPS Figures)

To best represent these findings in the paper, we will generate the following figures using `matplotlib` and `seaborn`:

### 1. Stopping Iteration Distribution (Bar Chart)
- **X-axis:** Search Iteration (1, 2, 3, 4)
- **Y-axis:** Percentage or count of claims that stopped at this iteration.
- **Series:** Two bars per iteration (one for MLP, one for LLM).

### 2. Sufficiency Score Evolution (Line Plot)
- **X-axis:** Search Iteration.
- **Y-axis:** Average Sufficiency Score.
- **Series:** Two lines with confidence intervals (MLP vs LLM). Shows how the LLM extracts higher sufficiency from the same initial evidence pool.

### 3. Iteration Difference Profile (Violin or Strip Plot)
- **Y-axis:** $\Delta$ Iterations (MLP iterations - LLM iterations)
- **Description:** Points $>0$ represent claims where LLM converged faster than MLP. We will plot all claims together to see the overall optimistic behavior.

### 4. Classifier Score Comparisons (Scatter & Violin Plots)
- **Format:** We will generate 4 distinct figures to analyze static sufficiency scores:
  1. **Scatter Plot:** SLM vs LLM
  2. **Scatter Plot:** SLM vs MLP
  3. **Scatter Plot:** LLM vs MLP
  4. **Combined Violin Plot:** MLP, SLM, LLM side-by-side
- **Scatter Plot Details:** Do not color or delineate by flip status (`flip_True` vs `flip_False`). Do not annotate the data points with claim indices to avoid visual clutter.
- **Purpose:** To visually and statistically prove the similarities and structural distribution differences across the three models. The scatter plots will reveal direct pairwise agreements, while the combined violin plot will highlight the dispersion and general optimism levels between the three classifiers. Data for this will be sourced from `results/slm_llm_ablation/`.

## Proposed Changes

We will create a specific script for extracting, merging, and plotting these details.

### Scripts: `scripts/sufficiency_classifier/slm_llm_ablation/`

#### [NEW] [compare_eval_runs.py](file:///hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct/scripts/sufficiency_classifier/slm_llm_ablation/compare_eval_runs.py)
- A script to traverse the two specified `signor_eval` directories for the LLM vs MLP dynamic behavior.
- For each claim and condition (`flip_False` and `flip_True`), find the final `iteration` and the array of confidences in `sufficiency_history` from `evidence_state.json`.
- Aggregate into a `pandas.DataFrame`, treating `flip_True` and `flip_False` combinations as distinct claims.
- Run statistical tests (e.g., paired t-test for iterations).
- Generate and save the dynamic charts (Charts 1, 2, 3) described above as high-res PNG/PDF files into `/results/slm_llm_ablation/signor_eval_ablation_plots/`.

#### [EXISTING] `visualize_sufficiency_scores.py` Updates
- Repurpose or wrap the existing plotting functions in `visualize_sufficiency_scores.py` to extract the individual score distributions for the violin plot and the arrays for the three pairwise scatter plots.
- Remove the code that annotates each scatter point with its SIGNOR ID index and removes hue grouping based on `flip = True/False`. 
- Ensure all four static ablation figures match the generic formatting of the NeurIPS paper.

## Verification Plan

### Automated Tests
- Run the extraction script over the two directories.
- Verify we successfully process exactly the same number of matching claims between the two runs.

### Manual Verification
- Output the Mean Iteration and standard deviation to the terminal so we can quickly sanity check the numbers before examining the plots.
- The user will inspect the generated plots (e.g., `iterations_distribution.png`, `sufficiency_trajectory.png`) for aesthetic and scientific suitability.

## Walkthrough Results

This section summarizes the execution of the requested ablation study comparing the dynamically evaluated LLM and MLP sufficiency classifiers, as well as the static agreement verification between SLM (Qwen 3.5-9B) and LLM (Claude Haiku).

### 1. Dynamic Behavior: LLM vs MLP
The LLM and MLP dynamic evaluation runs were parsed from their respective `evidence_state.json`. We observed the following metric summaries for **101 claims** commonly processed by both models:

- **Mean Iterations:** 
  - MLP: `3.010` (Std: `0.975`)
  - LLM: `2.277` (Std: `1.087`)
- **Resolved at Iteration 1:**
  - MLP: `9.9%`
  - LLM: `29.7%`

A paired t-test confirmed the statistical significance of earlier termination in the LLM run (`t = 5.060, p = 1.91e-06`).

> [!NOTE]  
> The LLM classifier terminates earlier by often assigning a much higher sufficiency score at Iteration 1.

**Figures generated:**
- `../results/slm_llm_ablation/signor_eval_ablation_plots/iterations_distribution.png`
- `../results/slm_llm_ablation/signor_eval_ablation_plots/sufficiency_trajectory.png`
- `../results/slm_llm_ablation/signor_eval_ablation_plots/iteration_difference_profile.png`

### 2. Static Distributions: MLP vs SLM vs LLM
Side-by-side violin plots were generated to compare the static sufficiency scores assigned to evidence across the dataset between MLP, Qwen 3.5-9B, and Claude Haiku. 

> [!TIP]  
> The visualization features formatting appropriate for the NeurIPS paper and highlights the differences in score dispersion among the classifiers.

**Figure generated:**
- `../results/slm_llm_ablation/sufficiency_comparison_violin.png`

All required scripts are updated, successfully verified, and executable using `uv run python`.
