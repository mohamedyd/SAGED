#!/bin/bash
set -e

# Define parameters
RUNS=10

# Command 1
echo "Executing the labeling budget experiment ..."
nohup python3 scripts/exp_labeling_budget.py --runs $RUNS --verbose > nohup_labeling_budget.txt || true
echo "Done."

# Command 2
echo "Executing the detection accuracy experiment ..."
nohup python3 scripts/exp_detection_accuracy.py --runs $RUNS --verbose > nohup_detection_accuracy.txt || true
echo "Done."

# Command 3
echo "Executing the modeling accuracy experiment ..."
nohup python3 scripts/exp_modeling_accuracy.py --runs $RUNS --verbose > nohup_modeling_accuracy.txt || true
echo "Done."

# Command 4
echo "Executing the scalability experiment ..."
nohup python3 scripts/exp_scalability.py --runs $RUNS --verbose > nohup_scalability.txt || true
echo "Done."

# Command 5
echo "Executing the robustness (error rate) experiment ..."
nohup python3 scripts/exp_robustness.py --runs $RUNS --evaluate-error-rate --verbose > nohup_robustness_error_rate.txt || true
echo "Done."

# Command 6
echo "Executing the robustness (outlier degree) experiment ..."
nohup python3 scripts/exp_robustness.py --runs $RUNS --verbose > nohup_robustness_outlier_degree.txt || true
echo "Done."

