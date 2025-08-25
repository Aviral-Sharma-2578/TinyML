# Phase 2: Iterative Model Pruning

This script takes the baseline model from Phase 1 and applies iterative structured pruning to reduce its size and parameter count while attempting to preserve accuracy.

---

### ## Purpose

The goal of `prune_baseline.py` is to create a smaller, more efficient model by physically removing the least important neural network channels. The script iteratively prunes and fine-tunes the model to recover performance.

---

### ## How to Run

First, ensure that **Phase 1 has been run successfully**. Then, execute this script from the project's root directory:

```bash
python phase-2/prune_baseline.py
```

---

### ## Dependencies

* `torch`
* `transformers`
* `datasets`
* `evaluate`
* `numpy`

---

### ## Inputs

* **`outputs/phase-1/baseline_model/`**: The script requires the saved baseline model and tokenizer from Phase 1 to begin the pruning process.

---

### ## Outputs

This script will generate an `outputs/phase-2` directory containing:

* **`pruned_true_reduction.pt`**: The saved `state_dict` of the final, structurally smaller model. Note that this model has a different architecture than the baseline.
* **`true_reduction_pruning_metrics.json`**: A JSON file containing a detailed report of the pruning process, including step-by-step changes in accuracy, parameter count, and final compression ratio.
* **`ft_step*/`**: Checkpoint directories from the fine-tuning steps.