# Phase 1: Baseline Model Training 🎯

This script establishes the performance benchmark for the entire project. It fine-tunes a standard `distilbert-base-uncased` model on the GLUE SST-2 dataset and records its key performance metrics.

---

### ## Purpose

The goal of `baselines.py` is to create a foundational model against which all subsequent optimizations (pruning, quantization) will be compared. It measures the model's initial accuracy, size, and inference speed.

---

### ## How to Run

Ensure you have the required dependencies installed and then run the script from the root directory of the project:

```bash
python phase-1/baselines.py
```

---

### ## Dependencies

* `torch`
* `transformers`
* `datasets`
* `evaluate`
* `psutil`

---

### ## Inputs

* This script is self-contained and downloads the `distilbert-base-uncased` model and the `glue/sst2` dataset from the Hugging Face Hub.

---

### ## Outputs

This script will generate an `outputs/phase-1` directory containing:

* **`baseline_model/`**: A directory containing the fine-tuned model, tokenizer, and training configuration. This is the primary input for Phase 2 and Phase 3.
* **`baseline_metrics.json`**: A JSON file detailing the performance of the baseline model, including accuracy, parameter count, model size (MB), and average inference latency.