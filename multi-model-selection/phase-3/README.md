# Phase 3: Model Quantization

This phase applies 8-bit weight-only quantization to reduce model size and accelerate inference speed. It contains two scripts to quantize both the original baseline model and the pruned model from Phase 2.

---

### ## Purpose

The scripts in this folder use the `torchao` library to convert the model's `float32` weights into `int8`, a more efficient data type. This process is applied to two different models to compare the effects of quantization alone versus a combined pruning-and-quantization approach.

1.  **`quantize_baseline.py`**: Applies quantization to the original model from Phase 1.
2.  **`quantize_pruned.py`**: Applies quantization to the smaller, pruned model from Phase 2.

---

### ## How to Run

Ensure that **Phase 1 and Phase 2 have been run successfully**. Then, execute the scripts from the project's root directory:

```bash
# To quantize the original baseline model
python phase-3/quantize_baseline.py

# To quantize the pruned model
python phase-3/quantize_pruned.py
```

---

### ## Dependencies

* `torch`
* `torchao`
* `transformers`
* `datasets`
* `evaluate`
* `numpy`

---

### ## Inputs

* **`quantize_baseline.py`** requires the model from `outputs/phase-1/baseline_model/`.
* **`quantize_pruned.py`** requires both the baseline tokenizer from `outputs/phase-1/` and the pruned model state dict from `outputs/phase-2/pruned_true_reduction.pt`.

---

### ## Outputs

This script will generate an `outputs/phase-3` directory containing:

* **`quantized_baseline_weight_only_int8.pt`**: The `state_dict` for the quantized baseline model.
* **`quantized_weight_only_int8.pt`**: The `state_dict` for the quantized pruned model.
* **`torchao_quantization_baseline_results.json`**: A report comparing the baseline model before and after quantization.
* **`torchao_quantization_results.json`**: A report comparing the pruned model before and after quantization.