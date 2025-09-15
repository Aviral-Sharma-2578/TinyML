# Adaptive Intra-Inference Execution for Efficient NLP

This repository contains a prototype exploring various strategies for **dynamic, energy-aware neural inference**. The core idea is to move beyond using a single, static model and instead adapt the computational path in real-time to balance the trade-off between **accuracy** and **resource consumption** (e.g., energy, latency).

This is particularly relevant for deploying complex models like transformers on resource-constrained edge devices.

## The Core Concept: A Spectrum of Switching Granularity

The project investigates dynamic inference strategies across three distinct levels of granularity, from making one decision before inference to making continuous decisions during inference.

---
## Implemented Strategies

### 1. Coarse-Grained (Inter-Model Selection)
This approach treats each model as an atomic unit. A high-level controller selects one entire model from a pool to perform the full inference task.

* **Mechanism**: A "gating network" makes a decision *prior* to inference based on the current system state.
* **Policies Implemented**:
    * **Energy-Aware**: Selects the most accurate model that the current energy budget can afford.
    * **Complexity-Aware**: Uses fuzzy logic rules to select a model based on the input's estimated complexity (e.g., LOW, MEDIUM, HIGH).

### 2. Medium-Grained (Early-Exit Architecture)
This strategy uses a single, unified network with multiple opportunities to exit during inference, saving computation on "easy" inputs.

* **Mechanism**: A single backbone model is augmented with lightweight classifier heads at intermediate layers (`BranchyNet`). If a prediction's **confidence threshold** is met at an early exit, the result is returned immediately, skipping the deeper layers.

### 3. Fine-Grained (Intra-Inference Adaptation)
This offers the highest level of control by adapting the computational strategy *during* a single forward pass. This allows the system to react to resource changes in real-time.

* **Conceptual Models Explored**:
    * **Segment-level Switching**: The model is partitioned into segments (e.g., blocks of layers). A policy controller selects a pre-stored precision variant (e.g., INT4, INT8) for the *next segment* based on the real-time energy budget.
    * **Resource-Driven MoE Routing**: The Mixture-of-Experts gating concept is adapted to be resource-driven, selecting an "expert" variant to execute the next computational step.
    * **Progressive Refinement**: A full, low-cost inference pass is performed first. If confidence is low and resources permit, critical parts of the network are re-executed with a higher-fidelity model.

---
## File Descriptions 

This repository contains prototypes for each of the strategies discussed:

* `mms_controller.py`: Implements the **coarse-grained** greedy, energy-based Multi-Model Selection (MMS) strategy.
* `MoE.py`: Reframes the MMS controller using Mixture-of-Experts (MoE) terminology, where the models are "experts" and the selection logic is a "gating network".
* `fuzzy_mms_controller.py`: A **coarse-grained** controller that uses fuzzy logic based on task complexity instead of energy.
* `early_exit.py`: A **medium-grained** implementation of a BranchyNet with a confidence-based early-exit policy, demonstrated on the MNIST dataset.
* `intra_switch.py`: A research prototype for **fine-grained**, layer-wise precision switching in a DistilBERT model.

---
## Results & Trade-offs 

Model compression techniques like pruning and quantization create a spectrum of models with different performance characteristics. The goal of this project's dynamic selectors is to intelligently navigate these trade-offs in real-time.
