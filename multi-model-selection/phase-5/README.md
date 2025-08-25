# Energy-Aware Dynamic Model Selector

## Overview

This Python script provides a practical implementation of the **Energy-Aware Dynamic Neural Inference** concept, inspired by the research paper "Energy-Aware Dynamic Neural Inference" (arXiv:2411.02471).

The core idea is to simulate an intelligent system, like a small IoT device with a solar panel, that must perform machine learning tasks. The device has a limited-capacity battery and a variable energy supply. Instead of using a single ML model, it has access to several versions of the same model, each with a different trade-off between **accuracy** and **energy cost**:

* **Baseline:** Highest accuracy, highest energy cost.
* **Pruned:** Slightly lower accuracy, medium energy cost.
* **Quantized:** Good accuracy, low energy cost.
* **Pruned + Quantized:** Lowest accuracy, lowest energy cost.

The script implements a **Multi-Model Selection (MMS)** strategy. Before each task, the system checks its battery level and dynamically chooses the best (most accurate) model it can afford to run. This allows it to maximize performance when energy is abundant and gracefully degrade to a more efficient model when energy is scarce, ensuring continuous operation.

---

## How It Works

The main logic is encapsulated within the `EnergyAwareModelSelector` class.

### 1. Initialization (`__init__`)

When an instance of the class is created, it sets up the entire environment:

* **Models:** It prepares to load the four different model variations into memory.
* **Energy Simulation:** It initializes a virtual battery with a `max_energy_capacity` and a starting `current_energy` level.
* **Energy Costs:** It defines a dictionary (`model_energy_costs`) that assigns a specific energy cost to an inference run for each model type. These are illustrative values.
* **Model Preference:** It establishes a clear hierarchy (`model_preference`) from the most desirable model (`baseline`) to the most efficient (`pruned_quantized`). The system will always try to use the highest-ranking model in this list that it can afford.

```python
# From the __init__ method
self.model_energy_costs = {
    'baseline': 15.0,
    'quantized_baseline': 10.0,
    'pruned': 8.0,
    'pruned_quantized': 5.0,
}
self.model_preference = ['baseline', 'quantized_baseline', 'pruned', 'pruned_quantized']
```

### 2. The Inference Cycle (`select_and_infer`)

This is the heart of the script. When a new task (e.g., classifying a sentence) arrives, it executes a three-step process:

#### **Step 1: Harvest Energy**

The system first updates its battery level by adding any `harvested_energy` from its environment. The battery level cannot exceed its maximum capacity.

```python
self.current_energy = min(self.current_energy + harvested_energy, self.max_energy_capacity)
```

#### **Step 2: Select the Best Affordable Model**

This is the key decision-making step. The code iterates through the `model_preference` list. For each model, it checks if the `current_energy` is greater than or equal to the model's `energy_cost`. The **first** model that meets this condition is selected, and the search stops.

```python
model_choice = None
for model_name in self.model_preference:
    if model_name in self.models and self.current_energy >= self.model_energy_costs[model_name]:
        model_choice = model_name
        break # Found the best affordable model
```

If the battery doesn't have enough energy for even the cheapest model (`pruned_quantized`), `model_choice` remains `None`, and the inference task is **skipped** for that cycle to conserve energy.

#### **Step 3: Perform Inference and Consume Energy ⚡**

If a model was selected:

1.  The corresponding model is retrieved.
2.  The inference is performed, and the time taken (`latency`) is measured.
3.  The model's `energy_cost` is subtracted from the `current_energy`.
4.  The results, including the prediction, the model used, and the updated energy state, are returned.

---

## How to Run the Simulation

The `if __name__ == "__main__":` block at the end of the file runs a demonstration.

1.  **Controller:** An instance of the `EnergyAwareModelSelector` is created with a 50-unit battery capacity and 12 units of initial energy.
2.  **Simulation Loop:** The script simulates a series of events. In each step, a different amount of energy is "harvested." The `select_and_infer` method is called, and the script prints out which model was chosen based on the available energy.
