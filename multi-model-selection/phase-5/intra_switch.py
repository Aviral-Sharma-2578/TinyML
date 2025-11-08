"""
Prototype: Layer-wise precision switching for DistilBERT (with visualization)
-----------------------------------------------------------------------------

Adds a visualization section to show:
- Layer-wise chosen precision (fp32, int8, int4)
- Remaining energy after each layer

This helps visualize how energy-aware adaptive precision switching behaves dynamically.
"""

import copy
import time
from typing import Dict, List

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


# -------------------------------
# Utilities: fake quantize / dequantize
# -------------------------------

def uniform_quantize_dequantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Simulate symmetric uniform quantization (dequantized back to float32)."""
    if bits >= 32:
        return tensor.clone()
    with torch.no_grad():
        max_val = tensor.abs().max()
        if max_val == 0:
            return tensor.clone()
        q_levels = 2 ** (bits - 1) - 1
        scaled = tensor / (max_val + 1e-12) * q_levels
        q = scaled.round().clamp(-q_levels, q_levels)
        deq = q / q_levels * max_val
        return deq


def quantize_linear_weights_inplace(linear: nn.Linear, bits: int):
    """Simulate quantized linear weights (int8/int4) in-place."""
    linear.weight.data = uniform_quantize_dequantize(linear.weight.data, bits)
    if linear.bias is not None:
        linear.bias.data = uniform_quantize_dequantize(linear.bias.data, bits)


# -------------------------------
# Build per-layer precision variants
# -------------------------------

def build_layer_variants(layer_module: nn.Module) -> Dict[str, nn.Module]:
    """Create fp32, int8, int4 variants of a transformer layer."""
    variants = {}
    variants['fp32'] = layer_module

    int8_mod = copy.deepcopy(layer_module)
    for sub in int8_mod.modules():
        if isinstance(sub, nn.Linear):
            quantize_linear_weights_inplace(sub, bits=8)
    variants['int8'] = int8_mod

    int4_mod = copy.deepcopy(layer_module)
    for sub in int4_mod.modules():
        if isinstance(sub, nn.Linear):
            quantize_linear_weights_inplace(sub, bits=4)
    variants['int4'] = int4_mod

    return variants


# -------------------------------
# Controller / Policy
# -------------------------------

class SimpleEnergyPolicy:
    """Toy policy that chooses precision based on available energy."""
    def __init__(self, energy_capacity: float):
        self.energy = energy_capacity
        self.initial_energy = energy_capacity
        self.costs = {'fp32': 10.0, 'int8': 6.0, 'int4': 3.0}
        self.energy_trace: List[float] = []

    def choose_precision(self, layer_idx: int) -> str:
        safety = 1.0
        if self.energy >= self.costs['fp32'] + safety:
            chosen = 'fp32'
        elif self.energy >= self.costs['int8'] + safety:
            chosen = 'int8'
        else:
            chosen = 'int4'

        self.energy = max(0.0, self.energy - self.costs[chosen])
        self.energy_trace.append(self.energy)
        print(f"Layer {layer_idx}: chose {chosen}, remaining energy = {self.energy:.2f}")
        return chosen

    def add_energy(self, amount: float):
        self.energy += amount


# -------------------------------
# Layer-wise execution harness
# -------------------------------

class LayerwiseSwitcher:
    """Runs DistilBERT with adaptive precision switching per layer."""
    def __init__(self, base_model: DistilBertForSequenceClassification, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        self.model = base_model.to(self.device)
        self.model.eval()

        self.layers = list(self.model.distilbert.transformer.layer)
        self.layer_variants = [build_layer_variants(layer) for layer in self.layers]

        self.embeddings = self.model.distilbert.embeddings
        self.pre_classifier = self.model.pre_classifier
        self.classifier = self.model.classifier

    def forward_with_switching(self, input_text: str, policy: SimpleEnergyPolicy):
        """Run forward pass with adaptive per-layer precision."""
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            hidden_state = self.embeddings(input_ids)
            layer_choices, layer_times = [], []

            attention_mask = attention_mask.to(torch.bool)
            for idx, variants in enumerate(self.layer_variants):
                choice = policy.choose_precision(idx)
                chosen_layer = variants[choice].to(self.device)

                t0 = time.perf_counter()
                try:
                    hidden_state = chosen_layer(hidden_state, attn_mask=attention_mask)[0]
                except TypeError:
                    out = chosen_layer(hidden_state, attn_mask=attention_mask)
                    hidden_state = out[0] if isinstance(out, (tuple, list)) else out
                t1 = time.perf_counter()

                layer_choices.append(choice)
                layer_times.append(t1 - t0)

            pooled = hidden_state[:, 0]
            pooled = self.pre_classifier(pooled)
            pooled = nn.ReLU()(pooled)
            logits = self.classifier(pooled)

        return {
            'logits': logits.cpu(),
            'layer_choices': layer_choices,
            'layer_times': layer_times,
            'energy_trace': policy.energy_trace,
            'remaining_energy': policy.energy,
        }


# -------------------------------
# Visualization
# -------------------------------

def visualize_policy(choices: List[str], energy_trace: List[float], initial_energy: float):
    plt.figure(figsize=(10, 5))

    # 1. Energy over layers
    plt.subplot(1, 2, 1)
    plt.plot(range(len(energy_trace)), energy_trace, marker='o')
    plt.title('Energy Remaining per Layer')
    plt.xlabel('Layer Index')
    plt.ylabel('Energy')
    plt.ylim(0, initial_energy + 5)
    plt.grid(True)

    # 2. Precision choices
    precision_map = {'fp32': 3, 'int8': 2, 'int4': 1}
    precision_values = [precision_map[c] for c in choices]
    plt.subplot(1, 2, 2)
    plt.step(range(len(choices)), precision_values, where='mid', linewidth=2)
    plt.yticks([1, 2, 3], ['int4', 'int8', 'fp32'])
    plt.title('Chosen Precision per Layer')
    plt.xlabel('Layer Index')
    plt.ylabel('Precision Level')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# -------------------------------
# Demo run
# -------------------------------

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading base model...')
    base = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    harness = LayerwiseSwitcher(base, device=device)

    # Try different capacities: e.g., 35.0, 60.0, 80.0
    policy = SimpleEnergyPolicy(energy_capacity=40.0)

    sentence = "The movie had great cinematography and a moving story, but pacing could've been better."

    print('\nRunning layer-wise adaptive forward...')
    out = harness.forward_with_switching(sentence, policy)

    probs = torch.softmax(out['logits'], dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()

    print('\nLayer choices (per-layer precision):', out['layer_choices'])
    print('Layer timings (seconds):', ['{:.4f}'.format(t) for t in out['layer_times']])
    print('Remaining energy:', out['remaining_energy'])
    print('Prediction probs:', probs.numpy())

    # visualize behavior
    visualize_policy(out['layer_choices'], out['energy_trace'], policy.initial_energy)
