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
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm


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
        self.verbose = False

    def reset(self, energy_capacity: float = None):
        """Reset policy for a new sample."""
        if energy_capacity is not None:
            self.initial_energy = energy_capacity
        self.energy = self.initial_energy
        self.energy_trace = []

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
        if self.verbose:
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
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
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
            'energy_trace': policy.energy_trace.copy(),
            'remaining_energy': policy.energy,
            'total_energy_used': policy.initial_energy - policy.energy,
        }

    def forward_fp32_only(self, input_text: str):
        """Run forward pass using fp32 precision for all layers (baseline)."""
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            t0 = time.perf_counter()
            hidden_state = self.embeddings(input_ids)
            attention_mask = attention_mask.to(torch.bool)
            
            for layer in self.layers:
                try:
                    hidden_state = layer(hidden_state, attn_mask=attention_mask)[0]
                except TypeError:
                    out = layer(hidden_state, attn_mask=attention_mask)
                    hidden_state = out[0] if isinstance(out, (tuple, list)) else out

            pooled = hidden_state[:, 0]
            pooled = self.pre_classifier(pooled)
            pooled = nn.ReLU()(pooled)
            logits = self.classifier(pooled)
            t1 = time.perf_counter()

        # Calculate energy cost (all layers use fp32)
        num_layers = len(self.layers)
        fp32_cost_per_layer = 10.0  # Match SimpleEnergyPolicy costs
        total_energy_used = num_layers * fp32_cost_per_layer

        return {
            'logits': logits.cpu(),
            'latency_sec': t1 - t0,
            'total_energy_used': total_energy_used,
        }


# -------------------------------
# Evaluation Functions
# -------------------------------

def evaluate_adaptive_strategy(harness: LayerwiseSwitcher, test_data: List[Tuple[str, int]], 
                                energy_capacity: float, random_energy: bool = True):
    """Evaluate adaptive layer-wise precision switching strategy."""
    policy = SimpleEnergyPolicy(energy_capacity=energy_capacity)
    policy.verbose = False
    
    predictions = []
    true_labels = []
    energy_costs = []
    latencies = []
    all_layer_choices = []
    precision_counts = {'fp32': 0, 'int8': 0, 'int4': 0}
    
    for text, label in tqdm(test_data, desc="Adaptive Strategy"):
        # Simulate random energy for each sample if enabled
        if random_energy:
            simulated_energy = random.uniform(energy_capacity * 0.3, energy_capacity)
            policy.reset(simulated_energy)
        else:
            policy.reset(energy_capacity)
        
        result = harness.forward_with_switching(text, policy)
        
        # Get prediction
        probs = torch.softmax(result['logits'], dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
        predictions.append(pred_idx)
        true_labels.append(label)
        energy_costs.append(result['total_energy_used'])
        latencies.append(sum(result['layer_times']))
        
        # Track precision choices
        for choice in result['layer_choices']:
            precision_counts[choice] += 1
        all_layer_choices.append(result['layer_choices'])
    
    accuracy = accuracy_score(true_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'avg_energy': np.mean(energy_costs),
        'avg_latency': np.mean(latencies),
        'precision_counts': precision_counts,
        'all_layer_choices': all_layer_choices,
        'energy_costs': energy_costs,
        'latencies': latencies,
    }

def evaluate_fp32_baseline(harness: LayerwiseSwitcher, test_data: List[Tuple[str, int]]):
    """Evaluate fp32-only baseline strategy."""
    predictions = []
    true_labels = []
    energy_costs = []
    latencies = []
    
    for text, label in tqdm(test_data, desc="FP32 Baseline"):
        result = harness.forward_fp32_only(text)
        
        # Get prediction
        probs = torch.softmax(result['logits'], dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
        predictions.append(pred_idx)
        true_labels.append(label)
        energy_costs.append(result['total_energy_used'])
        latencies.append(result['latency_sec'])
    
    accuracy = accuracy_score(true_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'avg_energy': np.mean(energy_costs),
        'avg_latency': np.mean(latencies),
        'energy_costs': energy_costs,
        'latencies': latencies,
    }

# -------------------------------
# Visualization
# -------------------------------

def visualize_policy(choices: List[str], energy_trace: List[float], initial_energy: float):
    # --- IEEE Styling (Local scope to avoid affecting global state if preferred) ---
    with plt.style.context("seaborn-v0_8-paper"):
        # Figure size: 3.5 inches wide (single column).
        # Height 3.5 inches allows for two stacked plots comfortably.
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(3.5, 3.5), sharex=True)
        
        # --- Plot 1: Energy Trace ---
        ax1.plot(range(len(energy_trace)), energy_trace, marker='o', 
                 markersize=4, linewidth=1.5, color='dodgerblue', clip_on=False)
        
        ax1.set_ylabel('Energy')
        ax1.set_ylim(0, initial_energy * 1.1)  # 10% padding for aesthetics
        ax1.grid(True, linestyle='--', alpha=0.5)
        # Note: Title removed to save space (use LaTeX caption instead)

        # --- Plot 2: Precision Choices ---
        precision_map = {'fp32': 3, 'int8': 2, 'int4': 1}
        # Add a dummy value at the end or use 'post'/'pre' steps carefully. 
        # Here we map directly to ensure length matches x-axis.
        precision_values = [precision_map[c] for c in choices]
        
        # 'where="mid"' places the step directly over the layer index tick
        ax2.step(range(len(choices)), precision_values, where='mid', 
                 linewidth=1.5, color='crimson')
        
        # Custom Y-ticks for categorical labels
        ax2.set_yticks([1, 2, 3])
        ax2.set_yticklabels(['int4', 'int8', 'fp32'])
        
        ax2.set_ylabel('Precision')
        ax2.set_xlabel('Layer Index') # Only set xlabel on the bottom plot
        ax2.grid(True, linestyle='--', alpha=0.5)

        # Align y-labels to ensure the left edge is straight
        fig.align_ylabels([ax1, ax2])

        plt.tight_layout(pad=0.5, h_pad=1.0) # h_pad controls space between top/bottom
        
        # Save as PDF for vector quality in LaTeX
        # plt.savefig("policy_viz.pdf", bbox_inches='tight')
        plt.show()

def visualize_comparison(adaptive_results: Dict, baseline_results: Dict, energy_capacity: float):
    """Create comprehensive comparison visualizations."""
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Accuracy Comparison
    ax1 = plt.subplot(2, 3, 1)
    strategies = ['Adaptive\nSwitching', 'FP32\nBaseline']
    accuracies = [adaptive_results['accuracy'], baseline_results['accuracy']]
    colors = ['steelblue', 'coral']
    bars = ax1.bar(strategies, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax1.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Energy Consumption Comparison
    ax2 = plt.subplot(2, 3, 2)
    avg_energies = [adaptive_results['avg_energy'], baseline_results['avg_energy']]
    bars = ax2.bar(strategies, avg_energies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Avg Energy per Sample', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.set_title('Energy Consumption', fontsize=12, fontweight='bold')
    for bar, energy in zip(bars, avg_energies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{energy:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Latency Comparison
    ax3 = plt.subplot(2, 3, 3)
    avg_latencies = [adaptive_results['avg_latency'] * 1000, baseline_results['avg_latency'] * 1000]  # Convert to ms
    bars = ax3.bar(strategies, avg_latencies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Avg Latency (ms)', fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax3.set_title('Inference Latency', fontsize=12, fontweight='bold')
    for bar, lat in zip(bars, avg_latencies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{lat:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Precision Distribution (Adaptive only)
    ax4 = plt.subplot(2, 3, 4)
    precisions = ['fp32', 'int8', 'int4']
    counts = [adaptive_results['precision_counts'][p] for p in precisions]
    total = sum(counts)
    percentages = [c / total * 100 if total > 0 else 0 for c in counts]
    bars = ax4.bar(precisions, percentages, color=['red', 'orange', 'green'], alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Usage (%)', fontsize=11)
    ax4.set_xlabel('Precision', fontsize=11)
    ax4.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax4.set_title('Precision Distribution\n(Adaptive Strategy)', fontsize=12, fontweight='bold')
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. Energy Distribution Histogram
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(adaptive_results['energy_costs'], bins=30, alpha=0.6, label='Adaptive', 
             color='steelblue', edgecolor='black', linewidth=0.5)
    ax5.axvline(baseline_results['avg_energy'], color='coral', linestyle='--', 
                linewidth=2, label='FP32 Baseline (avg)')
    ax5.set_xlabel('Energy per Sample', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax5.set_title('Energy Distribution', fontsize=12, fontweight='bold')
    
    # 6. Latency Distribution Histogram
    ax6 = plt.subplot(2, 3, 6)
    adaptive_lat_ms = [l * 1000 for l in adaptive_results['latencies']]
    baseline_lat_ms = [l * 1000 for l in baseline_results['latencies']]
    ax6.hist(adaptive_lat_ms, bins=30, alpha=0.6, label='Adaptive', 
             color='steelblue', edgecolor='black', linewidth=0.5)
    ax6.hist(baseline_lat_ms, bins=30, alpha=0.6, label='FP32 Baseline', 
             color='coral', edgecolor='black', linewidth=0.5)
    ax6.set_xlabel('Latency (ms)', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.legend()
    ax6.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax6.set_title('Latency Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout(pad=2.0)
    plt.savefig("phase-5/layerwise_comparison.png", dpi=300, bbox_inches='tight')
    print("\n💾 Comparison visualization saved to phase-5/layerwise_comparison.png")
    plt.show()


# -------------------------------
# Main Evaluation
# -------------------------------

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    energy_capacity = 40.0  # Energy capacity for adaptive strategy

    print("="*80)
    print("🚀 Layer-wise Precision Switching Evaluation")
    print("="*80)
    
    # 1. Load model
    print('\n📦 Loading base model...')
    base = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    harness = LayerwiseSwitcher(base, device=device)
    print(f"✅ Model loaded on {device}")

    # 2. Load test data
    print('\n📚 Loading IMDb Test Data...')
    try:
        dataset = load_dataset("imdb")
        test_subset = dataset['test'].shuffle(seed=1337).select(range(500))
        test_data = [(x['text'], x['label']) for x in test_subset]
        print(f"✅ Loaded {len(test_data)} test samples")
    except Exception as e:
        print(f"⚠️ Could not load IMDb ({e}). Using dummy data.")
        test_data = [
            ("This movie was fantastic and thrilling.", 1), 
            ("Terrible plot and bad acting.", 0)
        ] * 250

    # 3. Evaluate Adaptive Strategy
    print(f'\n🔄 Evaluating Adaptive Layer-wise Switching (Energy Capacity: {energy_capacity})...')
    adaptive_results = evaluate_adaptive_strategy(harness, test_data, energy_capacity, random_energy=True)

    # 4. Evaluate FP32 Baseline
    print('\n🔄 Evaluating FP32 Baseline...')
    baseline_results = evaluate_fp32_baseline(harness, test_data)

    # 5. Print Results
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    print(f"\n{'Metric':<30} {'Adaptive':<20} {'FP32 Baseline':<20}")
    print("-" * 80)
    print(f"{'Accuracy':<30} {adaptive_results['accuracy']:<20.4f} {baseline_results['accuracy']:<20.4f}")
    print(f"{'Avg Energy per Sample':<30} {adaptive_results['avg_energy']:<20.2f} {baseline_results['avg_energy']:<20.2f}")
    print(f"{'Avg Latency (ms)':<30} {adaptive_results['avg_latency']*1000:<20.2f} {baseline_results['avg_latency']*1000:<20.2f}")
    
    # Energy savings
    energy_savings = (1 - adaptive_results['avg_energy'] / baseline_results['avg_energy']) * 100
    print(f"\n💡 Energy Savings: {energy_savings:.2f}%")
    
    # Precision distribution
    print(f"\n🎯 Precision Distribution (Adaptive Strategy):")
    total_precision_ops = sum(adaptive_results['precision_counts'].values())
    for prec, count in adaptive_results['precision_counts'].items():
        pct = (count / total_precision_ops * 100) if total_precision_ops > 0 else 0
        print(f"   - {prec:<6}: {count:6d} operations ({pct:5.1f}%)")
    
    print("="*80)

    # 6. Visualize Results
    print('\n📈 Generating comparison visualizations...')
    visualize_comparison(adaptive_results, baseline_results, energy_capacity)
    
    # 7. Optional: Show example single-sample visualization
    print('\n📊 Generating example single-sample policy visualization...')
    example_policy = SimpleEnergyPolicy(energy_capacity=energy_capacity)
    example_policy.verbose = False
    example_text, _ = test_data[0]
    example_result = harness.forward_with_switching(example_text, example_policy)
    visualize_policy(example_result['layer_choices'], example_result['energy_trace'], 
                     example_policy.initial_energy)
