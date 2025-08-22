# prune_baseline.py - TRUE PARAMETER REMOVAL VERSION
import os
import time
import json
import psutil
import torch
import torch.nn as nn
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from copy import deepcopy

# ------------------ PATHS ------------------ #
BASE_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-2a"))
os.makedirs(BASE_OUTPUT, exist_ok=True)
BASELINE_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-1", "baseline_model"))

# ------------------ LOAD DATA ------------------ #
print("📊 Loading dataset...")
dataset = load_dataset("glue", "sst2")
tokenizer = DistilBertTokenizerFast.from_pretrained(BASELINE_MODEL)

def tokenize_fn(batch):
    return tokenizer(
        batch["sentence"], 
        truncation=True, 
        max_length=512
    )

# Tokenize and clean the dataset
encoded_dataset = dataset.map(
    tokenize_fn, 
    batched=True,
    remove_columns=["sentence", "idx"]  # Remove original text columns, keep labels
)

# Set format for PyTorch
encoded_dataset.set_format("torch")

# ------------------ LOAD BASELINE MODEL ------------------ #
print("🔄 Loading baseline model...")
model = DistilBertForSequenceClassification.from_pretrained(BASELINE_MODEL)
original_param_count = sum(p.numel() for p in model.parameters())
print(f"📈 Original parameters: {original_param_count:,}")

# ------------------ STRUCTURED PRUNING FUNCTIONS ------------------ #

def get_neuron_importance_scores(layer, method='l1'):
    """Calculate importance scores for neurons in a layer"""
    if method == 'l1':
        # L1 norm of outgoing weights (columns for input features, rows for output neurons)
        return torch.sum(torch.abs(layer.weight), dim=1)  # Sum across input features
    elif method == 'l2':
        return torch.sum(layer.weight ** 2, dim=1)
    else:
        raise ValueError(f"Unknown method: {method}")

def prune_linear_layer(layer, prune_ratio=0.3):
    """
    Prune a linear layer by removing entire neurons
    Returns new layer and mapping for connecting layers
    """
    if not isinstance(layer, nn.Linear):
        return layer, None
    
    # Get neuron importance scores
    importance_scores = get_neuron_importance_scores(layer, method='l1')
    
    # Determine how many neurons to keep
    total_neurons = layer.out_features
    neurons_to_keep = int(total_neurons * (1 - prune_ratio))
    
    if neurons_to_keep <= 0:
        neurons_to_keep = 1  # Keep at least one neuron
    
    # Get indices of neurons to keep (highest importance)
    _, keep_indices = torch.topk(importance_scores, neurons_to_keep)
    keep_indices = keep_indices.sort()[0]  # Sort for consistency
    
    # Create new layer with reduced size
    new_layer = nn.Linear(layer.in_features, neurons_to_keep, bias=(layer.bias is not None))
    
    # Copy weights and biases for kept neurons
    with torch.no_grad():
        new_layer.weight.data = layer.weight.data[keep_indices, :]
        if layer.bias is not None and new_layer.bias is not None:
            new_layer.bias.data = layer.bias.data[keep_indices]
    
    return new_layer, keep_indices

def adjust_next_layer(next_layer, keep_indices):
    """Adjust the next layer to match the pruned previous layer"""
    if not isinstance(next_layer, nn.Linear):
        return next_layer
    
    # Create new layer with reduced input features
    new_layer = nn.Linear(len(keep_indices), next_layer.out_features, bias=(next_layer.bias is not None))
    
    # Copy weights for kept input features
    with torch.no_grad():
        new_layer.weight.data = next_layer.weight.data[:, keep_indices]
        if next_layer.bias is not None and new_layer.bias is not None:
            new_layer.bias.data = next_layer.bias.data.clone()
    
    return new_layer

def prune_distilbert_model(model, prune_ratio=0.3):
    """
    Prune DistilBERT model by removing neurons from feedforward layers
    This is a simplified version - full implementation would need to handle all layer types
    """
    print(f"🔧 Applying structured pruning with ratio {prune_ratio}")
    
    # Keep track of modifications
    modifications = []
    
    # Get the transformer layers
    transformer_layers = model.distilbert.transformer.layer
    
    for layer_idx, transformer_layer in enumerate(transformer_layers):
        # Prune the feedforward network in each transformer layer
        ffn = transformer_layer.ffn
        
        # Prune the intermediate layer (lin1)
        if hasattr(ffn, 'lin1'):
            original_neurons = ffn.lin1.out_features
            new_lin1, keep_indices = prune_linear_layer(ffn.lin1, prune_ratio)
            
            if keep_indices is not None:
                # Adjust the output layer (lin2) to match
                new_lin2 = adjust_next_layer(ffn.lin2, keep_indices)
                
                # Replace the layers
                ffn.lin1 = new_lin1
                ffn.lin2 = new_lin2
                
                modifications.append({
                    'layer': f'transformer.layer.{layer_idx}.ffn',
                    'original_neurons': original_neurons,
                    'kept_neurons': len(keep_indices),
                    'reduction': f"{(1 - len(keep_indices)/original_neurons)*100:.1f}%"
                })
    
    return model, modifications

# Alternative: Unstructured pruning with true zero removal
def remove_zero_parameters(model):
    """
    Remove parameters that are exactly zero after unstructured pruning
    This creates a sparse representation but doesn't change layer dimensions
    """
    print("🧹 Removing zero parameters...")
    
    total_original = 0
    total_removed = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            original_size = param.numel()
            zero_mask = param.data == 0
            zeros_count = zero_mask.sum().item()
            
            total_original += original_size
            total_removed += zeros_count
            
            if zeros_count > 0:
                print(f"  {name}: {zeros_count}/{original_size} zeros ({zeros_count/original_size*100:.1f}%)")
    
    compression_ratio = (total_original - total_removed) / total_original
    print(f"📊 Total compression: {total_removed:,}/{total_original:,} removed ({(1-compression_ratio)*100:.1f}%)")
    
    return model

# ------------------ APPLY PRUNING ------------------ #
print("✂️ Applying structured pruning...")

# Option 1: Structured pruning (changes architecture)
pruned_model, modifications = prune_distilbert_model(model, prune_ratio=0.3)

print("📝 Pruning modifications:")
for mod in modifications:
    print(f"  {mod['layer']}: {mod['original_neurons']} -> {mod['kept_neurons']} neurons ({mod['reduction']} reduction)")

# Option 2: If you prefer unstructured pruning with zero removal
# Uncomment this section instead of structured pruning above:
"""
print("✂️ Applying unstructured pruning...")

# Apply L1 unstructured pruning first
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.3)

# Make pruning permanent (convert masks to actual zeros)
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        if hasattr(module, 'weight_mask'):
            prune.remove(module, "weight")

# Remove zero parameters (this doesn't actually change model size but marks them)
pruned_model = remove_zero_parameters(model)
modifications = ["Unstructured pruning with zero marking applied"]
"""

# ------------------ EVALUATION ------------------ #
print("🧪 Evaluating pruned model...")

device = torch.device("cpu")
pruned_model.to(device)
pruned_model.eval()

# Create data loader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
eval_dataloader = DataLoader(
    encoded_dataset["validation"],
    batch_size=32,
    collate_fn=data_collator,
    shuffle=False
)

# Manual evaluation
metric = evaluate.load("glue", "sst2")
all_predictions = []
all_labels = []
total_loss = 0
num_batches = 0

print("🔄 Running evaluation...")
with torch.no_grad():
    for batch in eval_dataloader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        try:
            # Forward pass
            outputs = pruned_model(**batch)
            
            # Get predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Accumulate results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            
            # Accumulate loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                total_loss += outputs.loss.item()
            
            num_batches += 1
            
            # Progress update
            if num_batches % 10 == 0:
                print(f"   Processed {num_batches} batches...")
                
        except Exception as e:
            print(f"❌ Error in batch {num_batches}: {e}")
            break

# Calculate metrics
if len(all_predictions) > 0:
    eval_results = metric.compute(predictions=all_predictions, references=all_labels)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"✅ Evaluation complete: {eval_results}")
else:
    print("❌ No predictions generated - evaluation failed")
    eval_results = {"accuracy": 0.0}
    avg_loss = float('inf')

# ------------------ METRICS ------------------ #
print("📏 Calculating metrics...")

# Model parameters (actual count after pruning)
current_params = sum(p.numel() for p in pruned_model.parameters())
param_reduction = original_param_count - current_params

# Calculate actual sparsity (for unstructured) or compression (for structured)
if modifications and isinstance(modifications[0], dict):  # Structured pruning
    # For structured pruning, sparsity is the reduction in total parameters
    sparsity = param_reduction / original_param_count
    actual_zeros = 0  # No zeros in structured pruning
else:  # Unstructured pruning
    # Count actual zeros
    total_params = 0
    zero_params = 0
    for param in pruned_model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    sparsity = zero_params / total_params if total_params > 0 else 0
    actual_zeros = zero_params

# Model size
temp_model_path = os.path.join(BASE_OUTPUT, "temp.pt")
torch.save(pruned_model.state_dict(), temp_model_path)
model_size_mb = os.path.getsize(temp_model_path) / (1024**2)
os.remove(temp_model_path)

# Inference latency
sample = tokenizer("This movie is fantastic!", return_tensors="pt", truncation=True)
sample = {k: v.to(device) for k, v in sample.items()}

try:
    with torch.no_grad():
        # Warm up
        for _ in range(10):
            _ = pruned_model(**sample)
        
        # Actual timing
        start = time.perf_counter()
        for _ in range(100):
            _ = pruned_model(**sample)
        latency = (time.perf_counter() - start) / 100
except Exception as e:
    print(f"❌ Latency measurement failed: {e}")
    latency = float('inf')

# Memory usage
process = psutil.Process(os.getpid())
mem_usage_mb = process.memory_info().rss / (1024**2)

# ------------------ SAVE RESULTS ------------------ #
print("💾 Saving results...")

metrics = {
    # Model size metrics
    "original_param_count": int(original_param_count),
    "current_param_count": int(current_params),
    "param_reduction": int(param_reduction),
    "compression_ratio": round(original_param_count / current_params, 2),
    
    # Performance metrics
    "accuracy": float(eval_results["accuracy"]),
    "eval_loss": float(avg_loss),
    
    # Sparsity/compression
    "sparsity": round(sparsity, 4),
    "zero_params": int(actual_zeros),
    
    # Efficiency metrics
    "model_size_mb": round(model_size_mb, 2),
    "avg_latency_sec": round(latency, 6),
    "memory_usage_mb": round(mem_usage_mb, 2),
    
    # Configuration
    "pruning_amount": 0.3,
    "pruning_method": "structured" if modifications and isinstance(modifications[0], dict) else "unstructured_with_removal",
    "quantization": None,
    
    # Modifications details
    "modifications": modifications
}

# Save results
with open(os.path.join(BASE_OUTPUT, "true_removal_pruning_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# Save model
torch.save(pruned_model.state_dict(), os.path.join(BASE_OUTPUT, "truly_pruned_model.pt"))

print("✅ Phase-2a (true parameter removal) done! Results + model saved in:", BASE_OUTPUT)
print(f"📊 Original parameters: {metrics['original_param_count']:,}")
print(f"📊 Current parameters: {metrics['current_param_count']:,}")
print(f"📊 Compression ratio: {metrics['compression_ratio']:.1f}x")
print(f"📊 Accuracy: {metrics['accuracy']:.2%}")
print(f"📊 Model size: {metrics['model_size_mb']:.1f} MB")

if latency != float('inf'):
    print(f"⚡ Avg latency: {metrics['avg_latency_sec']*1000:.2f} ms")
else:
    print("⚡ Latency: Could not measure (model may be broken)")