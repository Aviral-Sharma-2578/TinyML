# prune_iterative_structured_true_reduction.py
import os
import time
import json
import psutil
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from copy import deepcopy

from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ------------------ CONFIG ------------------ #
@dataclass
class Config:
    base_phase2_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-2"))
    baseline_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-1", "baseline_model"))
    eval_batch_size: int = 32
    use_cpu: bool = False

    # Iterative structured pruning with true reduction
    num_steps: int = 3                 # number of pruning rounds
    amount_per_step: float = 0.10      # fraction pruned each round (per-layer)
    min_channels: int = 32             # minimum channels to keep in any layer

    # Fine-tuning after each step
    finetune_epochs: int = 1
    finetune_lr: float = 5e-5
    finetune_warmup_ratio: float = 0.06
    finetune_weight_decay: float = 0.01

    # Safety: stop if accuracy drops too much compared to initial
    early_stop_abs_drop: float = 0.05

cfg = Config()
os.makedirs(cfg.base_phase2_dir, exist_ok=True)

# ------------------ DATA ------------------ #
dataset = load_dataset("glue", "sst2")
tokenizer = DistilBertTokenizerFast.from_pretrained(cfg.baseline_dir)

def tokenize_fn(batch):
    return tokenizer(batch["sentence"], truncation=True)

encoded_dataset = dataset.map(tokenize_fn, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ------------------ MODEL ------------------ #
model = DistilBertForSequenceClassification.from_pretrained(cfg.baseline_dir)

# ------------------ HELPERS ------------------ #
metric = evaluate.load("glue", "sst2")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

def count_total_params(m: torch.nn.Module) -> int:
    """Count actual parameter count (after true reduction)"""
    return sum(p.numel() for p in m.parameters())

def get_pruneable_layers(model) -> Dict[str, torch.nn.Linear]:
    """Get all Linear layers we can prune (avoid classifier)"""
    pruneable = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Skip classifier and pre-classifier
            if any(skip in name for skip in ['classifier', 'pre_classifier']):
                continue
            pruneable[name] = module
    return pruneable

def calculate_channel_importance(layer: torch.nn.Linear, method='l2_norm') -> torch.Tensor:
    """Calculate importance scores for output channels"""
    weight = layer.weight.data
    
    if method == 'l2_norm':
        # L2 norm of each output channel (row)
        importance = torch.norm(weight, p=2, dim=1)
    elif method == 'l1_norm':
        # L1 norm of each output channel  
        importance = torch.norm(weight, p=1, dim=1)
    elif method == 'mean_activation':
        # Could use activations if we had them - fallback to L2
        importance = torch.norm(weight, p=2, dim=1)
    
    return importance

def prune_layer_channels(layer: torch.nn.Linear, channels_to_remove: List[int]) -> torch.nn.Linear:
    """Remove specific output channels from a layer"""
    old_weight = layer.weight.data
    old_bias = layer.bias.data if layer.bias is not None else None
    
    # Create mask for channels to keep
    all_channels = set(range(old_weight.size(0)))
    keep_channels = sorted(list(all_channels - set(channels_to_remove)))
    
    # Create new layer with reduced dimensions
    new_out_features = len(keep_channels)
    new_layer = torch.nn.Linear(layer.in_features, new_out_features, bias=layer.bias is not None)
    
    # Copy weights for kept channels
    new_layer.weight.data = old_weight[keep_channels, :]
    if old_bias is not None:
        new_layer.bias.data = old_bias[keep_channels]
    
    return new_layer, keep_channels

def update_next_layer_inputs(layer: torch.nn.Linear, kept_channels: List[int]) -> torch.nn.Linear:
    """Update the input dimensions of the next layer"""
    old_weight = layer.weight.data
    old_bias = layer.bias.data if layer.bias is not None else None
    
    # Create new layer with reduced input features
    new_in_features = len(kept_channels)
    new_layer = torch.nn.Linear(new_in_features, layer.out_features, bias=layer.bias is not None)
    
    # Copy weights for kept input channels
    new_layer.weight.data = old_weight[:, kept_channels]
    if old_bias is not None:
        new_layer.bias.data = old_bias
    
    return new_layer

def find_layer_connections(model) -> Dict[str, str]:
    """Find which layers connect to each other in DistilBERT"""
    connections = {}
    
    # DistilBERT architecture connections
    for i in range(6):  # 6 transformer layers
        # Attention layers
        q_name = f"distilbert.transformer.layer.{i}.attention.q_lin"
        k_name = f"distilbert.transformer.layer.{i}.attention.k_lin" 
        v_name = f"distilbert.transformer.layer.{i}.attention.v_lin"
        out_name = f"distilbert.transformer.layer.{i}.attention.out_lin"
        
        # FFN layers
        ffn1_name = f"distilbert.transformer.layer.{i}.ffn.lin1"
        ffn2_name = f"distilbert.transformer.layer.{i}.ffn.lin2"
        
        # Attention: q,k,v -> out_lin input
        connections[q_name] = out_name
        connections[k_name] = out_name  
        connections[v_name] = out_name
        
        # FFN: lin1 -> lin2 input
        connections[ffn1_name] = ffn2_name
    
    return connections

def structured_prune_with_true_reduction(model, amount: float = 0.1):
    """
    Perform structured pruning with actual parameter reduction
    """
    pruneable_layers = get_pruneable_layers(model)
    layer_connections = find_layer_connections(model)
    
    pruning_plan = {}  # layer_name -> channels_to_remove
    
    print(f"\n=== Planning channel removal (target: {amount*100:.1f}% per layer) ===")
    
    # Step 1: Decide which channels to remove from each layer
    for layer_name, layer in pruneable_layers.items():
        current_channels = layer.out_features
        target_remove = max(1, int(amount * current_channels))
        
        # Safety: don't remove too many channels
        min_keep = max(cfg.min_channels, current_channels // 4)
        max_remove = current_channels - min_keep
        channels_to_remove = min(target_remove, max_remove)
        
        if channels_to_remove > 0:
            # Calculate importance scores
            importance = calculate_channel_importance(layer)
            
            # Find least important channels
            _, indices = torch.topk(importance, channels_to_remove, largest=False)
            pruning_plan[layer_name] = indices.tolist()
            
            print(f"{layer_name}: removing {channels_to_remove}/{current_channels} channels")
        else:
            print(f"{layer_name}: skipping (too few channels)")
    
    print(f"\n=== Applying structural changes ===")
    
    # Step 2: Apply the pruning by modifying the model structure
    channel_mappings = {}  # track how channels were remapped
    
    for layer_name, channels_to_remove in pruning_plan.items():
        # Get the layer
        layer_parts = layer_name.split('.')
        current_module = model
        for part in layer_parts[:-1]:
            current_module = getattr(current_module, part)
        
        old_layer = getattr(current_module, layer_parts[-1])
        
        # Prune this layer's output channels
        new_layer, kept_channels = prune_layer_channels(old_layer, channels_to_remove)
        setattr(current_module, layer_parts[-1], new_layer)
        
        channel_mappings[layer_name] = kept_channels
        
        print(f"✓ Pruned {layer_name}: {old_layer.out_features} -> {new_layer.out_features}")
        
        # Update connected layer's input dimensions
        if layer_name in layer_connections:
            next_layer_name = layer_connections[layer_name]
            
            # Handle special case for attention layers (q,k,v all connect to out_lin)
            if any(x in layer_name for x in ['q_lin', 'k_lin', 'v_lin']):
                # For attention, we need to handle concatenated outputs
                continue  # Skip for now - this is complex
            
            # Get the next layer
            next_parts = next_layer_name.split('.')
            next_module = model
            for part in next_parts[:-1]:
                next_module = getattr(next_module, part)
            
            if hasattr(next_module, next_parts[-1]):
                next_old_layer = getattr(next_module, next_parts[-1])
                next_new_layer = update_next_layer_inputs(next_old_layer, kept_channels)
                setattr(next_module, next_parts[-1], next_new_layer)
                
                print(f"✓ Updated {next_layer_name}: {next_old_layer.in_features} -> {next_new_layer.in_features} inputs")

def debug_model_structure(model):
    """Debug the actual model structure to understand layer access"""
    print("\n=== Model Structure Debug ===")
    try:
        print(f"Model type: {type(model)}")
        print(f"Has distilbert: {hasattr(model, 'distilbert')}")
        
        if hasattr(model, 'distilbert'):
            print(f"Transformer type: {type(model.distilbert.transformer)}")
            print(f"Has layer: {hasattr(model.distilbert.transformer, 'layer')}")
            
            if hasattr(model.distilbert.transformer, 'layer'):
                layers = model.distilbert.transformer.layer
                print(f"Layer type: {type(layers)}")
                print(f"Number of layers: {len(layers)}")
                
                # Check first layer structure
                if len(layers) > 0:
                    first_layer = layers[0]
                    print(f"First layer type: {type(first_layer)}")
                    print(f"First layer attributes: {dir(first_layer)}")
                    
                    if hasattr(first_layer, 'ffn'):
                        ffn = first_layer.ffn
                        print(f"FFN type: {type(ffn)}")
                        print(f"FFN attributes: {dir(ffn)}")
                        
                        if hasattr(ffn, 'lin1') and hasattr(ffn, 'lin2'):
                            print(f"lin1 type: {type(ffn.lin1)}")
                            print(f"lin1 shape: {ffn.lin1.weight.shape}")
                            print(f"lin2 type: {type(ffn.lin2)}")
                            print(f"lin2 shape: {ffn.lin2.weight.shape}")
    except Exception as e:
        print(f"Debug failed: {e}")

def safe_structured_prune_ffn_only(model, amount: float = 0.1):
    """
    Safer version: only prune FFN layers to avoid attention complexity
    """
    print(f"\n=== Pruning FFN layers only (amount: {amount*100:.1f}%) ===")
    
    # First, debug the structure
    debug_model_structure(model)
    
    try:
        layers = model.distilbert.transformer.layer
        num_layers = len(layers)
        print(f"Found {num_layers} transformer layers")
    except Exception as e:
        print(f"Failed to access transformer layers: {e}")
        return
    
    for i in range(num_layers):
        try:
            # Access layers using proper indexing
            layer_module = layers[i]
            
            # Verify FFN exists
            if not hasattr(layer_module, 'ffn'):
                print(f"Layer {i} has no FFN module, skipping")
                continue
                
            ffn_module = layer_module.ffn
            
            # Verify lin1 and lin2 exist
            if not (hasattr(ffn_module, 'lin1') and hasattr(ffn_module, 'lin2')):
                print(f"Layer {i} FFN missing lin1/lin2, skipping")
                continue
                
            ffn1_layer = ffn_module.lin1
            ffn2_layer = ffn_module.lin2
            
            print(f"\nProcessing Layer {i}:")
            print(f"  FFN1 shape: {ffn1_layer.weight.shape}")
            print(f"  FFN2 shape: {ffn2_layer.weight.shape}")
            
        except Exception as e:
            print(f"Could not access FFN layers for layer {i}: {e}")
            continue
        
        # Prune FFN1 (intermediate layer)
        current_channels = ffn1_layer.out_features
        channels_to_remove_count = max(1, int(amount * current_channels))
        
        # Safety check
        min_keep = max(cfg.min_channels, current_channels // 4)
        if current_channels - channels_to_remove_count < min_keep:
            channels_to_remove_count = current_channels - min_keep
        
        if channels_to_remove_count <= 0:
            print(f"  Skipping layer {i}: would remove too many channels")
            continue
            
        print(f"  Removing {channels_to_remove_count}/{current_channels} channels")
        
        try:
            # Calculate importance (L2 norm of output channels)
            importance = torch.norm(ffn1_layer.weight.data, p=2, dim=1)
            _, remove_indices = torch.topk(importance, channels_to_remove_count, largest=False)
            
            # Create channel masks
            all_channels = set(range(current_channels))
            keep_channels = sorted(list(all_channels - set(remove_indices.tolist())))
            
            print(f"  Keeping {len(keep_channels)} channels: {keep_channels[:5]}...")
            
            # Update FFN1 (remove output channels)
            new_ffn1 = torch.nn.Linear(
                ffn1_layer.in_features, 
                len(keep_channels), 
                bias=ffn1_layer.bias is not None
            ).to(ffn1_layer.weight.device)
            
            new_ffn1.weight.data = ffn1_layer.weight.data[keep_channels, :].clone()
            if ffn1_layer.bias is not None:
                new_ffn1.bias.data = ffn1_layer.bias.data[keep_channels].clone()
            
            # Update FFN2 (remove input channels)
            new_ffn2 = torch.nn.Linear(
                len(keep_channels), 
                ffn2_layer.out_features, 
                bias=ffn2_layer.bias is not None
            ).to(ffn2_layer.weight.device)
            
            new_ffn2.weight.data = ffn2_layer.weight.data[:, keep_channels].clone()
            if ffn2_layer.bias is not None:
                new_ffn2.bias.data = ffn2_layer.bias.data.clone()
            
            # Replace layers in model
            layer_module.ffn.lin1 = new_ffn1
            layer_module.ffn.lin2 = new_ffn2
            
            print(f"  ✓ Layer {i} FFN: {current_channels} -> {len(keep_channels)} channels")
            
        except Exception as e:
            print(f"  ❌ Failed to prune layer {i}: {e}")
            continue

def evaluate_model(m: torch.nn.Module, tag: str):
    args = TrainingArguments(
        output_dir=os.path.join(cfg.base_phase2_dir, f"temp_{tag}"),
        per_device_eval_batch_size=cfg.eval_batch_size,
        use_cpu=cfg.use_cpu,
        report_to=[],
        logging_strategy="no",
    )
    trainer = Trainer(
        model=m,
        args=args,
        eval_dataset=encoded_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer.evaluate()

def finetune_model(m: torch.nn.Module, step_idx: int):
    args = TrainingArguments(
        output_dir=os.path.join(cfg.base_phase2_dir, f"ft_step{step_idx}"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.finetune_lr,
        num_train_epochs=cfg.finetune_epochs,
        warmup_ratio=cfg.finetune_warmup_ratio,
        weight_decay=cfg.finetune_weight_decay,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        use_cpu=cfg.use_cpu,
        report_to=[],
    )
    trainer = Trainer(
        model=m,
        args=args,
        train_dataset=encoded_dataset["train"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

def measure_latency_and_mem(m: torch.nn.Module):
    if cfg.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    m.to(device)
    sample = {k: v.to(device) for k, v in tokenizer("This movie is fantastic!", return_tensors="pt").items()}
    
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = m(**sample)
        
        # Measure latency
        start = time.time()
        for _ in range(100):
            _ = m(**sample)
        latency = (time.time() - start) / 100
    
    process = psutil.Process(os.getpid())
    mem_usage_mb = process.memory_info().rss / (1024**2)
    return latency, mem_usage_mb

def temp_checkpoint_size_mb(m: torch.nn.Module) -> float:
    tmp_path = os.path.join(cfg.base_phase2_dir, "temp.pt")
    torch.save(m.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024**2)
    os.remove(tmp_path)
    return size_mb

# ------------------ BASELINE EVAL ------------------ #
print(">> Evaluating baseline...")
baseline_eval = evaluate_model(model, tag="baseline")
best_acc = float(baseline_eval.get("eval_accuracy", 0.0))
print(f">> Baseline accuracy: {best_acc:.4f}")

# Count initial parameters
params_before = count_total_params(model)
print(f">> Initial parameters: {params_before:,}")

# ------------------ ITERATIVE PRUNING WITH TRUE REDUCTION ------------------ #
history = []

for step in range(1, cfg.num_steps + 1):
    print(f"\n" + "="*60)
    print(f"PRUNING STEP {step}/{cfg.num_steps}")
    print("="*60)
    
    # Count before pruning
    params_before_step = count_total_params(model)
    print(f"Parameters before step {step}: {params_before_step:,}")
    
    # Apply pruning with true parameter reduction
    try:
        safe_structured_prune_ffn_only(model, amount=cfg.amount_per_step)
        print("✓ Structural pruning completed successfully")
    except Exception as e:
        print(f"❌ Pruning failed: {e}")
        print("Skipping this step...")
        break
    
    # Count after pruning
    params_after_prune = count_total_params(model)
    reduction = params_before_step - params_after_prune
    reduction_percent = (reduction / params_before_step) * 100
    
    print(f"\n📊 Pruning Results:")
    print(f"   Parameters after pruning: {params_after_prune:,}")
    print(f"   Reduction this step: {reduction:,} ({reduction_percent:.1f}%)")
    
    # Fine-tune to recover accuracy
    print(f"\n🔧 Fine-tuning for {cfg.finetune_epochs} epoch(s)...")
    try:
        finetune_model(model, step_idx=step)
        print("✓ Fine-tuning completed")
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
    
    # Evaluate
    eval_res = evaluate_model(model, tag=f"after_step{step}")
    acc = float(eval_res.get("eval_accuracy", 0.0))
    f1 = float(eval_res.get("eval_f1", 0.0)) if "eval_f1" in eval_res else 0.0
    
    # Measure efficiency
    size_mb = temp_checkpoint_size_mb(model)
    latency, mem_mb = measure_latency_and_mem(model)
    
    # Calculate cumulative metrics
    total_reduction = params_before - params_after_prune
    total_reduction_percent = (total_reduction / params_before) * 100
    compression_ratio = params_before / params_after_prune
    
    print(f"\n📈 Step {step} Results:")
    print(f"   Accuracy: {acc:.4f} (Δ{acc - best_acc:+.4f})")
    print(f"   Parameters: {params_after_prune:,}")
    print(f"   Total reduction: {total_reduction:,} ({total_reduction_percent:.1f}%)")
    print(f"   Compression ratio: {compression_ratio:.2f}x")
    print(f"   Model size: {size_mb:.2f} MB")
    print(f"   Latency: {latency:.6f}s")
    
    history.append({
        "step": step,
        "accuracy": acc,
        "f1": f1,
        "total_param_count": int(params_after_prune),
        "params_reduced_this_step": int(reduction),
        "total_params_reduced": int(total_reduction),
        "reduction_percent_this_step": round(reduction_percent, 2),
        "total_reduction_percent": round(total_reduction_percent, 2),
        "compression_ratio": round(compression_ratio, 2),
        "model_size_mb": round(size_mb, 2),
        "avg_latency_sec": round(latency, 6),
        "memory_usage_mb": round(mem_mb, 2),
        "amount_this_step": cfg.amount_per_step,
    })
    
    # Early stopping
    if acc + cfg.early_stop_abs_drop < best_acc:
        print(f"\n⚠️  Early stopping: Accuracy dropped by > {cfg.early_stop_abs_drop:.3f}")
        print(f"   Best: {best_acc:.4f} → Current: {acc:.4f}")
        break
    
    best_acc = max(best_acc, acc)

# ------------------ FINAL EVALUATION & SAVE ------------------ #
print(f"\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

final_eval = evaluate_model(model, tag="final")
final_acc = float(final_eval.get("eval_accuracy", 0.0))
final_f1 = float(final_eval.get("eval_f1", 0.0)) if "eval_f1" in final_eval else 0.0
params_final = count_total_params(model)
size_final_mb = temp_checkpoint_size_mb(model)
latency_final, mem_final_mb = measure_latency_and_mem(model)

# Final metrics
total_reduction_final = params_before - params_final
final_compression = params_before / params_final
final_reduction_percent = (total_reduction_final / params_before) * 100

metrics = {
    "method": "iterative_structured_pruning_with_true_reduction",
    "initial_accuracy": float(baseline_eval.get("eval_accuracy", 0.0)),
    "final_accuracy": final_acc,
    "final_f1": final_f1,
    "accuracy_change": round(final_acc - baseline_eval.get("eval_accuracy", 0.0), 4),
    
    "param_count_before": int(params_before),
    "param_count_after": int(params_final),
    "total_params_removed": int(total_reduction_final),
    "compression_ratio": round(final_compression, 2),
    "reduction_percentage": round(final_reduction_percent, 2),
    
    "model_size_mb_before": "unknown",
    "model_size_mb_after": round(size_final_mb, 2),
    "avg_latency_sec_final": round(latency_final, 6),
    "memory_usage_mb_final": round(mem_final_mb, 2),
    
    "config": {
        "num_pruning_steps": cfg.num_steps,
        "amount_per_step": cfg.amount_per_step,
        "min_channels_kept": cfg.min_channels,
        "pruning_target": "ffn_layers_only"
    },
    
    "step_history": history,
}

# Save results
metrics_path = os.path.join(cfg.base_phase2_dir, "true_reduction_pruning_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

model_path = os.path.join(cfg.base_phase2_dir, "pruned_true_reduction.pt")
torch.save(model.state_dict(), model_path)

# Summary
print(f"\n🎉 TRUE PARAMETER REDUCTION COMPLETE!")
print("="*60)
print(f"📊 RESULTS SUMMARY")
print(f"   Initial accuracy:     {baseline_eval.get('eval_accuracy', 0.0):.4f}")
print(f"   Final accuracy:       {final_acc:.4f}")
print(f"   Accuracy change:      {final_acc - baseline_eval.get('eval_accuracy', 0.0):+.4f}")
print()
print(f"   Initial parameters:   {params_before:,}")
print(f"   Final parameters:     {params_final:,}")
print(f"   Parameters removed:   {total_reduction_final:,}")
print(f"   Compression ratio:    {final_compression:.2f}x")
print(f"   Size reduction:       {final_reduction_percent:.1f}%")
print()
print(f"   Model size:          {size_final_mb:.2f} MB")
print(f"   Inference latency:   {latency_final:.6f} sec")
print()
print(f"💾 FILES SAVED:")
print(f"   Metrics: {metrics_path}")
print(f"   Model:   {model_path}")
print("="*60)