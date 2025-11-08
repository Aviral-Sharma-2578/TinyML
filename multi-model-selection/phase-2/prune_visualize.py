import os
import time
import json
import psutil
import torch
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ------------------ CONFIG ------------------ #
class Config:
    base_phase2_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-2"))
    baseline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-1", "baseline_model"))
    eval_batch_size = 32
    use_cpu = False
    num_steps = 1
    min_channels = 32

cfg = Config()
os.makedirs(cfg.base_phase2_dir, exist_ok=True)

# ------------------ DATA ------------------ #
dataset = load_dataset("glue", "sst2")
tokenizer = DistilBertTokenizerFast.from_pretrained(cfg.baseline_dir)

def tokenize_fn(batch):
    return tokenizer(batch["sentence"], truncation=True)

encoded_dataset = dataset.map(tokenize_fn, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

metric = evaluate.load("glue", "sst2")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# ------------------ HELPERS ------------------ #
def count_total_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

def safe_structured_prune_ffn_only(model, amount: float = 0.1):
    """
    Prunes FFN layers only — safer structured true reduction.
    """
    print(f"\n=== Pruning FFN layers only (amount: {amount*100:.1f}%) ===")

    layers = model.distilbert.transformer.layer
    for i, layer_module in enumerate(layers):
        if not hasattr(layer_module, "ffn"):
            continue

        ffn_module = layer_module.ffn
        if not hasattr(ffn_module, "lin1") or not hasattr(ffn_module, "lin2"):
            continue

        ffn1, ffn2 = ffn_module.lin1, ffn_module.lin2
        current_channels = ffn1.out_features
        channels_to_remove_count = max(1, int(amount * current_channels))

        min_keep = max(cfg.min_channels, current_channels // 4)
        if current_channels - channels_to_remove_count < min_keep:
            channels_to_remove_count = current_channels - min_keep

        if channels_to_remove_count <= 0:
            continue

        importance = torch.norm(ffn1.weight.data, p=2, dim=1)
        _, remove_indices = torch.topk(importance, channels_to_remove_count, largest=False)
        keep_channels = sorted(list(set(range(current_channels)) - set(remove_indices.tolist())))

        # Create new layers
        new_ffn1 = torch.nn.Linear(
            ffn1.in_features, len(keep_channels), bias=ffn1.bias is not None
        ).to(ffn1.weight.device)
        new_ffn2 = torch.nn.Linear(
            len(keep_channels), ffn2.out_features, bias=ffn2.bias is not None
        ).to(ffn2.weight.device)

        new_ffn1.weight.data = ffn1.weight.data[keep_channels, :].clone()
        if ffn1.bias is not None:
            new_ffn1.bias.data = ffn1.bias.data[keep_channels].clone()

        new_ffn2.weight.data = ffn2.weight.data[:, keep_channels].clone()
        if ffn2.bias is not None:
            new_ffn2.bias.data = ffn2.bias.data.clone()

        layer_module.ffn.lin1 = new_ffn1
        layer_module.ffn.lin2 = new_ffn2

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

def measure_latency_and_mem(m: torch.nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.use_cpu else "cpu")
    m.to(device)
    sample = {k: v.to(device) for k, v in tokenizer("This movie is fantastic!", return_tensors="pt").items()}

    with torch.no_grad():
        for _ in range(5):
            _ = m(**sample)
        start = time.time()
        for _ in range(50):
            _ = m(**sample)
        latency = (time.time() - start) / 50

    process = psutil.Process(os.getpid())
    mem_usage_mb = process.memory_info().rss / (1024**2)
    return latency, mem_usage_mb

def temp_checkpoint_size_mb(m: torch.nn.Module) -> float:
    tmp_path = os.path.join(cfg.base_phase2_dir, "temp.pt")
    torch.save(m.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024**2)
    os.remove(tmp_path)
    return size_mb

# ------------------ MULTI-AMOUNT EXPERIMENT ------------------ #
pruning_amounts = [0.05, 0.10, 0.15, 0.20]
all_results = []

print("\n>> Evaluating baseline once...")
baseline_model = DistilBertForSequenceClassification.from_pretrained(cfg.baseline_dir)
baseline_eval = evaluate_model(baseline_model, tag="baseline")
base_acc = float(baseline_eval.get("eval_accuracy", 0.0))
base_params = count_total_params(baseline_model)
print(f">> Baseline accuracy: {base_acc:.4f}")
print(f">> Baseline parameters: {base_params:,}")

for amt in pruning_amounts:
    print("\n" + "=" * 70)
    print(f"Running Structured True Reduction with amount_per_step = {amt:.2f}")
    print("=" * 70)

    model = DistilBertForSequenceClassification.from_pretrained(cfg.baseline_dir)

    params_before = count_total_params(model)
    safe_structured_prune_ffn_only(model, amount=amt)
    params_after = count_total_params(model)

    eval_res = evaluate_model(model, tag=f"amt_{amt}")
    acc = float(eval_res.get("eval_accuracy", 0.0))
    latency, mem = measure_latency_and_mem(model)
    size_mb = temp_checkpoint_size_mb(model)

    reduction_percent = (params_before - params_after) / params_before * 100
    compression = params_before / params_after

    all_results.append({
        "amount": amt,
        "accuracy": acc,
        "reduction_percent": reduction_percent,
        "compression_ratio": compression,
        "latency_sec": latency,
        "memory_mb": mem,
        "model_size_mb": size_mb,
    })

# ------------------ VISUALIZATION ------------------ #
if len(all_results) > 0:
    amounts = [r["amount"] * 100 for r in all_results]
    accs = [r["accuracy"] for r in all_results]
    reductions = [r["reduction_percent"] for r in all_results]
    compressions = [r["compression_ratio"] for r in all_results]
    latencies = [r["latency_sec"] for r in all_results]
    memories = [r["memory_mb"] for r in all_results]

    plt.style.use("seaborn-v0_8-muted")

    # 1️⃣ Accuracy vs Pruning %
    plt.figure(figsize=(7, 5))
    plt.plot(amounts, accs, marker="o", color="dodgerblue")
    plt.title("Accuracy vs. Pruning Amount")
    plt.xlabel("Pruning per step (%)")
    plt.ylabel("Final Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 2️⃣ Compression Ratio vs Pruning %
    plt.figure(figsize=(7, 5))
    plt.plot(amounts, compressions, marker="s", color="orange")
    plt.title("Compression Ratio vs. Pruning Amount")
    plt.xlabel("Pruning per step (%)")
    plt.ylabel("Compression Ratio (×)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 3️⃣ Accuracy–Compression Trade-off
    plt.figure(figsize=(7, 5))
    plt.plot(compressions, accs, marker="D", color="purple")
    for r in all_results:
        plt.text(r["compression_ratio"], r["accuracy"] - 0.002, f"{r['amount']*100:.0f}%", fontsize=9)
    plt.title("Accuracy vs. Compression Trade-off")
    plt.xlabel("Compression Ratio (×)")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 4️⃣ Latency and Memory vs Pruning %
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()
    ax1.plot(amounts, latencies, marker="o", color="steelblue", label="Latency (s)")
    ax2.plot(amounts, memories, marker="^", color="crimson", label="Memory (MB)")
    ax1.set_xlabel("Pruning per step (%)")
    ax1.set_ylabel("Latency (s)", color="steelblue")
    ax2.set_ylabel("Memory (MB)", color="crimson")
    plt.title("Latency and Memory vs. Pruning Amount")
    ax1.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Print summary table
    print("\n📊 Summary Results:")
    print(pd.DataFrame(all_results).round(4))
else:
    print("⚠️ No results found — skipping visualization.")
