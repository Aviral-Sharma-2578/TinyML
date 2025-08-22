# prune_baseline.py DO NOT USE
import os
import time
import json
import psutil
import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from torch.nn.utils import prune

# ------------------ PATHS ------------------ #
BASE_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-2a"))
os.makedirs(BASE_OUTPUT, exist_ok=True)

BASELINE_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-1", "baseline_model"))

# ------------------ LOAD DATA ------------------ #
dataset = load_dataset("glue", "sst2")
tokenizer = DistilBertTokenizerFast.from_pretrained(BASELINE_MODEL)

def tokenize_fn(batch):
    return tokenizer(batch["sentence"], truncation=True)

encoded_dataset = dataset.map(tokenize_fn, batched=True)

# ------------------ LOAD BASELINE MODEL ------------------ #
model = DistilBertForSequenceClassification.from_pretrained(BASELINE_MODEL)

# ------------------ PRUNING ------------------ #
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.3)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.remove(module, "weight")

# ------------------ EVALUATION ------------------ #
metric = evaluate.load("glue", "sst2")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

args = TrainingArguments(
    output_dir=os.path.join(BASE_OUTPUT, "temp"),
    per_device_eval_batch_size=32,
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

eval_results = trainer.evaluate()

# ------------------ METRICS ------------------ #
param_count = sum(p.numel() for p in model.parameters())

torch.save(model.state_dict(), os.path.join(BASE_OUTPUT, "temp.pt"))
model_size_mb = os.path.getsize(os.path.join(BASE_OUTPUT, "temp.pt")) / (1024**2)
os.remove(os.path.join(BASE_OUTPUT, "temp.pt"))

device = torch.device("cpu")
model.to(device)
sample = {k: v.to(device) for k, v in tokenizer("This movie is fantastic!", return_tensors="pt").items()}

with torch.no_grad():
    start = time.time()
    for _ in range(100):
        _ = model(**sample)
    latency = (time.time() - start) / 100

process = psutil.Process(os.getpid())
mem_usage_mb = process.memory_info().rss / (1024**2)

# ------------------ SAVE RESULTS ------------------ #
metrics = {
    "accuracy": float(eval_results["eval_accuracy"]),
    "f1": float(eval_results.get("eval_f1", 0.0)),
    "param_count": int(param_count),
    "model_size_mb": round(model_size_mb, 2),
    "avg_latency_sec": round(latency, 6),
    "memory_usage_mb": round(mem_usage_mb, 2),
    "pruning_amount": 0.3,
    "quantization": None,
}

with open(os.path.join(BASE_OUTPUT, "prune_baseline_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

torch.save(model.state_dict(), os.path.join(BASE_OUTPUT, "pruned_model.pt"))

print("✅ Phase-2a (prune only) done! Results + model saved in:", BASE_OUTPUT)
