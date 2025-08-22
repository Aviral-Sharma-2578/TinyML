# baseline.py
import os
import time
import json
import psutil
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ------------------ PATH SETUP ------------------ #
BASE_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-1"))
MODEL_DIR = os.path.join(BASE_OUTPUT, "baseline_model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------ LOAD DATA ------------------ #
dataset = load_dataset("glue", "sst2")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_fn(batch):
    return tokenizer(batch["sentence"], truncation=True)

encoded_dataset = dataset.map(tokenize_fn, batched=True)

# ------------------ MODEL ------------------ #
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# ------------------ TRAINING ------------------ #
metric = evaluate.load("glue", "sst2")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return metric.compute(predictions=preds, references=labels)

args = TrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(MODEL_DIR, "logs"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,   # keep small for baseline
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=1,
)

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ------------------ EVALUATION ------------------ #
eval_results = trainer.evaluate()

# Param count
param_count = sum(p.numel() for p in model.parameters())
# Model size (approx in MB)
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "temp.pt"))
model_size_mb = os.path.getsize(os.path.join(MODEL_DIR, "temp.pt")) / (1024**2)
os.remove(os.path.join(MODEL_DIR, "temp.pt"))

# Inference latency
# Ensure model and sample are on the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
sample = {k: v.to(device) for k, v in tokenizer("This movie is fantastic!", return_tensors="pt").items()}

model.eval()
with torch.no_grad():
    start = time.time()
    for _ in range(100):  # avg over 100 runs
        _ = model(**sample)
    latency = (time.time() - start) / 100

# Memory usage
process = psutil.Process(os.getpid())
mem_usage_mb = process.memory_info().rss / (1024**2)

# ------------------ SAVE RESULTS ------------------ #
metrics = {
    "accuracy": float(eval_results["eval_accuracy"]),
    "f1": float(eval_results.get("eval_f1", 0.0)),  # not always available in SST-2
    "param_count": int(param_count),
    "model_size_mb": round(model_size_mb, 2),
    "avg_latency_sec": round(latency, 6),
    "memory_usage_mb": round(mem_usage_mb, 2),
}

with open(os.path.join(BASE_OUTPUT, "baseline_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# Save final model + tokenizer
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("✅ Baseline training done! Results saved in:", BASE_OUTPUT)
