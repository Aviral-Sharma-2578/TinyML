# distillation.py
import os
import time
import json
import psutil
import torch
import evaluate
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DataCollatorWithPadding # Import the collator
)
from torch.optim import AdamW

# ------------------ PATHS ------------------ #
BASE_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-3"))
os.makedirs(BASE_OUTPUT, exist_ok=True)

BASELINE_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-1", "baseline_model"))
STUDENT_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-2", "prune_quantized_model.pt"))

# ------------------ LOAD DATA ------------------ #
dataset = load_dataset("glue", "sst2")
tokenizer = DistilBertTokenizerFast.from_pretrained(BASELINE_MODEL)

def tokenize_fn(batch):
    # Add both padding and truncation here
    return tokenizer(batch["sentence"], truncation=True, padding=True)

# Add remove_columns to prevent the original "sentence" column from being passed to the collator
encoded_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence"])

# ------------------ LOAD TEACHER + STUDENT ------------------ #
teacher = DistilBertForSequenceClassification.from_pretrained(BASELINE_MODEL)
student = DistilBertForSequenceClassification.from_pretrained(BASELINE_MODEL)
student.load_state_dict(torch.load(STUDENT_MODEL), strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher.to(device)
student.to(device)

teacher.eval()
optimizer = AdamW(student.parameters(), lr=5e-5)

# ------------------ KNOWLEDGE DISTILLATION ------------------ #
alpha = 0.5
T = 2.0

def distill_loss(student_logits, teacher_logits, labels):
    ce_loss = F.cross_entropy(student_logits, labels)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean"
    ) * (T * T)
    return alpha * ce_loss + (1 - alpha) * kd_loss

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Use the data collator for the training dataloader
train_loader = torch.utils.data.DataLoader(
    encoded_dataset["train"], 
    batch_size=16, 
    shuffle=True, 
    collate_fn=data_collator
)

student.train()
for epoch in range(1):
    for batch in train_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        labels = batch["labels"].to(device)

        with torch.no_grad():
            teacher_logits = teacher(**inputs).logits

        student_logits = student(**inputs).logits
        loss = distill_loss(student_logits, teacher_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("✅ Distillation complete.")

# ------------------ EVALUATION ------------------ #
metric = evaluate.load("glue", "sst2")

def evaluate_model(model, split="validation"):
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        encoded_dataset[split], 
        batch_size=32,
        collate_fn=data_collator
    )
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1)
            preds.extend(pred.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())
    return metric.compute(predictions=preds, references=labels)

results = evaluate_model(student)

# Model size
torch.save(student.state_dict(), os.path.join(BASE_OUTPUT, "distilled_model.pt"))
model_size_mb = os.path.getsize(os.path.join(BASE_OUTPUT, "distilled_model.pt")) / (1024**2)

# Latency
sample = tokenizer("This movie is fantastic!", return_tensors="pt").to(device)
with torch.no_grad():
    start = time.time()
    for _ in range(100):
        _ = student(**sample)
    latency = (time.time() - start) / 100

# Memory
process = psutil.Process(os.getpid())
mem_usage_mb = process.memory_info().rss / (1024**2)

# Save metrics
metrics = {
    "accuracy": float(results["accuracy"]),
    "param_count": sum(p.numel() for p in student.parameters()),
    "model_size_mb": round(model_size_mb, 2),
    "avg_latency_sec": round(latency, 6),
    "memory_usage_mb": round(mem_usage_mb, 2),
    "distillation": {
        "alpha": alpha,
        "temperature": T,
        "epochs": 1
    }
}

with open(os.path.join(BASE_OUTPUT, "distillation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Phase-3 done! Distilled model + metrics saved in:", BASE_OUTPUT)
