import os
import time
import torch
import json
import copy
import numpy as np
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import evaluate
from torchao.quantization import quantize_, Int8WeightOnlyConfig

# --- Config ---
BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-1", "baseline_model"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-3"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
EVAL_BATCH_SIZE = 32

# --- Data Setup ---
dataset = load_dataset("glue", "sst2")
tokenizer = DistilBertTokenizerFast.from_pretrained(BASELINE_DIR)

def tokenize_fn(batch):
    return tokenizer(batch["sentence"], truncation=True)

encoded_dataset = dataset.map(tokenize_fn, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.load("glue", "sst2")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# --- Helpers ---
def evaluate_model(model, tag="model"):
    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"temp_{tag}"),
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        report_to=[],
        logging_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=encoded_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer.evaluate()

def measure_inference_speed(model, num_samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    sample = tokenizer("This movie is fantastic!", return_tensors="pt")
    sample = {k: v.to(device) for k, v in sample.items()}

    with torch.no_grad():
        for _ in range(10):
            _ = model(**sample)
    start = time.time()
    with torch.no_grad():
        for _ in range(num_samples):
            _ = model(**sample)
    return (time.time() - start) / num_samples

def get_model_size_mb(model):
    temp_path = os.path.join(OUTPUT_DIR, "temp_size.pt")
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 ** 2)
    os.remove(temp_path)
    return size_mb

def load_baseline_model():
    print("Loading baseline model...")
    return DistilBertForSequenceClassification.from_pretrained(BASELINE_DIR)

# --- Main ---
def main():
    print("="*50)
    print("torchao Quantization - Weight Only Int8 (Baseline)")
    print("="*50)

    # Load baseline model
    model = load_baseline_model()
    
    # Evaluate original model
    print("\nEvaluating original model...")
    orig_eval = evaluate_model(model, tag="original")
    orig_metrics = {
        "accuracy": orig_eval.get("eval_accuracy", 0.0),
        "f1": orig_eval.get("eval_f1", 0.0),
        "inference_time": measure_inference_speed(model),
        "model_size_mb": get_model_size_mb(model),
        "param_count": sum(p.numel() for p in model.parameters())
    }
    print(f"Original: acc={orig_metrics['accuracy']:.4f}, size={orig_metrics['model_size_mb']:.2f}MB, params={orig_metrics['param_count']:,}")

    # --- Quantize (Weight-only Int8) ---
    print("\nApplying quantization...")
    quant_model = copy.deepcopy(model)
    
    try:
        quantize_(quant_model, Int8WeightOnlyConfig())
        print("✓ Applied Int8 weight-only quantization")
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return

    # Evaluate quantized model
    print("Evaluating quantized model...")
    q_eval = evaluate_model(quant_model, tag="quantized")
    q_metrics = {
        "accuracy": q_eval.get("eval_accuracy", 0.0),
        "f1": q_eval.get("eval_f1", 0.0),
        "inference_time": measure_inference_speed(quant_model),
        "model_size_mb": get_model_size_mb(quant_model),
        "param_count": sum(p.numel() for p in quant_model.parameters())
    }
    print(f"Quantized: acc={q_metrics['accuracy']:.4f}, size={q_metrics['model_size_mb']:.2f}MB, params={q_metrics['param_count']:,}")

    # --- Calculate improvements ---
    size_reduction = (orig_metrics['model_size_mb'] - q_metrics['model_size_mb']) / orig_metrics['model_size_mb'] * 100
    speed_improvement = (orig_metrics['inference_time'] - q_metrics['inference_time']) / orig_metrics['inference_time'] * 100
    accuracy_drop = orig_metrics['accuracy'] - q_metrics['accuracy']

    print(f"\n📊 QUANTIZATION RESULTS:")
    print(f"   Size reduction:     {size_reduction:+.1f}%")
    print(f"   Speed improvement:  {speed_improvement:+.1f}%")
    print(f"   Accuracy change:    {-accuracy_drop:+.4f}")

    # --- Save quantized model ---
    save_path = os.path.join(OUTPUT_DIR, "quantized_baseline_weight_only_int8.pt")
    torch.save(quant_model.state_dict(), save_path)
    
    # Also save full model object for easier loading later
    full_save_path = os.path.join(OUTPUT_DIR, "quantized_baseline_model_full.pt")
    torch.save(quant_model, full_save_path)
    
    print(f"✓ Saved quantized model state dict to {save_path}")
    print(f"✓ Saved full quantized model to {full_save_path}")

    # Save comprehensive results
    results = {
        "quantization_method": "torchao_int8_weight_only",
        "original": orig_metrics,
        "quantized_int8_wo": q_metrics,
        "improvements": {
            "size_reduction_percent": round(size_reduction, 2),
            "speed_improvement_percent": round(speed_improvement, 2),
            "accuracy_change": round(-accuracy_drop, 4)
        },
        "config": {
            "quantization_config": "Int8WeightOnlyConfig",
            "baseline_model_loaded": True
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "torchao_quantization_baseline_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("✓ Results saved to torchao_quantization_baseline_results.json")

if __name__ == "__main__":
    main()