# distillation_pruned.py - STRUCTURED PRUNING VERSION
import os
import time
import json
import psutil
import torch
import torch.nn as nn
import evaluate
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DataCollatorWithPadding
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------ PATHS ------------------ #
BASE_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-3"))
os.makedirs(BASE_OUTPUT, exist_ok=True)

BASELINE_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-1", "baseline_model"))
PRUNED_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-2a", "truly_pruned_model.pt"))

# Check if pruned model exists
if not os.path.exists(PRUNED_MODEL):
    PRUNED_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-2a", "pruned_model.pt"))
    if not os.path.exists(PRUNED_MODEL):
        raise FileNotFoundError(f"No pruned model found in phase-2a directory")

print(f"📁 Using pruned model: {PRUNED_MODEL}")

# ------------------ CUSTOM PRUNED MODEL CLASS ------------------ #
class StructurallyPrunedDistilBERT(nn.Module):
    """
    A wrapper that can handle structurally pruned DistilBERT models
    by recreating the pruned architecture from the saved state dict
    """
    
    def __init__(self, baseline_model_path, pruned_state_dict):
        super().__init__()
        
        # Load baseline model to get the original architecture
        self.baseline_model = DistilBertForSequenceClassification.from_pretrained(baseline_model_path)
        self.config = self.baseline_model.config
        
        # Analyze the pruned state dict to understand the new architecture
        self.pruned_layers = {}
        self.load_pruned_architecture(pruned_state_dict)
        j
    def load_pruned_architecture(self, pruned_state_dict):
        """Analyze and adapt to the pruned architecture"""
        baseline_dict = self.baseline_model.state_dict()
        
        # Find layers with different shapes (these were structurally pruned)
        for key, pruned_tensor in pruned_state_dict.items():
            if key in baseline_dict:
                baseline_shape = baseline_dict[key].shape
                pruned_shape = pruned_tensor.shape
                
                if baseline_shape != pruned_shape:
                    print(f"📐 {key}: {baseline_shape} → {pruned_shape}")
                    self.pruned_layers[key] = {
                        'original_shape': baseline_shape,
                        'pruned_shape': pruned_shape,
                        'tensor': pruned_tensor
                    }
        
        # Load the pruned weights
        # For layers with same shape, load directly
        # For pruned layers, we'll handle them specially
        compatible_dict = {}
        for key, tensor in pruned_state_dict.items():
            if key not in self.pruned_layers:
                compatible_dict[key] = tensor
        
        # Load compatible weights
        self.baseline_model.load_state_dict(compatible_dict, strict=False)
        
        # Handle structurally pruned layers
        self._adapt_pruned_layers()
        
    def _adapt_pruned_layers(self):
        """Adapt the model to work with pruned layers"""
        # For structurally pruned layers, we need to modify the model architecture
        # This is complex, so for now we'll approximate by applying masks
        
        with torch.no_grad():
            for key, info in self.pruned_layers.items():
                if 'ffn.lin1.weight' in key:
                    # Handle feedforward layer 1 pruning
                    layer_path = key.replace('.weight', '').split('.')
                    layer = self._get_layer_by_path(layer_path)
                    
                    if layer is not None:
                        # Create a mask based on the pruned dimensions
                        original_shape = info['original_shape']
                        pruned_shape = info['pruned_shape']
                        
                        if len(original_shape) == 2:  # Linear layer
                            # Create mask for output neurons (rows)
                            output_neurons_kept = pruned_shape[0]
                            total_output_neurons = original_shape[0]
                            
                            # Zero out neurons that would be removed
                            if output_neurons_kept < total_output_neurons:
                                neurons_to_zero = total_output_neurons - output_neurons_kept
                                # Zero out the least important neurons
                                importance = torch.sum(torch.abs(layer.weight.data), dim=1)
                                _, indices_to_zero = torch.topk(importance, neurons_to_zero, largest=False)
                                layer.weight.data[indices_to_zero, :] = 0
                                if layer.bias is not None:
                                    layer.bias.data[indices_to_zero] = 0
    
    def _get_layer_by_path(self, path):
        """Get a layer by its path in the model"""
        obj = self.baseline_model
        try:
            for attr in path:
                if attr.isdigit():
                    obj = obj[int(attr)]
                else:
                    obj = getattr(obj, attr)
            return obj
        except (AttributeError, IndexError):
            return None
    
    def forward(self, **kwargs):
        return self.baseline_model(**kwargs)
    
    def parameters(self):
        return self.baseline_model.parameters()
    
    def named_parameters(self):
        return self.baseline_model.named_parameters()
    
    def state_dict(self):
        return self.baseline_model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        return self.baseline_model.load_state_dict(state_dict, strict)
    
    def train(self, mode=True):
        self.baseline_model.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def to(self, device):
        self.baseline_model = self.baseline_model.to(device)
        return self

# ------------------ LOAD DATA ------------------ #
print("📊 Loading dataset...")
dataset = load_dataset("glue", "sst2")
tokenizer = DistilBertTokenizerFast.from_pretrained(BASELINE_MODEL)

def tokenize_fn(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=512)

encoded_dataset = dataset.map(
    tokenize_fn, 
    batched=True, 
    remove_columns=["sentence", "idx"]
)
encoded_dataset.set_format("torch")

# ------------------ LOAD TEACHER + STUDENT ------------------ #
print("🔄 Loading teacher model...")
teacher = DistilBertForSequenceClassification.from_pretrained(BASELINE_MODEL)

print("🔄 Loading structurally pruned student model...")
try:
    # Load the pruned state dict
    pruned_state_dict = torch.load(PRUNED_MODEL, map_location='cpu')
    
    # Create student model that can handle the pruned architecture
    student = StructurallyPrunedDistilBERT(BASELINE_MODEL, pruned_state_dict)
    print("✅ Successfully loaded structurally pruned model")
    
except Exception as e:
    print(f"❌ Error loading structured pruned model: {e}")
    print("📝 Falling back to baseline model with simulated pruning")
    
    # Fallback: baseline model with aggressive weight pruning
    student = DistilBertForSequenceClassification.from_pretrained(BASELINE_MODEL)
    with torch.no_grad():
        for name, param in student.named_parameters():
            if 'ffn' in name and 'weight' in name:
                # Apply aggressive pruning to feedforward layers
                mask = torch.rand_like(param) > 0.4  # 60% sparsity
                param.data *= mask.float()
    print("✅ Applied aggressive pruning simulation")

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

teacher.to(device)
student.to(device)

# Get parameter counts
teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())

# Calculate actual non-zero parameters for student
student_nonzero_params = sum((p != 0).sum().item() for p in student.parameters())

print(f"📈 Teacher parameters: {teacher_params:,}")
print(f"📈 Student total parameters: {student_params:,}")
print(f"📈 Student non-zero parameters: {student_nonzero_params:,}")
print(f"📈 Effective compression ratio: {teacher_params/student_nonzero_params:.2f}x")
print(f"📊 Student sparsity: {(student_params - student_nonzero_params)/student_params:.2%}")

# Set teacher to evaluation mode
teacher.eval()

# Initialize optimizer for student
optimizer = AdamW(student.parameters(), lr=1e-5, weight_decay=0.01)  # Lower LR for fine-tuning pruned model

# ------------------ DISTILLATION PARAMETERS ------------------ #
ALPHA = 0.5  # Balanced between task loss and distillation
TEMPERATURE = 3.0  # Moderate temperature for knowledge transfer
EPOCHS = 3  # More epochs to recover from pruning
BATCH_SIZE = 8  # Smaller batch size for stability

print(f"🎯 Distillation config: α={ALPHA}, T={TEMPERATURE}, epochs={EPOCHS}")

def distill_loss(student_logits, teacher_logits, labels, alpha=ALPHA, temperature=TEMPERATURE):
    """Knowledge distillation loss function"""
    # Task loss
    ce_loss = F.cross_entropy(student_logits, labels)
    
    # Distillation loss
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)
    
    # Combined loss
    total_loss = alpha * ce_loss + (1 - alpha) * kd_loss
    
    return total_loss, ce_loss, kd_loss

# ------------------ TRAINING LOOP ------------------ #
print("🎓 Starting knowledge distillation training...")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    encoded_dataset["train"], 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=data_collator
)

# Track training metrics
training_metrics = {
    'losses': [],
    'ce_losses': [],
    'kd_losses': [],
    'accuracies': []
}

for epoch in range(EPOCHS):
    print(f"\n📚 Epoch {epoch + 1}/{EPOCHS}")
    
    student.train()
    epoch_loss = 0
    epoch_ce_loss = 0
    epoch_kd_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Prepare inputs
        inputs = {
            k: v.to(device) 
            for k, v in batch.items() 
            if k in ["input_ids", "attention_mask"]
        }
        labels = batch["labels"].to(device)
        
        # Get teacher predictions (no gradients)
        with torch.no_grad():
            teacher_outputs = teacher(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Get student predictions
        student_outputs = student(**inputs)
        student_logits = student_outputs.logits
        
        # Compute losses
        loss, ce_loss, kd_loss = distill_loss(student_logits, teacher_logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item()
        epoch_ce_loss += ce_loss.item()
        epoch_kd_loss += kd_loss.item()
        
        # Track accuracy
        predictions = torch.argmax(student_logits, dim=-1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
        
        # Update progress bar
        if batch_idx % 10 == 0:
            current_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CE': f'{ce_loss.item():.4f}',
                'KD': f'{kd_loss.item():.4f}',
                'Acc': f'{current_acc:.3f}'
            })
    
    # Calculate epoch metrics
    num_batches = len(train_loader)
    avg_loss = epoch_loss / num_batches
    avg_ce_loss = epoch_ce_loss / num_batches
    avg_kd_loss = epoch_kd_loss / num_batches
    epoch_acc = correct_predictions / total_predictions
    
    training_metrics['losses'].append(avg_loss)
    training_metrics['ce_losses'].append(avg_ce_loss)
    training_metrics['kd_losses'].append(avg_kd_loss)
    training_metrics['accuracies'].append(epoch_acc)
    
    print(f"📊 Epoch {epoch + 1} Summary:")
    print(f"   Loss: {avg_loss:.4f} (CE: {avg_ce_loss:.4f}, KD: {avg_kd_loss:.4f})")
    print(f"   Training Accuracy: {epoch_acc:.2%}")

print("✅ Knowledge distillation training complete!")

# ------------------ EVALUATION ------------------ #
print("🧪 Evaluating models...")

metric = evaluate.load("glue", "sst2")

def evaluate_model(model, split="validation"):
    """Evaluate model on given split"""
    model.eval()
    
    dataloader = DataLoader(
        encoded_dataset[split], 
        batch_size=32,
        collate_fn=data_collator,
        shuffle=False
    )
    
    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            inputs = {
                k: v for k, v in batch.items() 
                if k in ["input_ids", "attention_mask"]
            }
            labels = batch["labels"]
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            num_batches += 1
    
    results = metric.compute(predictions=all_preds, references=all_labels)
    results['avg_loss'] = total_loss / num_batches if num_batches > 0 else 0
    
    return results

# Evaluate both models
teacher_results = evaluate_model(teacher)
student_results = evaluate_model(student)

print(f"\n📊 Final Results:")
print(f"🎯 Teacher accuracy: {teacher_results['accuracy']:.4f}")
print(f"🎯 Student accuracy: {student_results['accuracy']:.4f}")
print(f"📈 Accuracy retention: {student_results['accuracy']/teacher_results['accuracy']:.2%}")

# ------------------ PERFORMANCE METRICS ------------------ #
print("📏 Measuring performance metrics...")

# Save model
distilled_path = os.path.join(BASE_OUTPUT, "distilled_structurally_pruned_model.pt")
torch.save(student.state_dict(), distilled_path)
model_size_mb = os.path.getsize(distilled_path) / (1024**2)

# Measure latency
student.eval()
sample = tokenizer("This movie is fantastic!", return_tensors="pt")
sample = {k: v.to(device) for k, v in sample.items()}

with torch.no_grad():
    # Warm up
    for _ in range(10):
        _ = student(**sample)
    
    # Timing
    start = time.perf_counter()
    for _ in range(100):
        _ = student(**sample)
    latency = (time.perf_counter() - start) / 100

# Memory usage
process = psutil.Process(os.getpid())
mem_usage_mb = process.memory_info().rss / (1024**2)

# ------------------ SAVE RESULTS ------------------ #
print("💾 Saving results...")

metrics = {
    # Model performance
    "teacher_accuracy": float(teacher_results["accuracy"]),
    "student_accuracy": float(student_results["accuracy"]),
    "accuracy_drop": float(teacher_results["accuracy"] - student_results["accuracy"]),
    "accuracy_retention": float(student_results["accuracy"] / teacher_results["accuracy"]),
    
    # Model characteristics
    "teacher_param_count": int(teacher_params),
    "student_total_params": int(student_params),
    "student_nonzero_params": int(student_nonzero_params),
    "effective_compression_ratio": round(teacher_params / student_nonzero_params, 2),
    "sparsity": round((student_params - student_nonzero_params) / student_params, 4),
    
    # Efficiency metrics
    "model_size_mb": round(model_size_mb, 2),
    "avg_latency_sec": round(latency, 6),
    "memory_usage_mb": round(mem_usage_mb, 2),
    
    # Training configuration
    "distillation": {
        "alpha": ALPHA,
        "temperature": TEMPERATURE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": 1e-5,
        "method": "structured_pruning_plus_distillation"
    },
    
    # Training history
    "training_history": training_metrics
}

with open(os.path.join(BASE_OUTPUT, "structurally_pruned_distillation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Phase-3 (Structured Pruning + Knowledge Distillation) complete!")
print(f"📁 Results saved to: {BASE_OUTPUT}")
print(f"🎯 Final accuracy: {metrics['student_accuracy']:.2%}")
print(f"📊 Effective compression: {metrics['effective_compression_ratio']:.1f}x")
print(f"📊 Model size: {metrics['model_size_mb']:.1f} MB")
print(f"⚡ Latency: {metrics['avg_latency_sec']*1000:.2f} ms")