# file: train_gate.py

import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random

# New import for real data
from datasets import load_dataset

# Import the MoE system
from trainable_moe import TrainableEnergyAwareMoE

class GateDataset(Dataset):
    """
    Custom Dataset to handle the text and label pairs for the Gate.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_real_data(num_samples=1000):
    """
    Loads the IMDb dataset and selects a balanced subset to save time 
    during the expensive label-generation phase.
    """
    print(f"\n📚 Loading IMDb dataset (subset: {num_samples} samples)...")
    try:
        # Load IMDb dataset from Hugging Face
        dataset = load_dataset("imdb", split="train")
        
        # Shuffle and select a subset to keep label generation fast
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
        
        print(f"✅ Loaded {len(dataset)} real movie reviews.")
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Falling back to dummy data...")
        return None

def generate_gate_training_data(moe_system, dataset, confidence_threshold=0.90, baseline_premium_threshold=0.98):
    """
    Generates pseudo-labels using real dataset items (dicts with 'text' and 'label').
    """
    print(f"\n🔬 Generating pseudo-labels (Teacher-Student approach)...")
    training_data = []
    
    expert_names_sorted_by_cost = sorted(
        moe_system.expert_names,
        key=lambda name: moe_system.expert_energy_costs[name]
    )

    # Iterate through the real dataset
    # Note: 'dataset' here is a Hugging Face dataset object, allowing iteration
    for i, item in enumerate(tqdm(dataset, desc="Running Experts on Data")):
        text = item['text']
        true_label_idx = item['label']
        
        # Truncate text to 512 tokens to prevent DistilBert errors if reviews are huge
        inputs = moe_system.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(moe_system.device)
        
        # Find all experts that are confidently correct
        confidently_correct_experts = []
        for expert_name in expert_names_sorted_by_cost:
            expert_model = moe_system.experts[expert_name]
            with torch.no_grad():
                probabilities = F.softmax(expert_model(**inputs).logits, dim=1)
                confidence_score = probabilities.max().item()
                prediction_idx = probabilities.argmax().item()
            
            if prediction_idx == true_label_idx and confidence_score >= confidence_threshold:
                confidently_correct_experts.append({"name": expert_name, "confidence": confidence_score})

        if not confidently_correct_experts:
            # Skip data points where no expert is confident (too hard/ambiguous)
            continue
        
        # --- Optimal Expert Selection Heuristic ---
        baseline_choice = next((exp for exp in confidently_correct_experts if exp['name'] == 'baseline'), None)
        
        if baseline_choice and baseline_choice['confidence'] > baseline_premium_threshold:
            # If baseline is overwhelmingly confident, teach gate to use it (Quality bias)
            optimal_expert_name = 'baseline'
        else:
            # Otherwise, choose the cheapest valid expert (Efficiency bias)
            optimal_expert_name = confidently_correct_experts[0]['name']

        optimal_expert_idx = moe_system.expert_names.index(optimal_expert_name)
        
        # Store text and the index of the expert the gate *should* choose
        training_data.append((text, optimal_expert_idx))

    print(f"✅ Generated {len(training_data)} usable training samples from {len(dataset)} raw inputs.")
    return training_data

def train_gating_network(moe_system, training_data, epochs=5, lr=5e-4, batch_size=16):
    """
    Trains the gating network using Mini-Batch Gradient Descent.
    """
    print(f"\n🏋️ Training the gating network (Batch Size: {batch_size})...")
    gate = moe_system.gating_network
    gate.train()

    optimizer = optim.AdamW(gate.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Wrap data in a DataLoader
    dataset = GateDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_texts, batch_labels in pbar:
            
            # Tokenize the batch of texts
            inputs = moe_system.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(moe_system.device)
            
            # Extract features (using the baseline embedding layer)
            text_embeddings = moe_system._get_text_embedding(inputs)
            
            # Generate random energy levels for this batch
            # We want the gate to learn to adapt to ANY energy level
            simulated_energy = torch.rand((len(batch_texts), 1), device=moe_system.device) * moe_system.max_energy_capacity
            
            target_experts = batch_labels.to(moe_system.device)

            optimizer.zero_grad()
            
            # Forward pass
            logits = gate(text_embeddings, simulated_energy) 
            
            # Calculate loss
            loss = loss_fn(logits, target_experts)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            batches += 1
            pbar.set_postfix({'loss': total_loss / batches})

        avg_loss = total_loss / batches
        print(f"   Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    gate.eval()
    print("✅ Gating network training complete.")

if __name__ == "__main__":
    # 1. Initialize System
    moe_system = TrainableEnergyAwareMoE(max_energy_capacity=50.0, initial_energy=16.0)

    # 2. Load Real Data
    # We limit to 1000 samples because generating labels requires running 
    # ALL experts on EVERY sample, which is computationally expensive.
    raw_dataset = load_real_data(num_samples=1000)
    
    if raw_dataset is None:
        # Fallback if internet/library fails
        raw_dataset = [
            {"text": "This movie is a masterpiece.", "label": 1},
            {"text": "I hated every second of it.", "label": 0},
            {"text": "The plot was confusing but the acting was okay.", "label": 0}
        ] * 10

    # 3. Generate Training Data (The "Teacher" Phase)
    gate_training_data = generate_gate_training_data(moe_system, raw_dataset)

    # 4. Train the Gate (The "Student" Phase)
    if len(gate_training_data) > 0:
        train_gating_network(moe_system, gate_training_data, epochs=5, batch_size=16)
        
        # Save the trained gate
        torch.save(moe_system.gating_network.state_dict(), "trained_gate_imdb.pt")
        print("💾 Saved trained gate to 'trained_gate_imdb.pt'")
    else:
        print("⚠️ No valid training data generated. Try lowering confidence thresholds.")

    # 5. Inference Simulation
    print("\n" + "="*80)
    print("🚀 TEST: Real Inference on a New Review")
    print("="*80)
    
    test_sentence = "The cinematography was dazzling, but the story felt hollow and uninspired."
    
    # Test at different energy levels
    levels = [50.0, 12.0, 4.0] # High, Medium, Low
    
    for level in levels:
        moe_system.current_energy = level
        result = moe_system.predict(test_sentence, harvested_energy=0.0)
        print(f"Energy: {level:04.1f} | Expert: {result['expert_used']:<20} | Pred: {result['prediction']}")