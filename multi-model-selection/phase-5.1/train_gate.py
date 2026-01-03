# file: train_gate.py

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm
import random
import os

# Requires: pip install datasets
from datasets import load_dataset 

from trainable_moe import TrainableEnergyAwareMoE

def train_gating_network(moe_system, dataset, epochs=5, lr=1e-4, energy_penalty_lambda=0.05):
    """
    Trains the gate using Differentiable Routing and a Custom Loss Function.
    
    Loss = Task_Loss + Lambda * ReLU(Expected_Cost - Available_Energy)
    
    Note: A lower lambda (e.g., 0.05) encourages the model to pick the best expert 
    it can afford, rather than being overly conservative.
    """
    print(f"\n🏋️ Training Gate (Soft-Routing) with Lambda={energy_penalty_lambda}...")
    
    gate = moe_system.gating_network
    gate.train()
    
    # Freeze experts
    for expert in moe_system.experts.values():
        for param in expert.parameters():
            param.requires_grad = False
    
    optimizer = optim.AdamW(gate.parameters(), lr=lr)
    task_loss_fn = nn.CrossEntropyLoss()
    
    # Pre-fetch expert costs
    expert_costs = moe_system.expert_costs_tensor

    for epoch in range(epochs):
        total_loss = 0
        total_task_loss = 0
        total_energy_penalty = 0
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        # Training Loop
        # We use a subset or the full dataset depending on what was passed
        progress_bar = tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}")
        
        for text, true_label_idx in progress_bar:
            # 1. Prepare Inputs
            inputs = moe_system.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(moe_system.device)
            target = torch.tensor([true_label_idx], device=moe_system.device)
            
            # Simulate random energy level
            available_energy = random.uniform(0, moe_system.max_energy_capacity)
            energy_input = torch.tensor([[available_energy]], dtype=torch.float32, device=moe_system.device)
            
            # Get text embedding
            with torch.no_grad():
                text_emb = moe_system._get_text_embedding(inputs)

            optimizer.zero_grad()

            # 2. Forward Pass: Gate
            gate_logits = gate(text_emb, energy_input)
            gate_probs = F.softmax(gate_logits, dim=1) 
            
            # 3. Soft Routing (Run all experts)
            expert_logits_list = []
            with torch.no_grad():
                for name in moe_system.expert_names:
                    out = moe_system.experts[name](**inputs)
                    expert_logits_list.append(out.logits)
            
            stacked_expert_logits = torch.stack(expert_logits_list, dim=1)
            
            # 4. Weighted Mixture
            weighted_logits = torch.sum(stacked_expert_logits * gate_probs.unsqueeze(-1), dim=1)

            # 5. Custom Loss
            task_l = task_loss_fn(weighted_logits, target)
            
            expected_cost = torch.sum(gate_probs * expert_costs)
            energy_violation = F.relu(expected_cost - available_energy)
            penalty = energy_penalty_lambda * energy_violation
            
            loss = task_l + penalty
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_task_loss += task_l.item()
            total_energy_penalty += penalty.item()
            
            # Update progress bar occasionally
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Pnlty": f"{penalty.item():.4f}"})

        avg_loss = total_loss / len(dataset)
        print(f"   Epoch {epoch+1}: Avg Loss: {avg_loss:.4f} (Task: {total_task_loss/len(dataset):.4f} | Penalty: {total_energy_penalty/len(dataset):.4f})")

    gate.eval()
    print("✅ Training complete.")

    torch.save(moe_system.gating_network.state_dict(), "gate_trained.pt")
    print("💾 Gating network saved to gate_trained.pt")
    
if __name__ == "__main__":
    # 1. Initialize the MoE System
    moe_system = TrainableEnergyAwareMoE(max_energy_capacity=50.0, initial_energy=16.0)

    # 2. Load IMDb Dataset
    print("\n📚 Loading IMDb dataset...")
    try:
        # Load dataset
        imdb = load_dataset("imdb")
        
        # We'll use a subset of train data for faster demonstration
        # Shuffle and take 500 samples
        train_subset = imdb['train'].shuffle(seed=42).select(range(1000))
        
        # Convert to list of (text, label)
        # IMDb labels: 0 (neg), 1 (pos) match our model
        gate_training_data = [(sample['text'], sample['label']) for sample in train_subset]
        print(f"✅ Loaded {len(gate_training_data)} samples from IMDb.")

    except Exception as e:
        print(f"⚠️ Error loading IMDb: {e}")
        print("   Falling back to dummy data.")
        gate_training_data = [
            ("This movie is a masterpiece, a true work of art.", 1),
            ("I was completely bored from start to finish.", 0)
        ] * 50

    # 3. Train the gating network
    # We use a lower lambda to ensure the model isn't too afraid to use expensive experts
    train_gating_network(moe_system, gate_training_data, epochs=5, energy_penalty_lambda=0.2)

    # 4. Simulation
    print("\n" + "="*80)
    print("🚀 Starting Inference Simulation with the TRAINED Gating Network")
    print("="*80)
    
    sentence = "This film was not without its merits, a complex and nuanced piece."
    
    simulation_steps = [
        {"harvested": 0.0,  "task": "High initial energy"},  # Energy: 16 -> Gate routes to 'baseline' (cost 15), rem: 1
        {"harvested": 10.0, "task": "Medium energy"},         # Energy: 1+10=11 -> Gate routes to 'q_baseline' (cost 10), rem: 1
        {"harvested": 8.0,  "task": "Low-medium energy"},     # Energy: 1+8=9 -> Gate routes to 'pruned' (cost 8), rem: 1
        {"harvested": 5.0,  "task": "Low energy"},            # Energy: 1+5=6 -> Gate routes to 'pruned_quantized' (cost 5), rem: 1
        {"harvested": 2.0,  "task": "Insufficient energy"},   # Energy: 1+2=3 -> Gate returns 'None'
    ]
    
    # Reset energy for simulation
    moe_system.current_energy = 16.0

    for step in simulation_steps:
        print(f"\n--- {step['task']} ---")
        print(f"    ⚡️ Energy harvested: {step['harvested']:.1f} units")
        
        result = moe_system.predict(sentence, harvested_energy=step['harvested'])
        
        print(f"    🔋 Energy State: {result['energy_before']:.1f} -> {result['energy_after']:.1f} (Cost: {result['energy_cost']:.1f})")
        print(f"    🤖 Expert Used:  {result['expert_used']} ({result['status']})")