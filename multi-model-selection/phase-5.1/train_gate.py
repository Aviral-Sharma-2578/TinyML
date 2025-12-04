# file: train_gate.py

import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random

# Import the MoE system which contains the models and tokenizer
from trainable_moe import TrainableEnergyAwareMoE

def generate_gate_training_data(moe_system, dataset, confidence_threshold=0.90, baseline_premium_threshold=0.98):
    """
    Generates pseudo-labels, now with a special rule to prefer the baseline
    expert when it is exceptionally confident.
    """
    print(f"\n🔬 Generating pseudo-labels with confidence_threshold={confidence_threshold} and baseline_premium_threshold={baseline_premium_threshold}...")
    training_data = []
    
    expert_names_sorted_by_cost = sorted(
        moe_system.expert_names,
        key=lambda name: moe_system.expert_energy_costs[name]
    )

    for text, true_label_idx in tqdm(dataset, desc="Processing dataset"):
        inputs = moe_system.tokenizer(text, return_tensors="pt").to(moe_system.device)
        
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

        # --- NEW HEURISTIC TO CHOOSE THE OPTIMAL EXPERT ---
        if not confidently_correct_experts:
            print(f"\nInput: '{text[:30]}...' -> No expert was confidently correct. Skipping.")
            continue
        
        # Rule: If the baseline model is a valid choice and is extremely confident,
        #       prefer it to teach the gate the value of high quality.
        baseline_choice = next((exp for exp in confidently_correct_experts if exp['name'] == 'baseline'), None)
        
        if baseline_choice and baseline_choice['confidence'] > baseline_premium_threshold:
            optimal_expert_name = 'baseline'
        else:
            # Otherwise, fall back to the cheapest valid expert (the first in the sorted list)
            optimal_expert_name = confidently_correct_experts[0]['name']

        optimal_expert_idx = moe_system.expert_names.index(optimal_expert_name)
        confidence = next(exp['confidence'] for exp in confidently_correct_experts if exp['name'] == optimal_expert_name)
        print(f"\nInput: '{text[:30]}...' -> Optimal Expert: {optimal_expert_name} (Confidence: {confidence:.2f})")
        training_data.append((text, optimal_expert_idx))

    print(f"✅ Generated {len(training_data)} training samples for the gate.")
    return training_data

def train_gating_network(moe_system, training_data, epochs=5, lr=1e-4):
    """Trains the gating network on the generated pseudo-labeled data."""
    print("\n🏋️ Training the gating network...")
    gate = moe_system.gating_network
    gate.train() # Set to training mode

    optimizer = optim.AdamW(gate.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Create a TensorDataset for easier batching
    texts, labels = zip(*training_data)
    target_indices = torch.tensor(labels, dtype=torch.long)
    
    for epoch in range(epochs):
        total_loss = 0
        
        # We process one by one since tokenization padding can be complex in a simple example
        # In a real scenario, we'd use a custom collate_fn for batching.
        for i in tqdm(range(len(texts)), desc=f"Epoch {epoch+1}/{epochs}"):
            text = texts[i]
            target_expert = target_indices[i].unsqueeze(0).to(moe_system.device) # Shape [1]

            inputs = moe_system.tokenizer(text, return_tensors="pt").to(moe_system.device)
            text_embedding = moe_system._get_text_embedding(inputs)
            
            # --- Simulate a random energy level for each training step ---
            # This teaches the gate to factor energy into its decision.
            simulated_energy = random.uniform(0, moe_system.max_energy_capacity)
            energy_tensor = torch.tensor([[simulated_energy]], dtype=torch.float32, device=moe_system.device)

            optimizer.zero_grad()
            
            # Forward pass
            logits = gate(text_embedding, energy_tensor) # Shape [1, num_experts]
            
            # Calculate loss
            loss = loss_fn(logits, target_expert)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(texts)
        print(f"   Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    gate.eval() # Set back to evaluation mode
    print("✅ Gating network training complete.")


if __name__ == "__main__":
    # 1. Initialize the MoE System (this loads all the experts)
    moe_system = TrainableEnergyAwareMoE(max_energy_capacity=50.0, initial_energy=16.0)

    # 2. Create a dummy labeled dataset for generating gate training data
    # In a real scenario, this would be our validation or training set from a file.
    # Labels: 0 for NEGATIVE, 1 for POSITIVE
    dummy_sentiment_dataset = [
        ("This movie is a masterpiece, a true work of art.", 1),
        ("I was completely bored from start to finish.", 0),

        # Nuance/Sarcasm: Cheaper models might miss the negative sentiment
        ("I can't believe I sat through the entire thing. Truly a cinematic experience of all time.", 0),
        
        # Complex Sentence: Requires understanding the connecting phrase "despite"
        ("Despite a fantastic performance by the lead actor, the film's plot was convoluted and unsatisfying.", 0),
        
        # Faint Praise: Harder to classify as strictly positive or negative
        ("It was an interesting film, certainly not one I would forget easily, though perhaps not for the right reasons.", 0),

        # A clear positive case that all models should get
        ("Absolutely brilliant! I would recommend this to everyone.", 1),
    ]
    
    # 3. Generate the training data for the gate
    gate_training_data = generate_gate_training_data(moe_system, dummy_sentiment_dataset)

    # 4. Train the gating network
    if gate_training_data:
        train_gating_network(moe_system, gate_training_data, epochs=3)
        # Optional: Save the trained gating network's state dict
        # torch.save(moe_system.gating_network.state_dict(), "trained_gating_network.pt")
    else:
        print("No training data was generated. Skipping gate training.")

    # 5. Run a simulation with the TRAINED gate
    print("\n" + "="*80)
    print("🚀 Starting Inference Simulation with the TRAINED Gating Network")
    print("="*80)
    
    sentence = "This film was not without its merits, a complex and nuanced piece."
    
    simulation_steps = [
        {"harvested": 0.0, "task": "High initial energy"},  # Energy: 16 -> Gate should pick a powerful model
        {"harvested": 10.0,"task": "Medium energy"},        # Energy: 1+10=11 -> Gate might pick a quantized model
        {"harvested": 2.0, "task": "Insufficient energy"},  # Energy now very low -> Gate should pick cheapest or skip
    ]
    
    # Reset energy for simulation
    moe_system.current_energy = 16.0

    for i, step in enumerate(simulation_steps):
        print(f"\n--- Step {i+1}: {step['task']} ---")
        print(f"    ⚡️ Energy harvested: {step['harvested']:.1f} units")
        
        result = moe_system.predict(sentence, harvested_energy=step['harvested'])
        
        print(f"    🔋 Energy State: {result['energy_before']:.1f} -> {result['energy_after']:.1f} (Cost: {result['energy_cost']:.1f})")
        print(f"    🤖 Expert Used:  {result['expert_used']} ({result['status']})")