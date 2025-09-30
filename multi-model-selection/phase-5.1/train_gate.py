# file: train_gate.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random

# Import the MoE system which contains the models and tokenizer
from trainable_moe import TrainableEnergyAwareMoE

def generate_gate_training_data(moe_system, dataset):
    """
    Generates pseudo-labels for training the gating network.
    For each data point, it finds the cheapest expert that gets the correct answer.
    """
    print("\n🔬 Generating pseudo-labels for the gating network...")
    training_data = []
    
    expert_names_sorted_by_cost = sorted(
        moe_system.expert_names,
        key=lambda name: moe_system.expert_energy_costs[name]
    )

    for text, true_label_idx in tqdm(dataset, desc="Processing dataset"):
        inputs = moe_system.tokenizer(text, return_tensors="pt").to(moe_system.device)
        
        optimal_expert_idx = -1

        # Find the cheapest expert that is correct
        for expert_name in expert_names_sorted_by_cost:
            expert_model = moe_system.experts[expert_name]
            with torch.no_grad():
                outputs = expert_model(**inputs)
                prediction_idx = torch.argmax(outputs.logits, dim=1).item()
            
            if prediction_idx == true_label_idx:
                # This is the cheapest correct expert, we've found our target
                optimal_expert_idx = moe_system.expert_names.index(expert_name)
                break # Move to the next data sample
        
        # If no expert got it right, we can either skip it or assign the best-performing one
        # For simplicity, we'll only add data where at least one expert was correct.
        if optimal_expert_idx != -1:
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
        # In a real scenario, you'd use a custom collate_fn for batching.
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
    # In a real scenario, this would be your validation or training set from a file.
    # Labels: 0 for NEGATIVE, 1 for POSITIVE
    dummy_sentiment_dataset = [
        ("This movie is a masterpiece, a true work of art.", 1),
        ("I was completely bored from start to finish.", 0),
        ("A decent attempt, but it ultimately falls flat.", 0),
        ("Absolutely brilliant! I would recommend this to everyone.", 1),
        ("It was just okay, nothing special.", 0),
        ("A cinematic triumph of the highest order!", 1),
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