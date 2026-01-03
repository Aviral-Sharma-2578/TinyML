import torch
import random
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

# Import your existing system
from trainable_moe import TrainableEnergyAwareMoE

# --- Configuration ---
GATE_MODEL_PATH = "gate_trained.pt"

def load_trained_gate(moe_system, path):
    """Overwrites the random init gate with the trained weights."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Trained gate file '{path}' not found. Run train_gate.py first.")
    
    print(f"📥 Loading trained gate weights from {path}...")
    state_dict = torch.load(path, map_location=moe_system.device)
    moe_system.gating_network.load_state_dict(state_dict)
    moe_system.gating_network.eval()
    print("✅ Gate loaded and set to Eval mode.")

def evaluate_system(moe_system, test_data):
    print(f"\n📊 Evaluating on {len(test_data)} test samples...")
    print("   (Simulating random energy levels for each query to test adaptability)")
    
    predictions = []
    true_labels = []
    energy_costs = []
    expert_usage = {name: 0 for name in moe_system.expert_names}
    skipped = 0
    
    # Iterate through test data
    for text, label in tqdm(test_data, desc="Testing"):
        # 1. Simulate a random energy state for this user/moment
        # This tests if the gate makes the right choice for *that* specific energy level
        simulated_energy = random.uniform(0, moe_system.max_energy_capacity)
        moe_system.current_energy = simulated_energy
        
        # 2. Run Inference
        # We pass harvested_energy=0 because we manually set the "current" state above
        result = moe_system.predict(text, harvested_energy=0.0)
        
        if result['status'] == 'Success':
            # Map 'POSITIVE'/'NEGATIVE' back to 1/0
            pred_label = 1 if result['prediction'] == 'POSITIVE' else 0
            
            predictions.append(pred_label)
            true_labels.append(label)
            energy_costs.append(result['energy_cost'])
            expert_usage[result['expert_used']] += 1
        else:
            # If the gate (correctly or incorrectly) decided nothing was affordable
            skipped += 1
    
    # --- Report Generation ---
    if not predictions:
        print("\n❌ All queries were skipped. Check if your simulated energy range matches expert costs.")
        return

    acc = accuracy_score(true_labels, predictions)
    avg_energy = np.mean(energy_costs)
    total_processed = len(predictions)
    
    print("\n" + "="*50)
    print(f"🚀 FINAL EVALUATION REPORT")
    print("="*50)
    print(f"✅ Accuracy (on processed):   {acc:.4f} ({acc*100:.2f}%)")
    print(f"⚡ Avg Energy Cost/Query:     {avg_energy:.2f} units")
    print(f"⏭️  Skipped (Low Energy):      {skipped} ({skipped/len(test_data)*100:.1f}%)")
    print("-" * 50)
    print("🤖 Expert Selection Distribution:")
    for name in moe_system.expert_names:
        count = expert_usage.get(name, 0)
        percent = (count / total_processed * 100) if total_processed > 0 else 0
        print(f"   - {name:<20}: {count:4d} ({percent:5.1f}%)")
    print("="*50)

if __name__ == "__main__":
    # 1. Initialize System (Loads Experts)
    # We set initial energy high, but the evaluator overrides it per sample anyway
    moe = TrainableEnergyAwareMoE(max_energy_capacity=50.0, initial_energy=50.0)
    
    # 2. Load the Trained Brain
    try:
        load_trained_gate(moe, GATE_MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        exit()

    # 3. Prepare Real Test Data
    print("\n📚 Loading IMDb Test Data...")
    try:
        dataset = load_dataset("imdb")
        # Take a random subset of 500 test samples
        test_subset = dataset['test'].shuffle(seed=1337).select(range(500))
        test_data = [(x['text'], x['label']) for x in test_subset]
    except Exception as e:
        print(f"⚠️ Could not load IMDb ({e}). Using dummy data.")
        test_data = [
            ("This movie was fantastic and thrilling.", 1), 
            ("Terrible plot and bad acting.", 0)
        ] * 50

    # 4. Run Assessment
    evaluate_system(moe, test_data)