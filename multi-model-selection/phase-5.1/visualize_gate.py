import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import the architecture from your existing file
from trainable_moe import GatingNetwork

# --- Configuration ---
MODEL_PATH = "gate_trained.pt"  # Path to your saved gating network checkpoint
HIDDEN_DIM = 128                # Must match your GatingNetwork definition
EMBEDDING_DIM = 768             # DistilBERT embedding size
NUM_EXPERTS = 4                 # Number of experts in your system
MAX_ENERGY = 50.0               # Max energy to visualize

def visualize_energy_dynamics():
    # 1. Initialize and Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Gating Network from {MODEL_PATH}...")
    
    model = GatingNetwork(embedding_dim=EMBEDDING_DIM, num_experts=NUM_EXPERTS, hidden_dim=HIDDEN_DIM)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"⚠️ Warning: {MODEL_PATH} not found. Initializing with random weights (for demo).")
        print("   (To see real patterns, save your trained gate with: torch.save(gate.state_dict(), 'gate_trained.pt'))")
    
    model.to(device)
    model.eval()

    # 2. Generate Energy Sweep Data
    # We create a range of energy values from 0 to MAX_ENERGY
    energy_values = np.linspace(0, MAX_ENERGY, 200)
    energy_tensor = torch.tensor(energy_values, dtype=torch.float32).unsqueeze(1).to(device) # [200, 1]

    # 3. Extract Internal Activations
    # We specifically want to see what comes out of the Energy Projector (post-ReLU)
    # Architecture: Linear -> Tanh -> Linear -> ReLU
    with torch.no_grad():
        activations = model.energy_projector(energy_tensor).cpu().numpy()
        
    # activations shape is [200, HIDDEN_DIM]. 
    # Transpose to [HIDDEN_DIM, 200] for heatmap (Neurons on Y-axis, Energy on X-axis)
    activations_t = activations.T

    # Apply IEEE paper styling
    with plt.style.context("seaborn-v0_8-paper"):
        
        # Dimensions: 3.5" width (single column), 2.2" height
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        
        # 1. Create Heatmap
        # rasterized=True is CRITICAL for heatmaps in PDFs (prevents huge file sizes)
        # cbar_kws shrinks the colorbar to match the small plot height
        sns.heatmap(activations_t, cmap="viridis", ax=ax, rasterized=True,
                    yticklabels=False, xticklabels=False,
                    cbar_kws={'label': 'Activation', 'shrink': 0.8})
        
        # 2. Formatting X-Axis
        # Reduced ticks from 11 to 5 to prevent overlap in small figure
        num_ticks = 5 
        xticks = np.linspace(0, len(energy_values)-1, num_ticks)
        xticklabels = [f"{e:.0f}" for e in np.linspace(0, MAX_ENERGY, num_ticks)]
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0, fontsize=7)
        
        # 3. Titles and Labels (Simplified for space)
        # Shortened title to fit on one line or two short lines
        ax.set_title("Gating Network: Energy Representation", fontsize=8, pad=4)
        ax.set_xlabel("Input Energy Level", fontsize=7)
        ax.set_ylabel(f"Hidden Neurons (0-{HIDDEN_DIM})", fontsize=7)
        
        # Optional: If you want to show the expert markers, use thin lines
        # expert_costs = [5.0, 8.0, 10.0, 15.0]
        # for cost in expert_costs:
        #     x_idx = (cost / MAX_ENERGY) * len(energy_values)
        #     if 0 <= x_idx < len(energy_values):
        #         ax.axvline(x_idx, color='white', linestyle='--', alpha=0.7, linewidth=0.8)

        # 4. Saving
        plt.tight_layout(pad=0.2)
        output_file = "energy_dynamics_analysis.pdf" # PDF is best for text, heatmap is rasterized inside
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to {output_file}")
        plt.show()

if __name__ == "__main__":
    visualize_energy_dynamics()