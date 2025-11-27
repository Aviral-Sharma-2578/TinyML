# file: trainable_moe.py

import os
import copy
import time
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torchao.quantization import quantize_, Int8WeightOnlyConfig

# --- Configuration ---
# Creates an 'outputs' directory in the parent folder of where the script is located
BASE_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
BASELINE_DIR = os.path.join(BASE_OUTPUT_DIR, "phase-1", "baseline_model")
PRUNED_MODEL_PATH = os.path.join(BASE_OUTPUT_DIR, "phase-2", "pruned_true_reduction.pt")
QUANTIZED_BASELINE_PATH = os.path.join(BASE_OUTPUT_DIR, "phase-3", "quantized_baseline_weight_only_int8.pt")
QUANTIZED_PRUNED_PATH = os.path.join(BASE_OUTPUT_DIR, "phase-3", "quantized_weight_only_int8.pt")


# --- Helper Functions ---

def load_state_dict_safely(model_path, device):
    """Loads a state dictionary, handling potential errors."""
    if not os.path.exists(model_path):
        print(f"   ❌ State dictionary not found at {model_path}. Skipping.")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        return torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"   ❌ Failed to load state_dict from {model_path}: {e}")
        raise e

def create_pruned_model_architecture(state_dict):
    """Creates a model with an architecture matching the pruned state dict."""
    if not os.path.isdir(BASELINE_DIR):
        print(f"   ❌ Baseline model directory not found at {BASELINE_DIR} for architecture creation.")
        raise FileNotFoundError(f"Baseline directory not found: {BASELINE_DIR}")
        
    model = DistilBertForSequenceClassification.from_pretrained(BASELINE_DIR)
    ffn_dims = {}
    for key in state_dict.keys():
        if 'ffn.lin1.weight' in key:
            parts = key.split('.')
            if len(parts) >= 4:
                try:
                    layer_num = int(parts[3])
                    new_dim = state_dict[key].shape[0]
                    ffn_dims[layer_num] = new_dim
                except (ValueError, IndexError):
                    continue
    
    for layer_num, new_dim in ffn_dims.items():
        if layer_num < len(model.distilbert.transformer.layer):
            layer_module = model.distilbert.transformer.layer[layer_num]
            old_lin1, old_lin2 = layer_module.ffn.lin1, layer_module.ffn.lin2
            new_lin1 = torch.nn.Linear(old_lin1.in_features, new_dim, bias=old_lin1.bias is not None)
            new_lin2 = torch.nn.Linear(new_dim, old_lin2.out_features, bias=old_lin2.bias is not None)
            layer_module.ffn.lin1, layer_module.ffn.lin2 = new_lin1, new_lin2
    return model

# --- Trainable Gating Network Definition ---

class GatingNetwork(nn.Module):
    """
    A simple MLP that takes text embeddings and the current energy state to
    predict the most suitable expert.
    """
    def __init__(self, embedding_dim: int, num_experts: int, hidden_dim: int = 128):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, text_embedding: torch.Tensor, energy_level: torch.Tensor) -> torch.Tensor:
        combined_input = torch.cat([text_embedding, energy_level], dim=1)
        logits = self.layer_stack(combined_input)
        return logits

# --- Main MoE System with Trainable Gate ---

class TrainableEnergyAwareMoE:
    """
    Implements an MoE system with a trainable, data-driven gating network.
    """
    def __init__(self, max_energy_capacity=100.0, initial_energy=20.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experts = {}
        self.tokenizer = None
        self.labels = ['NEGATIVE', 'POSITIVE']
        
        self.expert_names = ['baseline', 'quantized_baseline', 'pruned', 'pruned_quantized']
        self.expert_energy_costs = {
            'baseline': 15.0, 'quantized_baseline': 10.0,
            'pruned': 8.0, 'pruned_quantized': 5.0,
        }
        
        self.max_energy_capacity = max_energy_capacity
        self.current_energy = min(initial_energy, max_energy_capacity)

        print("🚀 Initializing Trainable Energy-Aware MoE...")
        print(f"   - Device:           {self.device}")
        self._load_experts()

        self.gating_network = GatingNetwork(
            embedding_dim=768, 
            num_experts=len(self.expert_names)
        ).to(self.device)
        self.gating_network.eval()

        if 'baseline' in self.experts:
            self.text_embedder = self.experts['baseline'].distilbert.embeddings.to(self.device)
            self.text_embedder.eval()
        else:
            print("⚠️ Baseline expert not found. Gate cannot extract text features.")
            self.text_embedder = None

    def _load_experts(self):
        """
        Loads all expert models into memory.
        
        FOR DEMONSTRATION: This function will now create distinct copies of a base
        model and slightly alter their weights to simulate performance differences
        between experts. This is crucial for generating meaningful training data for the gate.
        """
        print("\n1. Loading and creating distinct expert models for demonstration...")
        
        if not os.path.isdir(BASELINE_DIR):
            print("   Baseline directory not found. Using 'distilbert-base-uncased' for all experts.")
            base_model_name = 'distilbert-base-uncased'
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(base_model_name)
        else:
            base_model_name = BASELINE_DIR
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(base_model_name)

        # --- Load the 'baseline' expert ---
        print("   - Loading 'baseline' expert...")
        try:
            baseline_model = DistilBertForSequenceClassification.from_pretrained(base_model_name).to(self.device)
            self.experts['baseline'] = baseline_model
        except Exception as e:
            print(f"   ⚠️ Could not load baseline expert: {e}")
            return # Cannot proceed without a baseline model

        # --- Create simulated experts with performance differences ---
        # We will deepcopy the baseline and add noise to simulate pruning/quantization
        experts_to_simulate = {
            'quantized_baseline': 0.005, # amount of noise to add
            'pruned': 0.01,
            'pruned_quantized': 0.02
        }

        for name, noise_level in experts_to_simulate.items():
            print(f"   - Simulating '{name}' expert...")
            # 1. Create a true, independent copy of the model
            simulated_model = copy.deepcopy(baseline_model)
            
            # 2. Add random noise to its weights to simulate performance degradation
            with torch.no_grad():
                for param in simulated_model.parameters():
                    param.add_(torch.randn(param.size()).to(self.device) * noise_level)
            
            self.experts[name] = simulated_model

        for expert in self.experts.values():
            expert.eval()
            
        print(f"\n✅ {len(self.experts)} distinct expert models are ready.")

    def _get_text_embedding(self, inputs: dict) -> torch.Tensor:
        """Extracts the [CLS] token embedding for the input text."""
        if self.text_embedder is None:
            raise RuntimeError("Text embedder is not available. Cannot generate features for the gate.")
        with torch.no_grad():
            embeddings = self.text_embedder(inputs['input_ids'])
            return embeddings[:, 0, :]

    def predict(self, text: str, harvested_energy: float = 0.0):
        """Performs a full MoE inference cycle using the trainable gate."""
        self.current_energy = min(self.current_energy + harvested_energy, self.max_energy_capacity)
        
        # --- FIX: Add truncation=True and max_length=512 ---
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        # ---------------------------------------------------

        text_embedding = self._get_text_embedding(inputs)
        energy_tensor = torch.tensor([[self.current_energy]], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            gate_logits = self.gating_network(text_embedding, energy_tensor)
        
        for i, expert_name in enumerate(self.expert_names):
            if self.expert_energy_costs.get(expert_name, float('inf')) > self.current_energy:
                gate_logits[0, i] = -torch.inf

        if torch.all(gate_logits == -torch.inf):
            chosen_expert_name = "None"
        else:
            chosen_expert_idx = torch.argmax(gate_logits, dim=1).item()
            chosen_expert_name = self.expert_names[chosen_expert_idx]
        
        if chosen_expert_name == "None":
            return {"prediction": "SKIPPED", "score": 0.0, "expert_used": "None", "latency_sec": 0.0, "energy_before": self.current_energy, "energy_after": self.current_energy, "energy_cost": 0.0, "status": "Insufficient energy"}

        expert_model = self.experts[chosen_expert_name]
        energy_cost = self.expert_energy_costs[chosen_expert_name]
        energy_before_inference = self.current_energy
        
        with torch.no_grad():
            start_time = time.perf_counter()
            outputs = expert_model(**inputs)
            latency = time.perf_counter() - start_time
        
        self.current_energy -= energy_cost
        
        logits = outputs.logits
        scores = torch.softmax(logits, dim=1)
        prediction_idx = torch.argmax(scores, dim=1).item()
        
        return {"prediction": self.labels[prediction_idx], "score": scores[0][prediction_idx].item(), "expert_used": chosen_expert_name, "latency_sec": latency, "energy_before": energy_before_inference, "energy_after": self.current_energy, "energy_cost": energy_cost, "status": "Success"}

# --- Example Usage ---
if __name__ == "__main__":
    
    print("--- Setting up dummy model files for demonstration ---")
    if not os.path.isdir(BASELINE_DIR):
        print(f"Creating dummy directory: {BASELINE_DIR}")
        os.makedirs(BASELINE_DIR, exist_ok=True)
        dummy_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        dummy_model.save_pretrained(BASELINE_DIR)
        DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased').save_pretrained(BASELINE_DIR)

        os.makedirs(os.path.dirname(PRUNED_MODEL_PATH), exist_ok=True)
        torch.save(dummy_model.state_dict(), PRUNED_MODEL_PATH)
        os.makedirs(os.path.dirname(QUANTIZED_BASELINE_PATH), exist_ok=True)
        torch.save(dummy_model.state_dict(), QUANTIZED_BASELINE_PATH)
        os.makedirs(os.path.dirname(QUANTIZED_PRUNED_PATH), exist_ok=True)
        torch.save(dummy_model.state_dict(), QUANTIZED_PRUNED_PATH)
    print("----------------------------------------------------\n")

    moe_system = TrainableEnergyAwareMoE(max_energy_capacity=50.0, initial_energy=16.0)
    
    sentence = "This movie is a masterpiece, a true work of art."
    
    print("\n" + "="*80)
    print("🚀 Starting Inference Simulation with an UNTRAINED Gating Network")
    print("="*80)
    
    # Note: Since the gate is untrained, its choices will be random/arbitrary.
    # The purpose here is to verify that the system runs end-to-end.
    result = moe_system.predict(sentence, harvested_energy=0.0)
    
    print(f"    🔋 Energy State: {result['energy_before']:.1f} -> {result['energy_after']:.1f} (Cost: {result['energy_cost']:.1f})")
    print(f"    🤖 Expert Used:  {result['expert_used']} ({result['status']})")
    if result['status'] == 'Success':
        print(f"    📝 Prediction:   {result['prediction']} (Score: {result['score']:.4f})")
        print(f"    ⏱️ Latency:      {result['latency_sec']*1000:.2f} ms")