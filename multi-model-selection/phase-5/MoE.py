import os
import time
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torchao.quantization import quantize_, Int8WeightOnlyConfig

# --- Configuration ---
BASE_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
BASELINE_DIR = os.path.join(BASE_OUTPUT_DIR, "phase-1", "baseline_model")
PRUNED_MODEL_PATH = os.path.join(BASE_OUTPUT_DIR, "phase-2", "pruned_true_reduction.pt")
QUANTIZED_BASELINE_PATH = os.path.join(BASE_OUTPUT_DIR, "phase-3", "quantized_baseline_weight_only_int8.pt")
QUANTIZED_PRUNED_PATH = os.path.join(BASE_OUTPUT_DIR, "phase-3", "quantized_weight_only_int8.pt")


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
    # This function requires a valid baseline model to exist for from_pretrained
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

# --- Mixture of Experts Implementation ---

class EnergyAwareMoE:
    """
    Implements a Mixture of Experts (MoE) system where the gating network
    selects an expert based on the available energy.
    """
    def __init__(self, max_energy_capacity=100.0, initial_energy=20.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experts = {} # The pool of expert models
        self.tokenizer = None
        self.labels = ['NEGATIVE', 'POSITIVE']
        
        # --- Energy Simulation Parameters ---
        self.max_energy_capacity = max_energy_capacity
        self.current_energy = min(initial_energy, max_energy_capacity)
        
        # Define the energy cost for an inference call on each expert.
        self.expert_energy_costs = {
            'baseline': 15.0,
            'quantized_baseline': 10.0,
            'pruned': 8.0,
            'pruned_quantized': 5.0,
        }
        
        # The gating network will use this preference order.
        self.expert_preference = ['baseline', 'quantized_baseline', 'pruned', 'pruned_quantized']

        print("🚀 Initializing Energy-Aware Mixture of Experts (MoE)...")
        print(f"   - Device:           {self.device}")
        print(f"   - Battery Capacity: {self.max_energy_capacity} units")
        print(f"   - Initial Energy:   {self.current_energy} units")
        self._load_experts()

    def _load_experts(self):
        """Loads all expert models into memory."""
        print("\n1. Loading shared tokenizer...")
        # A dummy check for the baseline directory to avoid crashing if it's missing
        if not os.path.isdir(BASELINE_DIR):
            print(f"   ❌ CRITICAL: Baseline directory not found at {BASELINE_DIR}. Cannot load models.")
            return
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(BASELINE_DIR)
        
        print("\n2. Loading Expert: 'baseline' (High Accuracy, High Cost)...")
        try:
            self.experts['baseline'] = DistilBertForSequenceClassification.from_pretrained(BASELINE_DIR).to(self.device)
        except Exception as e:
            print(f"   ⚠️ Could not load baseline expert: {e}")

        print("\n3. Loading Expert: 'pruned'...")
        try:
            pruned_sd = load_state_dict_safely(PRUNED_MODEL_PATH, self.device)
            pruned_expert_base = create_pruned_model_architecture(pruned_sd)
            pruned_expert_base.load_state_dict(pruned_sd)
            self.experts['pruned'] = pruned_expert_base.to(self.device)
        except Exception as e:
            print(f"   ⚠️ Could not load pruned expert: {e}")

        print("\n4. Loading Expert: 'quantized_baseline'...")
        try:
            model_to_quantize = DistilBertForSequenceClassification.from_pretrained(BASELINE_DIR)
            quantize_(model_to_quantize, Int8WeightOnlyConfig())
            quantized_sd = load_state_dict_safely(QUANTIZED_BASELINE_PATH, self.device)
            model_to_quantize.load_state_dict(quantized_sd)
            self.experts['quantized_baseline'] = model_to_quantize.to(self.device)
        except Exception as e:
            print(f"   ⚠️ Could not load quantized baseline expert: {e}")
            
        print("\n5. Loading Expert: 'pruned_quantized' (Low Latency, Low Cost)...")
        try:
            if 'pruned' in self.experts:
                # Re-create architecture on CPU first to avoid device mismatches during loading
                pruned_sd_cpu = load_state_dict_safely(PRUNED_MODEL_PATH, 'cpu')
                pruned_model_to_quantize = create_pruned_model_architecture(pruned_sd_cpu)
                quantize_(pruned_model_to_quantize, Int8WeightOnlyConfig())
                quantized_pruned_sd = load_state_dict_safely(QUANTIZED_PRUNED_PATH, self.device)
                pruned_model_to_quantize.load_state_dict(quantized_pruned_sd)
                self.experts['pruned_quantized'] = pruned_model_to_quantize.to(self.device)
            else:
                print("   ⚠️ Skipping because pruned expert failed to load.")
        except Exception as e:
            print(f"   ⚠️ Could not load pruned + quantized expert: {e}")

        for expert in self.experts.values():
            expert.eval()
        
        print(f"\n✅ {len(self.experts)} experts loaded successfully.")

    def _gate(self, current_energy: float) -> str:
        """
        The Gating Network.
        It decides which expert to use based on the current energy level.
        Returns the name of the chosen expert, or 'None' if unaffordable.
        """
        for expert_name in self.expert_preference:
            if expert_name in self.experts and current_energy >= self.expert_energy_costs[expert_name]:
                return expert_name # Route to the best affordable expert
        return "None" # No expert can be afforded

    def predict(self, text: str, harvested_energy: float = 0.0):
        """
        Performs a full MoE inference cycle:
        1. Harvests energy.
        2. Uses the gating network to select an expert.
        3. Invokes the selected expert for inference.
        4. Updates the energy state.
        """
        # 1. Energy Harvesting Step
        self.current_energy = min(self.current_energy + harvested_energy, self.max_energy_capacity)
        
        # 2. Gating Network Step: Select the expert
        chosen_expert_name = self._gate(self.current_energy)
        
        # If the gate returns 'None', no expert can be run.
        if chosen_expert_name == "None":
            return {
                "prediction": "SKIPPED",
                "score": 0.0,
                "expert_used": "None",
                "latency_sec": 0.0,
                "energy_before": self.current_energy,
                "energy_after": self.current_energy,
                "energy_cost": 0.0,
                "status": "Insufficient energy"
            }

        # 3. Expert Invocation and Energy Consumption Step
        expert_model = self.experts[chosen_expert_name]
        energy_cost = self.expert_energy_costs[chosen_expert_name]
        energy_before_inference = self.current_energy
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            start_time = time.perf_counter()
            outputs = expert_model(**inputs)
            latency = time.perf_counter() - start_time
        
        # Update energy state after successful inference
        self.current_energy -= energy_cost
        
        logits = outputs.logits
        scores = torch.softmax(logits, dim=1)
        prediction_idx = torch.argmax(scores, dim=1).item()
        
        return {
            "prediction": self.labels[prediction_idx],
            "score": scores[0][prediction_idx].item(),
            "expert_used": chosen_expert_name,
            "latency_sec": latency,
            "energy_before": energy_before_inference,
            "energy_after": self.current_energy,
            "energy_cost": energy_cost,
            "status": "Success"
        }

# --- Example Usage ---
if __name__ == "__main__":
    
    # Create dummy files and dirs if they don't exist, to allow the script to run for demonstration
    # In a real scenario, these trained models would already exist.
    print("--- Setting up dummy model files for demonstration ---")
    if not os.path.isdir(BASELINE_DIR):
        print(f"Creating dummy directory: {BASELINE_DIR}")
        os.makedirs(BASELINE_DIR)
        # A minimal set of files for from_pretrained to work
        with open(os.path.join(BASELINE_DIR, "config.json"), "w") as f:
            f.write('{"model_type": "distilbert"}')
        with open(os.path.join(BASELINE_DIR, "tokenizer_config.json"), "w") as f:
            f.write('{"model_type": "distilbert"}')
        # Create dummy state dicts as well, as they are expected by the loading logic
        dummy_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        torch.save(dummy_model.state_dict(), os.path.join(BASELINE_DIR, 'pytorch_model.bin'))
        
        os.makedirs(os.path.dirname(PRUNED_MODEL_PATH), exist_ok=True)
        torch.save(dummy_model.state_dict(), PRUNED_MODEL_PATH)

        os.makedirs(os.path.dirname(QUANTIZED_BASELINE_PATH), exist_ok=True)
        torch.save(dummy_model.state_dict(), QUANTIZED_BASELINE_PATH)
        
        os.makedirs(os.path.dirname(QUANTIZED_PRUNED_PATH), exist_ok=True)
        torch.save(dummy_model.state_dict(), QUANTIZED_PRUNED_PATH)
    print("----------------------------------------------------\n")


    # Start with a high initial energy to test the top model
    moe_system = EnergyAwareMoE(max_energy_capacity=50.0, initial_energy=16.0)
    
    sentence = "This movie is a masterpiece, a true work of art."
    
    # A simulation designed to trigger each expert based on energy levels
    simulation_steps = [
        {"harvested": 0.0,  "task": "High initial energy"},  # Energy: 16 -> Gate routes to 'baseline' (cost 15), rem: 1
        {"harvested": 10.0, "task": "Medium energy"},         # Energy: 1+10=11 -> Gate routes to 'q_baseline' (cost 10), rem: 1
        {"harvested": 8.0,  "task": "Low-medium energy"},     # Energy: 1+8=9 -> Gate routes to 'pruned' (cost 8), rem: 1
        {"harvested": 5.0,  "task": "Low energy"},            # Energy: 1+5=6 -> Gate routes to 'pruned_quantized' (cost 5), rem: 1
        {"harvested": 2.0,  "task": "Insufficient energy"},   # Energy: 1+2=3 -> Gate returns 'None'
    ]
    
    print("\n" + "="*80)
    print("🚀 Starting Energy-Aware MoE Inference Simulation")
    print("="*80)

    for i, step in enumerate(simulation_steps):
        print(f"\n--- Step {i+1}: {step['task']} ---")
        print(f"    ⚡️ Energy harvested: {step['harvested']:.1f} units")
        
        result = moe_system.predict(sentence, harvested_energy=step['harvested'])
        
        print(f"    🔋 Energy State: {result['energy_before']:.1f} -> {result['energy_after']:.1f} (Cost: {result['energy_cost']:.1f})")
        print(f"    🤖 Expert Used:  {result['expert_used']} ({result['status']})")

        if result['status'] == 'Success':
            print(f"    📝 Prediction:   {result['prediction']} (Score: {result['score']:.4f})")
            print(f"    ⏱️ Latency:      {result['latency_sec']*1000:.2f} ms")