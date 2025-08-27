import os
import time
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torchao.quantization import quantize_, Int8WeightOnlyConfig

# --- Configuration ---
# Paths to the actual trained models
BASE_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
BASELINE_DIR = os.path.join(BASE_OUTPUT_DIR, "phase-1", "baseline_model")
PRUNED_MODEL_PATH = os.path.join(BASE_OUTPUT_DIR, "phase-2", "pruned_true_reduction.pt")
QUANTIZED_BASELINE_PATH = os.path.join(BASE_OUTPUT_DIR, "phase-3", "quantized_baseline_weight_only_int8.pt")
QUANTIZED_PRUNED_PATH = os.path.join(BASE_OUTPUT_DIR, "phase-3", "quantized_weight_only_int8.pt")

# Helper function to safely load state dicts
def load_state_dict_safely(model_path, device):
    """Loads a state dictionary, handling potential errors."""
    try:
        return torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"   ❌ Failed to load state_dict from {model_path}: {e}")
        raise e

# Helper function to create the pruned model architecture
def create_pruned_model_architecture(state_dict):
    """Creates a model with an architecture matching the pruned state dict."""
    model = DistilBertForSequenceClassification.from_pretrained(BASELINE_DIR)
    ffn_dims = {}
    
    # Extract FFN dimensions from the state dict
    for key in state_dict.keys():
        if 'ffn.lin1.weight' in key:
            # Parse the layer number correctly from the key
            # Key format: distilbert.transformer.layer.{layer_num}.ffn.lin1.weight
            parts = key.split('.')
            if len(parts) >= 4:
                try:
                    layer_num = int(parts[3])
                    new_dim = state_dict[key].shape[0]
                    ffn_dims[layer_num] = new_dim
                    print(f"   📊 Layer {layer_num}: FFN dimension = {new_dim}")
                except (ValueError, IndexError) as e:
                    print(f"   ⚠️ Could not parse layer number from key: {key}")
                    continue
    
    print(f"   🔍 Found FFN dimensions: {ffn_dims}")
    
    # Update the model architecture to match the pruned dimensions
    for layer_num, new_dim in ffn_dims.items():
        if layer_num < len(model.distilbert.transformer.layer):
            layer_module = model.distilbert.transformer.layer[layer_num]
            old_lin1 = layer_module.ffn.lin1
            old_lin2 = layer_module.ffn.lin2
            
            # Create new linear layers with the pruned dimensions
            new_lin1 = torch.nn.Linear(old_lin1.in_features, new_dim, bias=old_lin1.bias is not None)
            new_lin2 = torch.nn.Linear(new_dim, old_lin2.out_features, bias=old_lin2.bias is not None)
            
            # Replace the layers
            layer_module.ffn.lin1 = new_lin1
            layer_module.ffn.lin2 = new_lin2
            
            print(f"   ✅ Updated layer {layer_num}: {old_lin1.out_features} -> {new_dim} -> {old_lin2.out_features}")
        else:
            print(f"   ⚠️ Layer {layer_num} not found in model architecture")
    
    return model

class EnergyAwareModelSelector:
    """
    Implements the Multi-Model Selection (MMS) strategy from the paper.
    It dynamically selects a model based on available energy.
    """
    def __init__(self, max_energy_capacity=100.0, initial_energy=20.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizer = None
        self.labels = ['NEGATIVE', 'POSITIVE']
        
        # --- Energy Simulation Parameters ---
        self.max_energy_capacity = max_energy_capacity
        self.current_energy = min(initial_energy, max_energy_capacity)
        
        # Define the energy cost for an inference call on each model.
        # These values are illustrative. In a real system, they would be
        # empirically measured.
        self.model_energy_costs = {
            'baseline': 15.0,
            'quantized_baseline': 10.0,
            'pruned': 8.0,
            'pruned_quantized': 5.0,
        }
        
        # Define the order of preference from most accurate to most efficient.
        # The selector will try to use the best model it can afford.
        self.model_preference = ['baseline', 'quantized_baseline', 'pruned', 'pruned_quantized']

        print("🚀 Initializing Energy-Aware Model Selector...")
        print(f"   - Battery Capacity: {self.max_energy_capacity} units")
        print(f"   - Initial Energy:   {self.current_energy} units")
        self._load_all_models()

    def _load_all_models(self):
        """Loads all available models into memory."""
        print("\n1. Loading tokenizer...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(BASELINE_DIR)
        
        print("\n2. Loading Baseline Model (High Accuracy, High Cost)...")
        self.models['baseline'] = DistilBertForSequenceClassification.from_pretrained(BASELINE_DIR).to(self.device)

        print("\n3. Loading Pruned Model...")
        try:
            pruned_sd = load_state_dict_safely(PRUNED_MODEL_PATH, self.device)
            pruned_model_base = create_pruned_model_architecture(pruned_sd)
            pruned_model_base.load_state_dict(pruned_sd)
            self.models['pruned'] = pruned_model_base.to(self.device)
        except Exception as e:
            print(f"   ⚠️ Could not load pruned model: {e}")

        print("\n4. Loading Quantized Baseline Model...")
        try:
            model_to_quantize = DistilBertForSequenceClassification.from_pretrained(BASELINE_DIR)
            quantize_(model_to_quantize, Int8WeightOnlyConfig())
            quantized_sd = load_state_dict_safely(QUANTIZED_BASELINE_PATH, self.device)
            model_to_quantize.load_state_dict(quantized_sd)
            self.models['quantized_baseline'] = model_to_quantize.to(self.device)
        except Exception as e:
            print(f"   ⚠️ Could not load quantized baseline model: {e}")

        print("\n5. Loading Pruned + Quantized Model (Low Latency, Low Cost)...")
        try:
            if 'pruned' in self.models:
                pruned_model_to_quantize = create_pruned_model_architecture(load_state_dict_safely(PRUNED_MODEL_PATH, 'cpu'))
                quantize_(pruned_model_to_quantize, Int8WeightOnlyConfig())
                quantized_pruned_sd = load_state_dict_safely(QUANTIZED_PRUNED_PATH, self.device)
                pruned_model_to_quantize.load_state_dict(quantized_pruned_sd)
                self.models['pruned_quantized'] = pruned_model_to_quantize.to(self.device)
            else:
                print("   ⚠️ Skipping because pruned model failed to load.")
        except Exception as e:
            print(f"   ⚠️ Could not load pruned + quantized model: {e}")

        for model in self.models.values():
            model.eval()
        
        print("\n✅ All available models loaded.")

    def select_and_infer(self, text: str, harvested_energy: float = 0.0):
        """
        Dynamically selects a model based on energy, performs inference,
        and updates the energy state.
        """
        # 1. Energy Harvesting Step
        self.current_energy = min(self.current_energy + harvested_energy, self.max_energy_capacity)
        
        # 2. Model Selection Step (The core of MMS)
        model_choice = None
        for model_name in self.model_preference:
            if model_name in self.models and self.current_energy >= self.model_energy_costs[model_name]:
                model_choice = model_name
                break # Found the best affordable model
        
        # If no model can be afforded, we must skip this inference cycle.
        if model_choice is None:
            return {
                "prediction": "SKIPPED",
                "score": 0.0,
                "model_used": "None",
                "latency_sec": 0.0,
                "energy_before": self.current_energy,
                "energy_after": self.current_energy,
                "energy_cost": 0.0,
                "status": "Insufficient energy"
            }

        # 3. Inference and Energy Consumption Step
        model = self.models[model_choice]
        energy_cost = self.model_energy_costs[model_choice]
        energy_before_inference = self.current_energy
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            start_time = time.perf_counter()
            outputs = model(**inputs)
            latency = time.perf_counter() - start_time
        
        # Update energy state after successful inference
        self.current_energy -= energy_cost
        
        logits = outputs.logits
        scores = torch.softmax(logits, dim=1)
        prediction_idx = torch.argmax(scores, dim=1).item()
        
        return {
            "prediction": self.labels[prediction_idx],
            "score": scores[0][prediction_idx].item(),
            "model_used": model_choice,
            "latency_sec": latency,
            "energy_before": energy_before_inference,
            "energy_after": self.current_energy,
            "energy_cost": energy_cost,
            "status": "Success"
        }

# --- Example Usage ---
if __name__ == "__main__":
    
    # Start with a high initial energy to test the top model
    controller = EnergyAwareModelSelector(max_energy_capacity=50.0, initial_energy=16.0)
    
    sentence = "This movie is a masterpiece, a true work of art."
    
    # A new simulation designed to trigger each model
    simulation_steps = [
        {"harvested": 0.0,  "task": "High initial energy"},  # Energy: 16 -> Chooses baseline (cost 15), remaining: 1
        {"harvested": 10.0, "task": "Medium energy"},         # Energy: 1+10=11 -> Chooses q_baseline (cost 10), remaining: 1
        {"harvested": 8.0,  "task": "Low-medium energy"},     # Energy: 1+8=9 -> Chooses pruned (cost 8), remaining: 1
        {"harvested": 5.0,  "task": "Low energy"},            # Energy: 1+5=6 -> Chooses pruned_quantized (cost 5), remaining: 1
        {"harvested": 2.0,  "task": "Insufficient energy"},   # Energy: 1+2=3 -> Chooses None
    ]
    
    print("\n" + "="*80)
    print("🚀 Starting Energy-Aware Inference Simulation (Revised)")
    print("="*80)

    for i, step in enumerate(simulation_steps):
        print(f"\n--- Step {i+1}: {step['task']} ---")
        print(f"    ⚡️ Energy harvested: {step['harvested']:.1f} units")
        
        result = controller.select_and_infer(sentence, harvested_energy=step['harvested'])
        
        print(f"    🔋 Energy State: {result['energy_before']:.1f} -> {result['energy_after']:.1f} (Cost: {result['energy_cost']:.1f})")
        print(f"    🤖 Model Used:   {result['model_used']} ({result['status']})")

        if result['status'] == 'Success':
            print(f"    📝 Prediction:   {result['prediction']} (Score: {result['score']:.4f})")
            print(f"    ⏱️ Latency:      {result['latency_sec']*1000:.2f} ms")