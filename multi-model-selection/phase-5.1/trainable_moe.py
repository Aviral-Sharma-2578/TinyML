# file: trainable_moe.py

import os
import copy
import time
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torchao.quantization import quantize_, Int8WeightOnlyConfig

# --- Configuration ---
BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-1", "baseline_model"))
PRUNED_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-2", "pruned_true_reduction.pt"))
QUANTIZED_BASELINE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-3", "quantized_baseline_weight_only_int8.pt"))
QUANTIZED_PRUNED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "phase-3", "quantized_weight_only_int8.pt"))

# --- Helpers shared with phase-5/MoE ---
def load_state_dict_safely(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return torch.load(model_path, map_location=device)

def create_pruned_model_architecture(state_dict):
    """
    Create a model with architecture matching the pruned state dict
    """
    model = DistilBertForSequenceClassification.from_pretrained(BASELINE_DIR)
    ffn_dims = {}
    for key in state_dict.keys():
        if 'ffn.lin1.weight' in key:
            try:
                layer_num = int(key.split('.')[3])
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
    Enhanced Gating Network where Energy is a first-class citizen.
    It projects the scalar energy value into a high-dimensional vector space
    before combining it with text embeddings.
    """
    def __init__(self, embedding_dim: int, num_experts: int, hidden_dim: int = 128):
        super().__init__()
        
        # Project text embeddings (768 -> hidden_dim)
        self.text_projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Project scalar energy (1 -> hidden_dim)
        # This gives the model a rich representation of "Energy State"
        self.energy_projector = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(), # Tanh normalizes the continuous energy signal
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Combine and decide
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # Concatenating both vectors
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, text_embedding: torch.Tensor, energy_level: torch.Tensor) -> torch.Tensor:
        # text_embedding: [Batch, 768]
        # energy_level:   [Batch, 1]
        
        text_vec = self.text_projector(text_embedding)
        energy_vec = self.energy_projector(energy_level)
        
        # Combine distinct representations
        combined_input = torch.cat([text_vec, energy_vec], dim=1)
        logits = self.classifier(combined_input)
        return logits

# --- Main MoE System ---

class TrainableEnergyAwareMoE:
    def __init__(self, max_energy_capacity=100.0, initial_energy=20.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experts = {}
        self.tokenizer = None
        self.labels = ['NEGATIVE', 'POSITIVE']
        
        self.expert_names = ['baseline', 'quantized_baseline', 'pruned', 'pruned_quantized']
        # Costs for the custom loss function
        self.expert_energy_costs = {
            'baseline': 15.0, 'quantized_baseline': 10.0,
            'pruned': 8.0, 'pruned_quantized': 5.0,
        }
        # Pre-compute cost tensor for faster loss calculation
        self.expert_costs_tensor = torch.tensor(
            [self.expert_energy_costs[name] for name in self.expert_names], 
            device=self.device
        )
        
        self.max_energy_capacity = max_energy_capacity
        self.current_energy = min(initial_energy, max_energy_capacity)

        print(f"🚀 Initializing MoE on {self.device}...")
        self._load_experts()

        self.gating_network = GatingNetwork(
            embedding_dim=768, 
            num_experts=len(self.expert_names)
        ).to(self.device)
        self.gating_network.eval()

        if 'baseline' in self.experts:
            self.text_embedder = self.experts['baseline'].distilbert.embeddings.to(self.device)
            self.text_embedder.eval()

    def _load_experts(self):
        print("   - Loading experts from saved checkpoints...")
        model_name = BASELINE_DIR if os.path.isdir(BASELINE_DIR) else 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

        # Baseline
        self.experts['baseline'] = DistilBertForSequenceClassification.from_pretrained(model_name).to(self.device)

        # Pruned
        try:
            pruned_sd = load_state_dict_safely(PRUNED_MODEL_PATH, self.device)
            pruned_model = create_pruned_model_architecture(pruned_sd)
            pruned_model.load_state_dict(pruned_sd)
            self.experts['pruned'] = pruned_model.to(self.device)
        except Exception as e:
            print(f"   ⚠️ Could not load pruned expert: {e}")

        # Quantized baseline
        try:
            q_base_model = DistilBertForSequenceClassification.from_pretrained(model_name)
            quantize_(q_base_model, Int8WeightOnlyConfig())
            q_base_sd = load_state_dict_safely(QUANTIZED_BASELINE_PATH, self.device)
            q_base_model.load_state_dict(q_base_sd)
            self.experts['quantized_baseline'] = q_base_model.to(self.device)
        except Exception as e:
            print(f"   ⚠️ Could not load quantized baseline expert: {e}")

        # Pruned + quantized
        try:
            q_pruned_sd_cpu = load_state_dict_safely(QUANTIZED_PRUNED_PATH, 'cpu')
            q_pruned_model = create_pruned_model_architecture(q_pruned_sd_cpu)
            quantize_(q_pruned_model, Int8WeightOnlyConfig())
            q_pruned_model.load_state_dict(q_pruned_sd_cpu)
            self.experts['pruned_quantized'] = q_pruned_model.to(self.device)
        except Exception as e:
            print(f"   ⚠️ Could not load pruned + quantized expert: {e}")

        for model in self.experts.values():
            model.eval()

    def _get_text_embedding(self, inputs: dict) -> torch.Tensor:
        with torch.no_grad():
            return self.text_embedder(inputs['input_ids'])[:, 0, :]

    def predict(self, text: str, harvested_energy: float = 0.0):
        self.current_energy = min(self.current_energy + harvested_energy, self.max_energy_capacity)
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        text_emb = self._get_text_embedding(inputs)
        energy_tensor = torch.tensor([[self.current_energy]], dtype=torch.float32, device=self.device)
        
        # 1. Gate Decision
        with torch.no_grad():
            gate_logits = self.gating_network(text_emb, energy_tensor)
            
            # Mask out unaffordable experts during inference (Hard constraint)
            for i, name in enumerate(self.expert_names):
                if self.expert_energy_costs[name] > self.current_energy:
                    gate_logits[0, i] = -float('inf')

            if torch.all(gate_logits == -float('inf')):
                return {
                    "prediction": "SKIPPED", "score": 0.0, 
                    "expert_used": "None", "latency_sec": 0.0, 
                    "energy_before": self.current_energy, "energy_after": self.current_energy, 
                    "energy_cost": 0.0, "status": "Insufficient energy"
                }

            expert_idx = torch.argmax(gate_logits, dim=1).item()
            expert_name = self.expert_names[expert_idx]

        # 2. Expert Execution
        expert = self.experts[expert_name]
        cost = self.expert_energy_costs[expert_name]
        energy_before = self.current_energy
        
        start = time.perf_counter()
        with torch.no_grad():
            outputs = expert(**inputs)
        latency = time.perf_counter() - start
        
        self.current_energy -= cost
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
        return {
            "prediction": self.labels[pred_idx],
            "score": probs[0][pred_idx].item(),
            "expert_used": expert_name,
            "latency_sec": latency,
            "energy_before": energy_before,
            "energy_after": self.current_energy,
            "energy_cost": cost,
            "status": "Success"
        }