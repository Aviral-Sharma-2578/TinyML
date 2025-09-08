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

# -------------------------------------------------------------------
#  Fuzzy-based Model Selector (Complexity-Aware)
# -------------------------------------------------------------------

class FuzzyComplexityModelSelector:
    """
    Implements a fuzzy inference–like approach:
      - Input: problem complexity (LOW / MEDIUM / HIGH)
      - Output: which model to use
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizer = None
        self.labels = ['NEGATIVE', 'POSITIVE']

        # Define membership scores for each model under each complexity level.
        # (In practice, we would design membership functions; here we keep it discrete/simple.)
        self.fuzzy_rules = {
            'LOW': {
                'baseline': 0.1,
                'quantized_baseline': 0.4,
                'pruned': 0.7,
                'pruned_quantized': 1.0,
            },
            'MEDIUM': {
                'baseline': 0.5,
                'quantized_baseline': 0.9,
                'pruned': 0.6,
                'pruned_quantized': 0.4,
            },
            'HIGH': {
                'baseline': 1.0,
                'quantized_baseline': 0.8,
                'pruned': 0.3,
                'pruned_quantized': 0.1,
            }
        }

        print("🚀 Initializing Fuzzy Complexity Model Selector...")
        self._load_all_models()

    def _load_all_models(self):
        """Loads all models"""
        print("\n1. Loading tokenizer...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(BASELINE_DIR)

        print("\n2. Loading Baseline Model...")
        self.models['baseline'] = DistilBertForSequenceClassification.from_pretrained(BASELINE_DIR).to(self.device)

        print("\n3. Loading Quantized Baseline...")
        model_to_quantize = DistilBertForSequenceClassification.from_pretrained(BASELINE_DIR)
        quantize_(model_to_quantize, Int8WeightOnlyConfig())
        self.models['quantized_baseline'] = model_to_quantize.to(self.device)

        print("\n4. Loading Pruned Model (simulated fallback, architecture changes skipped here)...")
        # Demo: For simplicity, just re-use baseline (pretend pruned)
        self.models['pruned'] = DistilBertForSequenceClassification.from_pretrained(BASELINE_DIR).to(self.device)

        print("\n5. Loading Pruned+Quantized...")
        # Demo: reuse baseline again, quantized
        model_qp = DistilBertForSequenceClassification.from_pretrained(BASELINE_DIR)
        quantize_(model_qp, Int8WeightOnlyConfig())
        self.models['pruned_quantized'] = model_qp.to(self.device)

        for model in self.models.values():
            model.eval()

        print("\n✅ All demo models loaded.")

    def select_model(self, complexity: str):
        """Apply fuzzy rules to pick the model with highest membership score."""
        scores = self.fuzzy_rules.get(complexity.upper())
        if scores is None:
            raise ValueError(f"Unknown complexity level: {complexity}")

        # Pick the model with the highest fuzzy membership
        chosen = max(scores.items(), key=lambda kv: kv[1])[0]
        return chosen, scores

    def infer(self, text: str, complexity: str):
        """Perform inference after fuzzy model selection."""
        chosen_model, scores = self.select_model(complexity)
        model = self.models[chosen_model]

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            start = time.perf_counter()
            outputs = model(**inputs)
            latency = time.perf_counter() - start

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()

        return {
            "prediction": self.labels[idx],
            "score": probs[0][idx].item(),
            "model_used": chosen_model,
            "latency_sec": latency,
            "fuzzy_scores": scores
        }

# -------------------------------------------------------------------
# Example Demo
# -------------------------------------------------------------------
if __name__ == "__main__":
    controller = FuzzyComplexityModelSelector()

    sentence = "This movie is absolutely stunning, I loved every moment."

    scenarios = [
        {"complexity": "HIGH", "task": "Complex input (e.g., long paragraph, ambiguous)"},
        {"complexity": "MEDIUM", "task": "Moderate complexity input"},
        {"complexity": "LOW", "task": "Simple and short input"},
    ]

    print("\n" + "="*80)
    print("🚀 Starting Fuzzy Complexity Inference Simulation")
    print("="*80)

    for i, step in enumerate(scenarios):
        print(f"\n--- Step {i+1}: {step['task']} ---")
        result = controller.infer(sentence, step['complexity'])
        print(f"    🎚 Complexity: {step['complexity']}")
        print(f"    🤖 Model Used: {result['model_used']}")
        print(f"    📊 Fuzzy Scores: {result['fuzzy_scores']}")
        print(f"    📝 Prediction: {result['prediction']} (Score: {result['score']:.4f})")
        print(f"    ⏱️ Latency: {result['latency_sec']*1000:.2f} ms")
