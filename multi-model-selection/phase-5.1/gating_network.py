# file: gating_network.py

import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    """
    A simple MLP that takes text embeddings and the current energy state to
    predict the most suitable expert.
    """
    def __init__(self, embedding_dim: int, num_experts: int, hidden_dim: int = 128):
        """
        Initializes the Gating Network.
        Args:
            embedding_dim (int): The dimension of the input text embedding (e.g., 768 for DistilBERT).
            num_experts (int): The number of experts to choose from.
            hidden_dim (int): The size of the hidden layer.
        """
        super().__init__()
        # The input dimension is the text embedding size + 1 for the scalar energy level
        self.layer_stack = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, text_embedding: torch.Tensor, energy_level: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the gating network.
        Args:
            text_embedding (torch.Tensor): Tensor of shape [batch_size, embedding_dim].
            energy_level (torch.Tensor): Tensor of shape [batch_size, 1] representing current energy.
        
        Returns:
            torch.Tensor: Logits for each expert, shape [batch_size, num_experts].
        """
        # Ensure energy level is normalized or scaled appropriately if needed,
        # here we assume it's used directly.
        combined_input = torch.cat([text_embedding, energy_level], dim=1)
        logits = self.layer_stack(combined_input)
        return logits