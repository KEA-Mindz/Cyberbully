"""Neural network architectures for cyberbullying detection."""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class MultiTaskClassifier(nn.Module):
    """Transformer encoder + mean pooling + binary head + 20-way auxiliary category head."""

    def __init__(
        self,
        model_name_or_config_dir: str | Path,
        n_categories: int = 20,
        dropout: float = 0.1,
        pretrained: bool = False,
    ):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(str(model_name_or_config_dir))
        else:
            config = AutoConfig.from_pretrained(str(model_name_or_config_dir))
            self.encoder = AutoModel.from_config(config)

        hidden_size = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.bin_head = nn.Linear(hidden_size, 2)
        self.cat_head = nn.Linear(hidden_size, n_categories)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute binary and category logits.
        
        Args:
            input_ids: Token indices tensor (batch_size, seq_len)
            attention_mask: Mask tensor (batch_size, seq_len)
            
        Returns:
            Tuple of (binary_logits, category_logits)
        """
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = self.drop(pooled)
        return self.bin_head(pooled), self.cat_head(pooled)
