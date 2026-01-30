# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F


from opentslm.model_config import TRANSFORMER_INPUT_DIM, ENCODER_OUTPUT_DIM, PATCH_SIZE
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class TransformerCNNEncoder(TimeSeriesEncoderBase):
    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        transformer_input_dim: int = TRANSFORMER_INPUT_DIM,
        num_heads: int = 8,
        num_layers: int = 6,
        patch_size: int = PATCH_SIZE,
        ff_dim: int = 1024,
        max_patches: int = 1024,
        trained_patches: int = 32,  # Number of patches seen during training (2.56s @ 50Hz / 4)
    ):
        """
        Args:
            embed_dim: dimension of patch embeddings
            num_heads: number of attention heads
            num_layers: number of TransformerEncoder layers
            patch_size: length of each patch
            ff_dim: hidden size of the feed‐forward network inside each encoder layer
            dropout: dropout probability
            max_patches: maximum number of patches expected per sequence (for pos emb)
        """
        super().__init__(output_dim, dropout)
        self.patch_size = patch_size
        self.trained_patches = trained_patches  # For position interpolation

        # 1) Conv1d patch embedding: (B, 1, L) -> (B, embed_dim, L/patch_size)
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=transformer_input_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # 2) Learnable positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_patches, transformer_input_dim)
        )

        # 3) Input norm + dropout
        self.input_norm = nn.LayerNorm(transformer_input_dim)
        self.input_dropout = nn.Dropout(self.dropout)

        # 4) Stack of TransformerEncoder layers with higher ff_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape [B, L], a batch of raw time series.
        Returns:
            FloatTensor of shape [B, N, embed_dim], where N = L // patch_size.
        """

        B, L = x.shape
        if L % self.patch_size != 0:
            raise ValueError(
                f"Sequence length {L} not divisible by patch_size {self.patch_size}"
            )

        # reshape to (B, 1, L)
        x = x.unsqueeze(1)

        # conv patch embedding -> (B, embed_dim, N)
        x = self.patch_embed(x)

        # transpose to (B, N, embed_dim)
        x = x.transpose(1, 2)

        # add positional embeddings with linear interpolation for longer sequences
        N = x.size(1)

        if N <= self.trained_patches:
            # Use learned positions directly (within training range)
            pos = self.pos_embed[:, :N, :]
        else:
            # Interpolate from trained positions to cover longer sequences
            print(f"Patches ({N}) > trained ({self.trained_patches}). Interpolating pos embeddings.")
            # Only use the trained portion of pos_embed for interpolation
            trained_pos = self.pos_embed[:, :self.trained_patches, :]  # (1, trained_patches, dim)
            # Shape: (1, trained_patches, dim) -> (1, dim, trained_patches) for interpolation
            pos_embed_t = trained_pos.transpose(1, 2)
            # Interpolate to target length
            pos_interp = F.interpolate(pos_embed_t, size=N, mode='linear', align_corners=True)
            # Back to (1, N, dim)
            pos = pos_interp.transpose(1, 2)

        x = x + pos

        # norm + dropout
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # apply Transformer encoder
        x = self.encoder(x)

        return x
