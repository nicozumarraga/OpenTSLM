# Phase 1: Query-Aware Perceiver Implementation Plan

## Executive Summary

This document outlines the implementation plan for Phase 1 of the OpenTSLM-TITANS integration project. The goal is to implement and benchmark a **query-aware Perceiver Resampler** that conditions time-series compression on user questions, testing the hypothesis that query-specific compression improves performance over query-agnostic compression.

**Target Dataset:** ECG-QA Chain-of-Thought (CoT) - Stage 5
- **Complexity:** 12-lead ECG signals (12,000 samples per patient @ 100Hz)
- **Baseline Performance (Llama-3.2-1B-Flamingo):**
  - F1 Score: 34.62%
  - Accuracy: 38.14%
- **Task:** Medical question answering with chain-of-thought reasoning

---

## 1. Baseline Benchmark Replication

### 1.1 Dataset Overview

**ECG-QA CoT Dataset (`ECGQACoTQADataset`):**
- Location: `OpenTSLM/src/opentslm/time_series_datasets/ecg_qa/ECGQACoTQADataset.py`
- Data source: PTB-XL ECG database + ECG-QA questions
- Features:
  - 12-lead ECG signals (I, II, III, aVR, aVL, aVF, V1-V6)
  - Each lead: 1000 samples (10 seconds @ 100Hz)
  - Total: 12,000 samples per patient
  - Clinical context + medical questions + chain-of-thought rationales
- Splits: Train / Validation / Test
- Question types: Diagnosis, measurement, comparison, temporal reasoning
- Output: Chain-of-thought rationale + final answer

**Why ECG-QA CoT?**
1. **Longest sequences:** 12 leads × 1000 samples = 12K samples per example
2. **Complex reasoning:** Requires medical knowledge + temporal pattern recognition
3. **Query diversity:** Different questions require attention to different ECG features
4. **Clinical relevance:** Real-world medical application

### 1.2 Training Configuration (Stage 5)

From `curriculum_learning.py:1332-1356`:

```python
def stage5_ecg_cot(self, batch_size=None, eval_only=False):
    return self._train_stage(
        stage_name="stage5_ecg_cot",
        dataset_class=ECGQACoTQADataset,
        num_epochs=60,
        lr_encoder=2e-4,      # For OpenTSLMSP
        lr_projector=1e-4,    # For OpenTSLMSP
        lr_base=2e-4,         # For OpenTSLMFlamingo
        metric_func=None,     # Test loss only (no accuracy during training)
        batch_size=batch_size or BATCH_SIZE,
        eval_only=eval_only,
    )
```

**Key Parameters (from `model_config.py`):**
- `BATCH_SIZE`: 4 (default)
- `GRAD_CLIP_NORM`: 1.0
- `WARMUP_FRAC`: 0.03
- `WEIGHT_DECAY`: 0.01
- `EARLY_STOP_PAT`: 10 epochs
- `PATCH_SIZE`: 64 (Conv1d patch size)
- LoRA enabled for stage 5 (LLM fine-tuning)

**Architecture Used:**
- Model: `OpenTSLMFlamingo` (Perceiver-based, better for long sequences)
- Base LLM: `meta-llama/Llama-3.2-1B`
- Perceiver: 64 learnable latents, depth=6, heads=8
- Cross-attention every N layers in LLM

### 1.3 Baseline Replication Steps

#### Step 1: Environment Setup

```bash
cd /path/to/OpenTSLM

# Install dependencies
uv sync --all-groups
source .venv/bin/activate

# Authenticate with Hugging Face (for Llama access)
huggingface-cli login

# Install WFDB for ECG data loading
pip install wfdb
```

#### Step 2: Download Pretrained Baseline Model

**Option A: Use pretrained checkpoint from HuggingFace**

```bash
# Test with pretrained model first to verify dataset and evaluation
python demo/huggingface/05_test_hf_ecg_qa_cot.py
```

The script uses: `REPO_ID = "OpenTSLM/llama-3.2-1b-ecg-flamingo"`

**Option B: Train from curriculum stages 1-5**

```bash
# Full curriculum training (stage 1 → 5)
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --llm_id meta-llama/Llama-3.2-1B \
    --device cuda

# Or train only stage 5 (requires stage 4 checkpoint)
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage5_ecg_cot \
    --device cuda
```

#### Step 3: Run Baseline Evaluation

**Checkpoint Location:**
```
results/Llama3_2_1B/OpenTSLMFlamingo/stage5_ecg_cot/checkpoints/best_model.pt
```

**Evaluation Script Location:**
```
evaluation/opentslm/ecg_qa_cot/parse_ecg_qa_cot_data.py
```

**Evaluation Process:**
1. Model generates predictions → saved to `test_predictions.jsonl`
2. Parser extracts final answers from chain-of-thought responses
3. F1 score calculated per template (each template has specific answer set)
4. Aggregated metrics: overall F1, accuracy, per-template breakdown

**Run Evaluation:**

```bash
# Generate predictions (automatically done during training)
# Or run eval-only mode:
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage5_ecg_cot \
    --eval_only \
    --device cuda

# Parse predictions and calculate metrics
cd evaluation/opentslm/ecg_qa_cot
python parse_ecg_qa_cot_data.py \
    --input_path ../../../results/Llama3_2_1B/OpenTSLMFlamingo/stage5_ecg_cot/results/test_predictions.jsonl \
    --output_path ./parsed_results.json
```

**Expected Output:**
```json
{
  "overall_metrics": {
    "f1_score": 0.3462,
    "accuracy": 0.3814,
    "num_samples": 1234
  },
  "per_template_metrics": {
    "template_1": {"f1": 0.45, "count": 100},
    "template_2": {"f1": 0.32, "count": 87},
    ...
  }
}
```

#### Step 4: Verify Baseline Results

**Success Criteria:**
- ✅ Training completes 60 epochs without OOM errors
- ✅ Test loss converges (check `loss_history.txt`)
- ✅ F1 ≈ 34.62%, Accuracy ≈ 38.14% (±2% variance acceptable)
- ✅ Predictions file contains valid chain-of-thought responses

**Debugging Checklist:**
- [ ] ECG data downloaded correctly (PTB-XL database)
- [ ] WFDB library installed (`pip install wfdb`)
- [ ] Sufficient GPU memory (16GB+ recommended)
- [ ] Gradient checkpointing enabled if OOM (`--gradient_checkpointing`)
- [ ] Correct LLM authentication (Hugging Face token)

---

## 2. Query-Aware Perceiver Implementation

### 2.1 Architecture Modifications

**Core Hypothesis:**
Current Perceiver learns **one** generic 64-token compression for all questions. Query-aware Perceiver should learn **different** compressions based on the question type.

**Example:**
- Question: "What is the heart rate?" → Compress to extract R-R intervals
- Question: "Is there ST elevation?" → Compress to extract ST segment morphology
- Question: "Is there left ventricular hypertrophy?" → Compress to extract QRS amplitude patterns

**Implementation Strategy:**

```
Current Flow:
  TS patches [B, N_patches, D] → Perceiver (fixed latents) → Compressed [B, 64, D]

Query-Aware Flow:
  Query text → LLM embeddings → Query vector [B, D]
  TS patches [B, N_patches, D] + Query vector [B, D] → Perceiver (conditioned latents) → Compressed [B, 64, D]
```

### 2.2 Implementation Strategy

**IMPORTANT:** The `open-flamingo` library is installed as a pip package (`open-flamingo>=0.0.2` in `pyproject.toml`). The code lives in `venv/lib/python3.12/site-packages/open_flamingo/`.

**We will NOT modify the open-flamingo package directly.** Instead, we will:
1. Create a custom `QueryAwarePerceiverResampler` class in the OpenTSLM codebase
2. Override the perceiver in `TimeSeriesFlamingoWithTrainableEncoder` after calling `super().__init__()`

This approach keeps all changes in version control and avoids breaking pip dependencies.

### 2.3 Files to Create/Modify

#### File 1 (NEW): `OpenTSLM/src/opentslm/model/perceiver/query_aware_perceiver.py`

**Create this new file** with the query-aware perceiver implementation:

```python
"""
Query-Aware Perceiver Resampler for OpenTSLM.

This module extends the standard PerceiverResampler from open_flamingo to support
query-conditioned compression of time series data.
"""

import torch
from torch import nn
from einops import rearrange, repeat
from einops_exts import rearrange_many


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    """Attention module for Perceiver - copied from open_flamingo for self-containment."""

    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class QueryAwarePerceiverResampler(nn.Module):
    """
    Query-Aware Perceiver Resampler that conditions time-series compression on user queries.

    This extends the standard PerceiverResampler to support three fusion methods:
    - cross_attn: Query attends to base latents (most expressive)
    - add: Additive fusion with learned gate (simplest)
    - gate: Gated fusion similar to GRU (middle ground)

    When query_conditioning=False, this behaves identically to the original PerceiverResampler.
    """

    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=64,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
        query_conditioning=False,
        query_fusion_method="cross_attn",  # "cross_attn", "add", "gate"
    ):
        super().__init__()
        self.num_latents = num_latents
        self.query_conditioning = query_conditioning
        self.query_fusion_method = query_fusion_method

        # Original learnable latents (base representation)
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # Frame and media time embeddings (same as original)
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        # Query conditioning modules
        if query_conditioning:
            if query_fusion_method == "cross_attn":
                # Query attends to base latents to select relevant ones
                self.query_to_latents = PerceiverAttention(
                    dim=dim, dim_head=dim_head, heads=heads
                )
            elif query_fusion_method == "add":
                # Additive fusion with learned gate
                self.query_projection = nn.Linear(dim, dim)
                self.fusion_gate = nn.Parameter(torch.zeros(1))
                # Initialize with small values for stable training
                nn.init.xavier_uniform_(self.query_projection.weight, gain=0.01)
            elif query_fusion_method == "gate":
                # Gated fusion (like GRU)
                self.query_gate = nn.Sequential(
                    nn.Linear(dim * 2, dim),
                    nn.Sigmoid()
                )
                self.query_projection = nn.Linear(dim, dim)
                # Initialize with small values for stable training
                nn.init.xavier_uniform_(self.query_projection.weight, gain=0.01)
            else:
                raise ValueError(f"Unknown query_fusion_method: {query_fusion_method}")

        # Perceiver attention layers (same as original)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, query_embedding=None):
        """
        Args:
            x (torch.Tensor): time series features, shape (b, T, F, v, D)
            query_embedding (torch.Tensor): query embeddings, shape (b, D) or (b, T, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # Frame and media time embeddings (same as original)
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(x, "b T F v d -> b T (F v) d")
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # Condition latents on query (NEW)
        if self.query_conditioning and query_embedding is not None:
            # Expand query to match temporal dimension if needed
            if query_embedding.ndim == 2:  # (b, D)
                query_embedding = repeat(query_embedding, "b d -> b T d", T=T)

            latents = self._condition_latents_on_query(query_embedding, b, T)
        else:
            # Original: use fixed base latents
            latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)

        # Perceiver attention layers (same as original)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)

    def _condition_latents_on_query(self, query_embedding, b, T):
        """Condition latent vectors on query embedding."""
        base_latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)

        if self.query_fusion_method == "cross_attn":
            # Query attends to base latents to select relevant ones
            query_expanded = repeat(query_embedding, "b T d -> b T 1 d")
            conditioned = self.query_to_latents(query_expanded, base_latents)
            return base_latents + conditioned  # Residual connection

        elif self.query_fusion_method == "add":
            # Additive fusion with learned scaling
            query_proj = self.query_projection(query_embedding)
            query_expanded = repeat(query_proj, "b T d -> b T n d", n=self.num_latents)
            gate = torch.sigmoid(self.fusion_gate)
            return base_latents + gate * query_expanded

        elif self.query_fusion_method == "gate":
            # Gated fusion (query gates which latents to emphasize)
            query_expanded = repeat(query_embedding, "b T d -> b T n d", n=self.num_latents)
            concat = torch.cat([base_latents, query_expanded], dim=-1)
            gate = self.query_gate(concat)
            query_proj = self.query_projection(query_expanded)
            return base_latents * gate + query_proj * (1 - gate)

        return base_latents
```

**Also create** `OpenTSLM/src/opentslm/model/perceiver/__init__.py`:

```python
from .query_aware_perceiver import QueryAwarePerceiverResampler

__all__ = ["QueryAwarePerceiverResampler"]
```

#### File 2: `OpenTSLM/src/opentslm/model/llm/TimeSeriesFlamingoWithTrainableEncoder.py`

**Modify the existing file** to support query-aware perceiver:

```python
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

import torch
from torch import nn
from open_flamingo import Flamingo
from einops import rearrange

from opentslm.model.perceiver import QueryAwarePerceiverResampler


class TimeSeriesFlamingoWithTrainableEncoder(Flamingo):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
        # NEW: Query-aware perceiver options
        query_conditioning: bool = False,
        query_fusion_method: str = "cross_attn",
    ):
        # Call parent __init__ which creates the default perceiver
        super().__init__(
            vision_encoder,
            lang_encoder,
            eoc_token_id,
            media_token_id,
            vis_dim,
            cross_attn_every_n_layers,
            gradient_checkpointing
        )

        # Store query conditioning settings
        self.query_conditioning = query_conditioning

        # Replace the default perceiver with query-aware version if enabled
        if query_conditioning:
            self.perceiver = QueryAwarePerceiverResampler(
                dim=vis_dim,
                depth=6,
                dim_head=64,
                heads=8,
                num_latents=64,
                query_conditioning=True,
                query_fusion_method=query_fusion_method,
            )

    def _encode_vision_x(self, vision_x, lang_x=None):
        """
        Encode time series data with optional query conditioning.

        Args:
            vision_x: Time series data [b, T_img, F, features]
            lang_x: Language input tokens [b, seq_len] (for query conditioning)
        """
        if vision_x.ndim == 4:  # For shape (b, T_img, F, features)
            b, T, F, features = vision_x.shape

            # Flatten batch, time and frame dimensions
            vision_x = rearrange(vision_x, "b T F c -> (b T F) c")

            # Process through encoder
            vision_x = self.vision_encoder(vision_x)  # [(b*T*F), patches, features]

            # Reshape to expected format for perceiver
            vision_x = rearrange(vision_x, "(b T F) p d -> b T F p d", b=b, T=T, F=F)

            # Extract query embedding if query conditioning is enabled
            query_embedding = None
            if self.query_conditioning and lang_x is not None:
                query_embedding = self._extract_query_embedding(lang_x)

            # Process through perceiver (with optional query conditioning)
            if self.query_conditioning and query_embedding is not None:
                vision_x = self.perceiver(vision_x, query_embedding=query_embedding)
            else:
                vision_x = self.perceiver(vision_x)

        else:
            # Original image processing path
            assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
            b, T, F = vision_x.shape[:3]
            assert F == 1, "Only single frame supported"

            vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
            vision_x = self.vision_encoder(vision_x)[1]
            vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

            # Extract query embedding if query conditioning is enabled
            query_embedding = None
            if self.query_conditioning and lang_x is not None:
                query_embedding = self._extract_query_embedding(lang_x)

            if self.query_conditioning and query_embedding is not None:
                vision_x = self.perceiver(vision_x, query_embedding=query_embedding)
            else:
                vision_x = self.perceiver(vision_x)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

    def _extract_query_embedding(self, lang_x):
        """
        Extract query embedding from language input.

        Args:
            lang_x: Language tokens [b, seq_len]

        Returns:
            Query embedding [b, dim]
        """
        # Get text embeddings from LLM (no gradient - we just want the representation)
        with torch.no_grad():
            text_embeds = self.lang_encoder.get_input_embeddings()(lang_x)

        # Mean pooling over sequence length
        # Alternative strategies: last token, attention-weighted pooling
        query_embedding = text_embeds.mean(dim=1)  # [b, seq_len, dim] -> [b, dim]

        return query_embedding

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Forward pass with query-aware encoding.

        Overrides parent to pass lang_x to _encode_vision_x for query conditioning.
        """
        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )

        if use_cached_vision_x:
            assert vision_x is None, "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()
        else:
            # Pass lang_x for query conditioning (NEW)
            self._encode_vision_x(vision_x=vision_x, lang_x=lang_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **generate_kwargs,
    ):
        """
        Generate with query-aware encoding.

        Overrides parent to pass lang_x to _encode_vision_x for query conditioning.
        """
        num_beams = generate_kwargs.get("num_beams", 1)
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        # Pass lang_x for query conditioning (NEW)
        self._encode_vision_x(vision_x=vision_x, lang_x=lang_x)

        output = self.lang_encoder.generate(
            lang_x,
            attention_mask=attention_mask,
            eos_token_id=self.eoc_token_id,
            **generate_kwargs,
        )

        self.lang_encoder.clear_conditioned_layers()
        return output
```

#### File 3: `OpenTSLM/src/opentslm/model/llm/OpenTSLMFlamingo.py`

**Modify the model initialization** to accept query-aware parameters:

```python
# In OpenTSLMFlamingo.__init__, add parameters and pass to TimeSeriesFlamingoWithTrainableEncoder:

class OpenTSLMFlamingo(TimeSeriesLLM):
    def __init__(
        self,
        device: str,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        cross_attn_every_n_layers: int = 1,
        decoder_layers_attr_name: str = None,
        freeze_lm_embeddings: bool = False,
        # NEW: Query-aware perceiver options
        query_conditioning: bool = False,
        query_fusion_method: str = "cross_attn",
        **flamingo_kwargs,
    ):
        # ... existing setup code ...

        model = TimeSeriesFlamingoWithTrainableEncoder(
            SimpleNamespace(visual=time_series_encoder),
            lang_encoder,
            text_tokenizer.encode("<|endofchunk|>")[-1],
            text_tokenizer.encode("<image>")[-1],
            vis_dim=ENCODER_OUTPUT_DIM,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            # NEW: Pass query-aware options
            query_conditioning=query_conditioning,
            query_fusion_method=query_fusion_method,
            **flamingo_kwargs,
        )

        # ... rest of initialization ...

        # If query conditioning is enabled, also unfreeze query conditioning parameters
        if query_conditioning:
            # Unfreeze the new query conditioning modules
            for name, param in model.perceiver.named_parameters():
                if "query" in name or "fusion" in name:
                    param.requires_grad = True
```

#### File 4: Configuration Updates

**Modify:** `OpenTSLM/src/opentslm/model_config.py` - Add query-aware flags:

```python
# Add to existing config
QUERY_AWARE_PERCEIVER = False  # Enable query-aware perceiver
QUERY_FUSION_METHOD = "cross_attn"  # Options: "cross_attn", "add", "gate"
```

#### File 5: `OpenTSLM/curriculum_learning.py`

**Add CLI arguments** for query-aware training:

```python
# Add to argument parser
parser.add_argument("--query_aware_perceiver", action="store_true",
                    help="Enable query-aware perceiver conditioning")
parser.add_argument("--query_fusion_method", type=str, default="cross_attn",
                    choices=["cross_attn", "add", "gate"],
                    help="Query fusion method for perceiver")

# Pass to model initialization
model = OpenTSLMFlamingo(
    device=args.device,
    llm_id=args.llm_id,
    query_conditioning=args.query_aware_perceiver,
    query_fusion_method=args.query_fusion_method,
    ...
)
```

### 2.4 Implementation Checklist

**Phase A: Core Implementation**
- [ ] Create `src/opentslm/model/perceiver/` directory
- [ ] Create `src/opentslm/model/perceiver/__init__.py`
- [ ] Create `src/opentslm/model/perceiver/query_aware_perceiver.py` with `QueryAwarePerceiverResampler` class
- [ ] Implement `_condition_latents_on_query()` with three fusion methods (cross_attn, add, gate)
- [ ] Modify `TimeSeriesFlamingoWithTrainableEncoder.__init__()` to accept `query_conditioning` and `query_fusion_method`
- [ ] Modify `TimeSeriesFlamingoWithTrainableEncoder.__init__()` to replace perceiver with `QueryAwarePerceiverResampler` when enabled
- [ ] Override `TimeSeriesFlamingoWithTrainableEncoder.forward()` to pass `lang_x` to `_encode_vision_x()`
- [ ] Override `TimeSeriesFlamingoWithTrainableEncoder.generate()` to pass `lang_x` to `_encode_vision_x()`
- [ ] Add `_extract_query_embedding()` method to `TimeSeriesFlamingoWithTrainableEncoder`
- [ ] Update `_encode_vision_x()` signature to accept `lang_x` parameter
- [ ] Modify `OpenTSLMFlamingo.__init__()` to accept and pass query-aware parameters
- [ ] Add CLI arguments to `curriculum_learning.py`
- [ ] Add configuration flags to `model_config.py`

**Phase B: Testing**
- [ ] Unit test: Verify `QueryAwarePerceiverResampler` forward pass with/without query
- [ ] Unit test: Verify output shape matches original `PerceiverResampler`
- [ ] Unit test: Verify query embedding extraction produces correct shape
- [ ] Integration test: End-to-end forward pass with query conditioning disabled
- [ ] Integration test: End-to-end forward pass with query conditioning enabled
- [ ] Gradient flow test: Verify backprop through query conditioning path

**Phase C: Backward Compatibility**
- [ ] Ensure `query_conditioning=False` produces identical results to baseline (default perceiver)
- [ ] Verify existing checkpoints load correctly (perceiver weights should be compatible)
- [ ] Add flag to enable/disable query-aware mode during training
- [ ] Document the new parameters in docstrings

---

## 3. Training Query-Aware Model

### 3.1 Training Configuration

**Experiment Setup:**

```bash
# Baseline (for comparison)
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage5_ecg_cot \
    --device cuda \
    --batch_size 4

# Query-Aware (with cross-attention fusion)
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage5_ecg_cot \
    --device cuda \
    --batch_size 4 \
    --query_aware_perceiver \
    --query_fusion_method cross_attn

# Query-Aware (with additive fusion)
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage5_ecg_cot \
    --device cuda \
    --batch_size 4 \
    --query_aware_perceiver \
    --query_fusion_method add

# Query-Aware (with gated fusion)
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage5_ecg_cot \
    --device cuda \
    --batch_size 4 \
    --query_aware_perceiver \
    --query_fusion_method gate
```

**NOTE:** Need to add CLI arguments to `curriculum_learning.py`:
```python
parser.add_argument("--query_aware_perceiver", action="store_true")
parser.add_argument("--query_fusion_method", type=str, default="cross_attn",
                    choices=["cross_attn", "add", "gate"])
```

### 3.2 Hyperparameter Tuning

**Parameters to Tune:**

1. **Query Fusion Method:**
   - `cross_attn`: Most expressive, but adds parameters
   - `add`: Simplest, minimal parameters
   - `gate`: Middle ground

2. **Learning Rates:**
   - Keep encoder/projector LR same as baseline: `2e-4`
   - Query conditioning module LR: Start with `1e-4` (lower than base)

3. **Query Pooling Strategy:**
   - Mean pooling (default)
   - Last token pooling
   - Attention-weighted pooling

4. **Latent Initialization:**
   - Start from baseline latents (transfer learning)
   - Random initialization

**Recommended Training Strategy:**
```python
# Stage 1-4: Train baseline (query-agnostic)
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --stages stage1_mcq stage2_captioning stage3_cot stage4_sleep_cot

# Stage 5: Fine-tune with query conditioning
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --stages stage5_ecg_cot \
    --query_aware_perceiver \
    --query_fusion_method cross_attn \
    --load_perceiver_from results/Llama3_2_1B/OpenTSLMFlamingo/stage4_sleep_cot/checkpoints/best_model.pt
```

### 3.3 Expected Training Time

**Hardware Assumptions:**
- GPU: NVIDIA A100 (40GB) or equivalent
- Batch size: 4
- Epochs: 60

**Estimated Time:**
- Baseline training: ~8-12 hours
- Query-aware training: ~10-14 hours (10-15% overhead from conditioning)

**Memory Requirements:**
- Baseline: ~16GB GPU memory
- Query-aware: ~18GB GPU memory (additional parameters)
- Use `--gradient_checkpointing` if OOM

---

## 4. Evaluation & Comparison

### 4.1 Metrics

**Primary Metrics (from `parse_ecg_qa_cot_data.py`):**

1. **F1 Score:** Macro F1 across all templates
2. **Accuracy:** Exact match accuracy
3. **Per-Template F1:** F1 score for each question template
4. **Supported Answer Rate:** % of predictions in valid answer set

**Additional Analysis Metrics:**

5. **Question-Type Breakdown:**
   - Diagnosis questions (e.g., "Is there LBBB?")
   - Measurement questions (e.g., "What is the heart rate?")
   - Comparison questions (e.g., "Which lead shows ST elevation?")
   - Temporal questions (e.g., "Is the QRS prolonged?")

6. **Attention Analysis:**
   - Visualize which ECG leads are attended to for different question types
   - Compare attention patterns: baseline vs. query-aware

7. **Compression Quality:**
   - Reconstruction loss: Can we reconstruct original TS from compressed representation?
   - Information retention: Mutual information between original and compressed

### 4.2 Evaluation Scripts

**Script 1: Generate Predictions**

```bash
# Baseline
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --stages stage5_ecg_cot \
    --eval_only \
    --device cuda

# Query-Aware
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --stages stage5_ecg_cot \
    --eval_only \
    --device cuda \
    --query_aware_perceiver \
    --query_fusion_method cross_attn
```

**Script 2: Parse and Calculate Metrics**

```bash
# Baseline
python evaluation/opentslm/ecg_qa_cot/parse_ecg_qa_cot_data.py \
    --input_path results/Llama3_2_1B/OpenTSLMFlamingo/stage5_ecg_cot/results/test_predictions.jsonl \
    --output_path results/baseline_metrics.json

# Query-Aware
python evaluation/opentslm/ecg_qa_cot/parse_ecg_qa_cot_data.py \
    --input_path results/Llama3_2_1B/OpenTSLMFlamingo_QueryAware/stage5_ecg_cot/results/test_predictions.jsonl \
    --output_path results/query_aware_metrics.json
```

**Script 3: Comparative Analysis**

Create `evaluation/compare_results.py`:

```python
#!/usr/bin/env python3
"""Compare baseline vs. query-aware results."""

import json
import pandas as pd
import matplotlib.pyplot as plt

def load_results(baseline_path, query_aware_path):
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(query_aware_path) as f:
        query_aware = json.load(f)
    return baseline, query_aware

def compare_overall_metrics(baseline, query_aware):
    print("=" * 80)
    print("OVERALL METRICS COMPARISON")
    print("=" * 80)

    metrics = ["f1_score", "accuracy"]
    data = []

    for metric in metrics:
        b_val = baseline["overall_metrics"][metric]
        qa_val = query_aware["overall_metrics"][metric]
        delta = qa_val - b_val
        delta_pct = (delta / b_val) * 100

        data.append({
            "Metric": metric.replace("_", " ").title(),
            "Baseline": f"{b_val:.4f}",
            "Query-Aware": f"{qa_val:.4f}",
            "Delta": f"{delta:+.4f}",
            "Delta (%)": f"{delta_pct:+.2f}%"
        })

    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()

def compare_per_template_metrics(baseline, query_aware):
    print("=" * 80)
    print("PER-TEMPLATE F1 COMPARISON (Top 10 Improvements)")
    print("=" * 80)

    improvements = []

    for template_id, b_metrics in baseline["per_template_metrics"].items():
        qa_metrics = query_aware["per_template_metrics"].get(template_id)
        if qa_metrics:
            b_f1 = b_metrics["f1"]
            qa_f1 = qa_metrics["f1"]
            delta = qa_f1 - b_f1

            improvements.append({
                "Template ID": template_id,
                "Question Type": b_metrics.get("question_type", "Unknown"),
                "Baseline F1": f"{b_f1:.3f}",
                "Query-Aware F1": f"{qa_f1:.3f}",
                "Improvement": f"{delta:+.3f}",
                "Count": b_metrics["count"]
            })

    # Sort by improvement
    improvements.sort(key=lambda x: float(x["Improvement"]), reverse=True)

    df = pd.DataFrame(improvements[:10])
    print(df.to_string(index=False))
    print()

def plot_comparison(baseline, query_aware, output_path="comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall metrics
    metrics = ["f1_score", "accuracy"]
    baseline_vals = [baseline["overall_metrics"][m] for m in metrics]
    query_aware_vals = [query_aware["overall_metrics"][m] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    axes[0].bar([i - width/2 for i in x], baseline_vals, width, label="Baseline")
    axes[0].bar([i + width/2 for i in x], query_aware_vals, width, label="Query-Aware")
    axes[0].set_xlabel("Metric")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Overall Metrics Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace("_", " ").title() for m in metrics])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Per-template F1 distribution
    template_ids = list(baseline["per_template_metrics"].keys())
    baseline_f1s = [baseline["per_template_metrics"][t]["f1"] for t in template_ids]
    query_aware_f1s = [query_aware["per_template_metrics"][t]["f1"] for t in template_ids]

    axes[1].scatter(baseline_f1s, query_aware_f1s, alpha=0.6)
    axes[1].plot([0, 1], [0, 1], 'r--', label="y=x (no improvement)")
    axes[1].set_xlabel("Baseline F1")
    axes[1].set_ylabel("Query-Aware F1")
    axes[1].set_title("Per-Template F1 Comparison")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline metrics JSON")
    parser.add_argument("--query_aware", required=True, help="Query-aware metrics JSON")
    parser.add_argument("--output_plot", default="comparison.png")
    args = parser.parse_args()

    baseline, query_aware = load_results(args.baseline, args.query_aware)

    compare_overall_metrics(baseline, query_aware)
    compare_per_template_metrics(baseline, query_aware)
    plot_comparison(baseline, query_aware, args.output_plot)
```

**Usage:**
```bash
python evaluation/compare_results.py \
    --baseline results/baseline_metrics.json \
    --query_aware results/query_aware_metrics.json \
    --output_plot results/comparison.png
```

### 4.3 Success Criteria

**Hypothesis Validation:**

✅ **Successful if:**
- Query-aware F1 > Baseline F1 by ≥3% (absolute)
- Improvement statistically significant (p < 0.05, paired t-test)
- At least 60% of question types show improvement

⚠️ **Partial Success if:**
- Query-aware F1 > Baseline F1 by 1-3%
- Specific question types show >5% improvement (e.g., diagnosis questions)

❌ **Unsuccessful if:**
- No significant difference (|ΔF1| < 1%)
- Query-aware performs worse than baseline

**Interpretation:**
- **Large improvement (>5%):** Query-agnostic compression is a severe bottleneck → Proceed to Phase 2 (Titans)
- **Moderate improvement (2-5%):** Compression is a bottleneck, but not critical → Consider Phase 2 or iterate on query conditioning
- **Small improvement (<2%):** 64-token compression is sufficient → Phase 2 may not be worth the complexity

---

## 5. Project Timeline

### Week 1: Baseline Replication & Environment Setup
- **Days 1-2:** Environment setup, dependency installation, data download
- **Days 3-4:** Run baseline training (stage 5) or download pretrained
- **Days 5-6:** Run baseline evaluation, verify metrics match paper
- **Day 7:** Document baseline results, identify any issues

**Deliverable:** Baseline metrics report, verified F1 ≈ 34.62%

### Week 2: Query-Aware Implementation
- **Days 1-3:** Implement `PerceiverResampler` modifications (all 3 fusion methods)
- **Days 4-5:** Modify Flamingo forward pass, add query extraction
- **Days 6-7:** Unit tests, integration tests, gradient flow verification

**Deliverable:** Working query-aware implementation, passing all tests

### Week 3: Training & Experimentation
- **Days 1-2:** Train query-aware model (cross-attention fusion)
- **Days 3-4:** Train query-aware model (additive fusion)
- **Days 5-6:** Train query-aware model (gated fusion)
- **Day 7:** Hyperparameter tuning (if time permits)

**Deliverable:** 3 trained query-aware models

### Week 4: Evaluation & Analysis
- **Days 1-2:** Run evaluation on all models, generate predictions
- **Days 3-4:** Parse results, calculate metrics, statistical analysis
- **Days 5-6:** Comparative analysis, attention visualization, write report
- **Day 7:** Buffer for revisions, prepare presentation

**Deliverable:** Final comparison report, recommendations for Phase 2

**Total Estimated Time:** 4 weeks (full-time) or 6-8 weeks (part-time)

---

## 6. Potential Issues & Mitigations

### Issue 1: Out-of-Memory (OOM) Errors
**Symptoms:** CUDA OOM during training
**Root Cause:** ECG data has 12 leads × 1000 samples = 12K samples per example
**Mitigations:**
- Enable gradient checkpointing: `--gradient_checkpointing`
- Reduce batch size: `--batch_size 2` or `--batch_size 1`
- Use mixed precision training (FP16/BF16)
- Use FSDP (Fully Sharded Data Parallel) if multi-GPU available

### Issue 2: Query Embedding Mismatch
**Symptoms:** Query embedding shape doesn't match perceiver input
**Root Cause:** Text sequence length varies, pooling strategy may be incorrect
**Mitigations:**
- Verify query embedding shape: should be `[B, dim]`
- Test with fixed-length padding first
- Use attention mask when pooling

### Issue 3: No Improvement Over Baseline
**Symptoms:** Query-aware F1 ≤ Baseline F1
**Root Causes:**
1. Query conditioning not learning (gradient flow issue)
2. Fusion method too weak (additive may not be expressive enough)
3. Learning rate too high (overfitting to query conditioning)

**Mitigations:**
1. Verify gradients flowing through query conditioning path
2. Try more expressive fusion (cross-attention)
3. Lower LR for query conditioning module: `1e-5` instead of `1e-4`
4. Add regularization (dropout on query projection)

### Issue 4: Training Instability
**Symptoms:** Loss spikes, NaN losses
**Root Causes:**
- Large gradients from query conditioning
- Poorly initialized query modules

**Mitigations:**
- Initialize query projection with small weights: `nn.init.xavier_uniform_(..., gain=0.01)`
- Clip gradients aggressively: `GRAD_CLIP_NORM = 0.5`
- Use gradient accumulation to stabilize training

### Issue 5: Backward Compatibility Broken
**Symptoms:** Baseline model fails to load or produces different results
**Root Cause:** Changed Flamingo forward signature without fallback
**Mitigations:**
- Add `query_conditioning=False` flag to disable new behavior
- Version models: save model type in checkpoint metadata
- Unit test: verify `query_conditioning=False` matches old behavior exactly

---

## 7. Expected Outcomes & Next Steps

### Scenario A: Large Improvement (ΔF1 > 5%)

**Interpretation:** Query-agnostic compression is a severe bottleneck.

**Next Steps:**
1. ✅ **Phase 1 validated** → Publish ablation study
2. 🚀 **Proceed to Phase 2** (Titans integration) with high confidence
3. Analyze which question types benefit most (guide Titans design)
4. Consider applying query-aware perceiver to other tasks (HAR, Sleep)

### Scenario B: Moderate Improvement (2% < ΔF1 < 5%)

**Interpretation:** Compression is a bottleneck, but not the only one.

**Next Steps:**
1. ✅ **Phase 1 validated** (marginal success)
2. 🤔 **Evaluate Phase 2 cost-benefit:**
   - If goal is production deployment → Proceed (Titans solves memory issues)
   - If goal is research only → Iterate on Phase 1 (try more fusion methods)
3. Profile model: identify other bottlenecks (LLM reasoning, projector)

### Scenario C: Small/No Improvement (ΔF1 < 2%)

**Interpretation:** 64-token compression is sufficient for current task.

**Next Steps:**
1. ❌ **Phase 1 hypothesis rejected**
2. 🔍 **Root cause analysis:**
   - Is query embedding quality poor? (Try different pooling strategies)
   - Is fusion method too simple? (Try learnable attention pooling)
   - Is 64 latents already over-complete? (Try reducing to 32, measure impact)
3. 🔄 **Pivot strategy:**
   - Option A: Focus on LLM reasoning (LoRA, prompt engineering)
   - Option B: Focus on encoder quality (better TS representations)
   - Option C: Abandon query-aware compression, proceed to Phase 2 for memory benefits only

### Scenario D: Degradation (ΔF1 < 0)

**Interpretation:** Query conditioning hurts performance.

**Next Steps:**
1. 🐛 **Debug implementation:**
   - Check gradient flow (use `torch.autograd.grad`)
   - Verify query embedding quality (visualize with t-SNE)
   - Check for catastrophic forgetting (is base model knowledge lost?)
2. 🔧 **Fix or abandon:**
   - Try freezing base latents, only train query conditioning
   - Reduce query influence (lower gate values)
   - If unfixable → Document failure, revert to baseline

---

## 8. Code Organization

### Suggested Directory Structure

```
OpenTSLM/                              # Project root
├── src/opentslm/
│   ├── model/
│   │   ├── perceiver/                 # NEW: Query-aware perceiver module
│   │   │   ├── __init__.py            # NEW: Exports QueryAwarePerceiverResampler
│   │   │   └── query_aware_perceiver.py  # NEW: QueryAwarePerceiverResampler class
│   │   ├── llm/
│   │   │   ├── OpenTSLMFlamingo.py    # MODIFIED: Add query_conditioning params
│   │   │   └── TimeSeriesFlamingoWithTrainableEncoder.py  # MODIFIED: Override perceiver
│   │   └── encoder/
│   │       └── CNNTokenizer.py        # Unchanged
│   ├── model_config.py                # MODIFIED: Add QUERY_AWARE_PERCEIVER flag
│   └── ...
├── evaluation/
│   ├── opentslm/
│   │   └── ecg_qa_cot/
│   │       └── parse_ecg_qa_cot_data.py  # Unchanged
│   └── compare_results.py             # NEW: Comparison script
├── curriculum_learning.py             # MODIFIED: Add CLI args
├── test/
│   └── test_query_aware_perceiver.py  # NEW: Unit tests
│
├── PHASE1_IMPLEMENTATION_PLAN.md      # This document
├── RESEARCH.md                        # Research context
├── OpenTSLM-TITANS.md                 # Technical notes
│
└── experiments/                       # NEW: Experiment tracking
    ├── baseline/
    │   ├── config.yaml
    │   ├── metrics.json
    │   └── model.pt
    ├── query_aware_crossattn/
    │   ├── config.yaml
    │   ├── metrics.json
    │   └── model.pt
    ├── query_aware_add/
    └── query_aware_gate/

# NOTE: open_flamingo is installed via pip and NOT modified
# venv/lib/python3.12/site-packages/open_flamingo/  <- DO NOT EDIT
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/query-aware-perceiver

# Step 1: Create new perceiver module
git add src/opentslm/model/perceiver/__init__.py
git add src/opentslm/model/perceiver/query_aware_perceiver.py
git commit -m "feat: Add QueryAwarePerceiverResampler with cross_attn/add/gate fusion"

# Step 2: Modify TimeSeriesFlamingoWithTrainableEncoder
git add src/opentslm/model/llm/TimeSeriesFlamingoWithTrainableEncoder.py
git commit -m "feat: Override perceiver with query-aware version, pass lang_x for conditioning"

# Step 3: Update OpenTSLMFlamingo to accept query-aware params
git add src/opentslm/model/llm/OpenTSLMFlamingo.py
git commit -m "feat: Add query_conditioning and query_fusion_method params to OpenTSLMFlamingo"

# Step 4: Add CLI arguments
git add curriculum_learning.py
git commit -m "feat: Add --query_aware_perceiver and --query_fusion_method CLI args"

# Step 5: Add tests
git add test/test_query_aware_perceiver.py
git commit -m "test: Add unit tests for QueryAwarePerceiverResampler"

# Merge to main after validation
git checkout main
git merge feature/query-aware-perceiver
```

---

## 9. Summary Checklist

### Prerequisites
- [ ] OpenTSLM repository cloned
- [ ] Dependencies installed (`uv sync --all-groups`)
- [ ] WFDB library installed (`pip install wfdb`)
- [ ] Hugging Face authentication configured
- [ ] GPU with ≥16GB VRAM available
- [ ] PTB-XL ECG dataset downloaded

### Baseline Replication
- [ ] Baseline training completed (60 epochs)
- [ ] Baseline evaluation metrics calculated
- [ ] Baseline F1 ≈ 34.62% ± 2% verified
- [ ] Baseline results documented

### Implementation
- [ ] Create `src/opentslm/model/perceiver/` directory structure
- [ ] Implement `QueryAwarePerceiverResampler` class with all fusion methods
- [ ] Query fusion methods implemented (cross_attn, add, gate)
- [ ] `TimeSeriesFlamingoWithTrainableEncoder` modified to use custom perceiver
- [ ] Override `forward()` and `generate()` to pass `lang_x` for query conditioning
- [ ] Query embedding extraction implemented (`_extract_query_embedding()`)
- [ ] `OpenTSLMFlamingo` updated to accept query-aware parameters
- [ ] CLI arguments added to `curriculum_learning.py`
- [ ] Configuration flags added to `model_config.py`
- [ ] Unit tests written and passing
- [ ] Backward compatibility verified

### Training
- [ ] Query-aware model trained (cross-attention)
- [ ] Query-aware model trained (additive)
- [ ] Query-aware model trained (gated)
- [ ] Training logs reviewed for stability
- [ ] Checkpoints saved correctly

### Evaluation
- [ ] Predictions generated for all models
- [ ] Metrics calculated (F1, accuracy, per-template)
- [ ] Comparative analysis completed
- [ ] Statistical significance tested
- [ ] Attention patterns visualized
- [ ] Results report written

### Next Steps
- [ ] Phase 1 results presented
- [ ] Decision made: Proceed to Phase 2 or iterate?
- [ ] Lessons learned documented
- [ ] Code merged to main branch

---

## 10. Contact & Support

**For Questions:**
- Technical issues: Check OpenTSLM GitHub issues
- Dataset questions: Refer to PTB-XL and ECG-QA documentation
- Titans questions: Refer to Titans paper (arXiv:2501.00663)

**Useful Resources:**
- OpenTSLM Paper: https://doi.org/10.13140/RG.2.2.14827.60963
- Titans Paper: https://arxiv.org/abs/2501.00663
- PTB-XL Dataset: https://physionet.org/content/ptb-xl/
- ECG-QA Dataset: https://github.com/Jwoo5/ecg-qa

---

**Document Version:** 1.0
**Last Updated:** 2026-01-16
**Author:** Claude Code (Sonnet 4.5)
