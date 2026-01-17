# Purpose

OpenTSLM and other TSLMs suffer from context saturation and out of memory errors. If we want to move this to production, we have to find a way to compress information or set a kind of compressed memory from which we can retrieve from. This is also useful for long running surveillance tasks (monitoring, security footage, etc.)

Goal: Query-aware associative memory for OpenTSLM.

_Put simply_: Based on the user query, what are the regions of the time series I need to answer the question.


## Theoretical Background - OpenTSLM

### Attention Transformers and Soft Prompting variant

Attending to specific tokens in a sequence is done by OpenTSLM (@OpenTSLM/README.md) by combining time series data with the query in a transformer architecture (two variants exist - LLaVA-style soft prompting and Flamingo-style Cross-Attention with a resampler).

The Soft Prompting variant first uses a `Conv1d patch embedding: (B, 1, L) -> (B, embed_dim, L/patch_size)` to extract local features into patches and the passes those patches to a Transformer using the `nn.TransformerEncoderLayer` which uses self attention (flash attention) + ff layers.

Then the TS tokens are passed through the LLM with concatenated text + time-series tokens. Effectively there are two rounds of attention, one with and one without attending to the query.

patches = self.patch_embed(ts_padded)      # [N_patches, embed_dim]
query_emb = get_query_embedding(lang_x)
ts_compressed = query_perceiver(patches, query_emb)  # Cross-attention: O(64 × N_patches)

The results in the benchmarks are better for the Soft-prompting variant, the problem is they run out of memory a lot, and with longer context, it performs worse.

### Flamingo style variant

The Flamingo variant of OpenTSLM uses a preceiver resampler:

@open_flamingo/open_flamingo/src/flamingo.py
@OpenTSLM/src/opentslm/model/llm/OpenTSLMFlamingo.py
open_flamingo/open_flamingo/src/helpers.py -- self.latents = [...] _line 82_

The perceiver resampler uses (**cross-attention** + ff) layers of depty 6 and input shape (b, T, F, v, D) and output shape (b, T, n, D) where n is self.num_latents and n << v, as well as n being fixed independent of length of v.

Cross-attention means:
  - Q (queries) come from fixed learnable latent vectors (64 by default)
  - K, V (keys/values) come from variable-length input
  - Latents "query" the input to extract relevant information
  - Output length = number of latents (always 64), regardless of input length
  - Fixed memory footprint

This enables to compress into fixed latents using attention the importances for the multi-modal input before passing it to the LLM, thus going from having an exploding memory requirement to a fixed size memory based on the nº of latents outputed by the resampler.

The 64 learnable latent vectors (self.latents) are the "queries" that extract information from the time series. They're:
  - Initialized randomly at helpers.py:82
  - Learned during training via backpropagation
  - Query-agnostic: same latents used for all user questions

Performance seems to be similar in both cases, showing that the sequence can indeed be compressed without loss of performance.

### The hypothesis

This resampling is not based on the user query. Our hypothesis is that for different user queries, the timeseries has different places in which the attention should be placed, hence providing a representation to mix in with the model (could both work with LLaVA style or X-Attention architectures).

### Ideas

#### Option A: Query-Conditional Perceiver

Inject query embeddings into Perceiver's learnable queries before compression. The 64 latent "slots" become query-biased.

Current setup:
  - Perceiver learns ONE generic compression via backprop from task loss
  - Same time series → always same compression, regardless of query
  - Learns "on average what's important" across all queries

Query-conditional perceiver:
  - Perceiver receives explicit query input and produces different compressions
  - Same time series → different compressions for different queries
  - "What is the trend?" → attends to low-frequency patterns
  - "What is the peak value?" → attends to local maxima

Make sure we are smart about gating mechanisms, hyperparameter values and clean modular architecture decisions - thinking step by step.

**4. Marginal Benefit Question** ⚠️ ⚠️

  The current architecture already does query-aware processing via cross-attention after compression. The question is: how much information is lost in the
  query-agnostic compression step?

  If the 64-token compression already preserves most relevant information, gains might be modest. If it's a severe bottleneck, gains could be large.

**Implementation Plan:**

1. **Modify PerceiverResampler class** (`open_flamingo/src/helpers.py:68-132`)
   - Add query embedding input parameter to `__init__` and `forward`
   - Replace fixed latents with query-conditioned latents:
     - Option 1: Cross-attention between query embeddings and base latents
     - Option 2: Additive combination of query embeddings and base latents
     - Option 3: Learned projection: `latents = MLP([base_latents; query_embeddings])`

2. **Add query encoder in Flamingo** (`flamingo.py:177-197`)
   - Extract query embeddings from `lang_x` before perceiver
   - Use LLM's text embeddings to encode the question tokens
   - Pool query embeddings (mean/CLS/last token) to get fixed-size representation

3. **Update `_encode_vision_x` method** (`flamingo.py:177-197`)
   - Use the existing `self.lang_encoder.get_input_embeddings()` rather than adding a new encoder
   - Pass query embeddings to perceiver: `vision_x = self.perceiver(vision_x, query_embeddings)`
   - Ensure query encoding happens before perceiver compression

4. **Maintain backward compatibility**
   - Make query conditioning optional with flag
   - Allow fallback to original query-agnostic mode

5. **Validate approach**
   - Compare performance on same tasks with/without query conditioning
   - Analyze attention patterns to verify query-specific compression

### Option B: Porting the titans neural memory into Time series language modelling

Implement a neural memory to store the time series as it comes into a fixed size memory.

Use the user query to retrieve information using memories = Memory(query)

In training, (time series, query) pairs. Time series is passed through a NN




# Core readings:

Titans: Learning to Memorize at Test Time https://arxiv.org/abs/2501.00663

OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text and Time-Series Data https://arxiv.org/abs/2510.02410
