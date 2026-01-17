# Regular Attention

Keys and Values are projection of the input, in our case the timeseries.

Queries will be text data and we will use X-attention.

By the formula, the similarity (dot product) of queries and keys is computed then scaled to [0, 1] and sum to 1 with softmax. Then that is multipled with the value, which is going to be the the representation of the timeseries. This weighted fusion of values is what provides the output.

The output is a matrix of the same shape as the Queries (Q).

# How memory works

During Training (Store):
  M(key) → value

During Inference (Retrieve):
  M(query) → retrieved_value

The MLP has learned to approximate the attention mechanism, so:
  - A new query → MLP forward pass → predicted value
  - The MLP has internalized the key-value patterns from training
  - It bypasses attention computation by having learned the mapping
  - Replaces the explicit key-value storage of traditional Transformers

# During training

## Inner loop

The model learns a function to predict the values based on the keys of a sequence.

During the forward pass, for each chunk of tokens, the memory:
  1. Computes keys (k_t) and values (v_t) from the sequence
  2. Updates its MLP weights to minimize |M(k_t) - v_t|²
  3. Computes "surprises" (gradients) from prediction errors
  4. These surprises flow back to train the outer network

**Note**:

## Outer loop

While the memory's weights are updated at test time (the "inner loop"), the projection matrices $W_Q$, $W_K$, and $W_V$ are learned hyperparameters optimized during the global training phase (the "outer loop").

# During inference

The Query is passed to the Memory and the values are retrieved for that specific memory. When it is time to retrieve information, the model uses the query $q_t = x_t W_Q$ as the input for a forward pass through the memory.

During inference:
  - Query → Neural Memory MLP → Retrieved Values
  - Retrieved Values are either:
    - MAC: Prepends the Memory tokens to the LLM "short memory" (attention) so it can attend to those tokens.
    - MAG: Used as gates to modulate attention outputs (controlling information flow)
    - MAL (Memory as a Layer): Added to the residual stream (enriching context with learned memories)
  - Then standard transformer processing (attention + FF) continues

## Why interchangeability of keys ($k$) during training and queries ($q$) during retrieval works:

Just as in standard Transformers where a query $q_t$ is compared to a key $k_i$ via a dot product to find similarity, the Titans model learns that a query $q_t$ representing a specific "search signal" should look similar to the keys $k_i$ it is meant to retrieve.

## Dimensions analysis: The neural memory outputs the same shape as the input

  This is exactly the same as classical attention:

  **Classical Attention:**
```py
  Q = x @ W_q  # [b, n, dim]
  K = x @ W_k  # [b, n, dim]
  V = x @ W_v  # [b, n, dim]

  # Multi-head attention
  out = MultiHeadAttention(Q, K, V)  # [b, n, dim]

  x = x + out  # Residual connection
```

  **Neural Memory (TITANS):**
```py
  queries = x @ W_q  # [b, n, dim]

  # Neural memory replaces attention computation
  out = NeuralMemory(queries)  # [b, n, dim]

  x = x + out  # Residual connection (same as attention!) !!!!! - this is different than the MAC variant explained in the paper - where we should prepend the memory tokens as context for the short term attention.
```

Let's use concrete numbers from the MAC transformer initialization:
  - dim = 512 (model dimension)
  - dim_head = 64 (per-head dimension)
  - heads = 8 (number of heads)
  - seq_len = n (sequence length)
  - batch = b

1. Initial Input (line 856-860 in neural_memory.py:856-860)

```py
  seq = self.retrieve_norm(seq)          # [b, n, 512]
  queries = self.to_queries(seq)          # [b, n, 512] @ [512, 512] = [b, n, 512]
```

to_queries projections: dim → dim_inner where dim_inner = dim_head * heads = 64 * 8 = 512

2. Split Into Heads (line 331, 335, 864 in neural_memory.py:331)

```py
  dim_inner = dim_head * heads  # 64 * 8 = 512
  queries = self.split_heads(queries)  # [b, n, 512] → [b, 8, n, 64] # Rearrange('b n (h d) -> b h n d', h=8)
```

# Porting the the OpenTSLM paradigm - using MAC variant

To apply the Titans architecture to a time-series question-answering (QA) task, we would leverage its "neural long-term memory" module as a dynamic database that **updates its own weights as it "reads" the time-series patches/embeddings** --> good for long running monitoring tasks.

**The Projection Strategy**: Before reaching the memory, any input $x_t$ (whether a time-series slice or a text token) is projected into the same dimensionality.

**The "Outer-Loop" Training**: During the global training phase (the "outer loop"), the model learns the projection matrices $W_Q, W_K,$ and $W_V$2. If the model is trained on a task like "Time-Series Question Answering," these matrices will learn to map the text query and the time-series keys into a shared semantic space where they are mathematically compatible3

## 1. The "Write" phase

The time-series is processed and stored in the Long-term Memory (the MLP).

**Projection**: Each time-series segment $x_t$ is projected into a Key ($k_t$) and a Value ($v_t$) using learnable matrices ($W_K, W_V$).

**Test-Time Training (TTT)**: The memory module (an MLP) is updated using a "surprise" metric. If a new data point is unexpected (high gradient), the memory updates its weights more significantly to "memorize" that event.

**Associative Mapping**: The memory learns a function where $M(k_t) \approx v_t$. Effectively, the weights of the MLP now "store" the patterns of the time-series.

## 2. The "Read" phase

When you ask a question about the time-series (e.g., "What was the maximum value during the power surge?"), the question acts as the retrieval signal. The Question (acting as the current segment in titans) is used to query the MLP to retrieve historical context ($h_t$).

**Query Generation**: The question is projected into a Query ($q$) vector

**Neural Retrieval**: This $q$ is passed through the trained MLP: $y = M^*(q)$. Because the MLP was trained to map keys (time-series contexts) to values (data), the query "activates" the weights representing the relevant past information.

**Output**: The resulting vector $y$ is a compressed "memory" of the specific time-series segments relevant to the question.
**Attention Short term memory:** The Short-term Memory (Attention) receives a concatenated input: [Persistent Memory + Retrieved Context + Question]. The LLM uses its standard attention to "reason" across the retrieved facts and the current question to generate the final answer.

## Challenges in the integration

  1. **Sequence vs. Patches**:
      - TITANS processes token sequences --> Time-series needs patch embeddings from Conv1d (already exists in OpenTSLM, so we can use that).

  2. **Cross-Modal Queries**:
      - Question (text) queries time-series memory
      - W_Q must map text embeddings to time-series embedding space
      - May need alignment loss or shared projection space

  3. **Chunk Size Selection**:
     - Time-series may have variable-length relevant segments (although I am not very concerned about this right now, it can be tuned).
     - Need to tune neural_memory_segment_len for your data
     - Trade-off: smaller chunks = more updates, but less context per chunk

## Architecture Flow

  1. **Time-Series Encoder** (keep existing OpenTSLM patch embedding)
     TS: [B, 1, L] → Conv1d → [B, embed_dim, L/patch_size] → Transformer → [B, N_patches, dim]

  2. **Neural Memory Layer** (replace Perceiver or add after)
     TS_patches: [B, N_patches, dim]
     ↓ W_K, W_V (learned during training)
     Keys: [B, N_patches, dim]
     Values: [B, N_patches, dim]
     ↓ NeuralMemory.store_memories()
     Memory weights updated via TTT

  3. **Question Processing**
     Question: [B, N_text, dim_text]
     ↓ W_Q (learned projection)
     Queries: [B, N_text, dim]
     ↓ NeuralMemory.retrieve_memories()
     Retrieved: [B, N_text, dim]

  4. **Cross-Attention or Concatenation**
     [Question + Retrieved_Memory] → LLM → Answer

## Key files
[def forward_and_loss() @titans/titans_pytorch/neural_memory.py]

## Read these files to understand integration points:
  - OpenTSLM/src/opentslm/model/llm/OpenTSLMFlamingo.py - Current perceiver location
  - open_flamingo/src/helpers.py - Perceiver resampler implementation

# Next Steps
