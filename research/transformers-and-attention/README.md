# Transformers & Attention: How Tokens Talk to Each Other

**Tutorial by:** Koyilbek Valiev  
**Topics:** Deep Learning · Transformers · Attention · RoPE · Architecture

---

## Why Transformers?

Before transformers, sequences were processed with RNNs — one token at a time, left to right. The hidden state had to carry information across hundreds of steps. Long-range dependencies (the word at position 1 influencing position 200) were systematically forgotten.

Transformers threw that out entirely. Every token looks at every other token **simultaneously**, in a single operation. No sequential bottleneck, no forgetting.

```
RNN (sequential, forgets long-range):

  h₀ → h₁ → h₂ → h₃ → ... → hₙ
  x₀   x₁   x₂   x₃         xₙ

  Position 0 must pass through n steps to reach position n.
  Information decays with distance.

Transformer (parallel, full context):

  x₀  x₁  x₂  x₃  ...  xₙ
   ↕   ↕   ↕   ↕         ↕
  Every token attends to every other token directly.
  Distance is irrelevant.
```

This is what made GPT, BERT, LLaMA, and ViT possible.

---

## Part 1 — The Architecture at a Glance

A transformer has two logical halves:

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRANSFORMER                                 │
│                                                                 │
│   ENCODER                        DECODER                        │
│   (reads input)                  (generates output)            │
│                                                                 │
│  ┌─────────────┐                ┌─────────────────────────┐    │
│  │ Input Embed │                │ Output Embed (shifted)  │    │
│  │ + Pos Enc   │                │ + Pos Enc               │    │
│  └──────┬──────┘                └──────────┬──────────────┘    │
│         │                                  │                    │
│  ┌──────▼──────┐  ×N            ┌──────────▼──────────────┐    │
│  │  Self-Attn  │                │  Masked Self-Attn       │    │
│  │  FFN        │                │  Cross-Attn (← encoder) │    │
│  │  LayerNorm  │                │  FFN                    │    │
│  └──────┬──────┘                └──────────┬──────────────┘    │
│         │                                  │                    │
│  ┌──────▼──────┐                ┌──────────▼──────────────┐    │
│  │  Encoding   │ ─────────────▶ │  Linear + Softmax       │    │
│  └─────────────┘                └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Encoder-only** (BERT, ViT): reads and understands, no generation.  
**Decoder-only** (GPT, LLaMA): generates text autoregressively.  
**Encoder-Decoder** (original Transformer, T5): translation, summarization.

For Vision Transformers (ViT), only the encoder stack is used — the image patches are encoded and a classification head reads the `[CLS]` token.

---

## Part 2 — Scaled Dot-Product Self-Attention

### The Problem Attention Solves

Attention is **position-blind** by design — `[cat sat mat]` and `[mat cat sat]` look identical to a linear layer. Attention solves a different problem: it lets each token **dynamically decide which other tokens matter** for its own representation.

### Q, K, V — The Three Roles

Every token projects itself into three separate spaces:

```
Given input token x ∈ ℝᵈ:

  Q = x · Wq    "What am I looking for?"   (Query)
  K = x · Wk    "What do I contain?"       (Key)
  V = x · Wv    "What do I output?"        (Value)

  Wq, Wk, Wv ∈ ℝᵈˣᵈᵏ   (learned weight matrices)
```

The analogy: Q is a search query, K is a document index, V is the document content. You compute similarity between Q and every K, then fetch the corresponding V weighted by that similarity.

### The Formula

```
                  Q · Kᵀ
Attention(Q,K,V) = softmax( ─────── ) · V
                              √d_k

Where:
  Q ∈ ℝᴺˣᵈᵏ    N tokens, each with a d_k-dim query
  K ∈ ℝᴺˣᵈᵏ    N tokens, each with a d_k-dim key
  V ∈ ℝᴺˣᵈᵛ    N tokens, each with a d_v-dim value

  Q · Kᵀ  ∈ ℝᴺˣᴺ   raw similarity scores (one score per token pair)
  √d_k            scaling factor
  softmax(...)    convert scores to probabilities (row-wise, sum to 1)
  (·) · V         weighted sum of values
```

### Why Divide by √d_k?

```
If d_k = 64:
  dot products grow as ≈ √64 = 8 on average (variance of QK entries)

Without scaling:
  softmax([30, 0.1, 0.2, 0.05])
  → [≈1.0, ≈0.0, ≈0.0, ≈0.0]   ← one token dominates, gradients vanish

With √d_k scaling:
  softmax([30/8, 0.1/8, 0.2/8, 0.05/8])
  → [0.98, 0.007, 0.008, 0.006]  ← still peaked but gradient-safe
```

Large dot products push softmax into near-zero gradient regions. The √d_k fix keeps learning stable regardless of dimension size.

### Concrete Walkthrough (3 tokens, d_k=2)

```
Tokens: ["cat", "sat", "on"]  →  already embedded

Q = [[1, 0],    K = [[1, 0],    V = [[0.9, 0.1],
     [0, 1],         [0, 1],         [0.1, 0.9],
     [1, 1]]         [0, 0]]         [0.5, 0.5]]

Step 1: Raw scores = Q · Kᵀ
                         K_cat  K_sat  K_on
  Q_cat · all K  → [  1·1+0·0,  1·0+0·1,  1·0+0·0 ] = [1, 0, 0]
  Q_sat · all K  → [  0·1+1·0,  0·0+1·1,  0·0+1·0 ] = [0, 1, 0]
  Q_on  · all K  → [  1·1+1·0,  1·0+1·1,  1·0+1·0 ] = [1, 1, 0]

  scores = [[1, 0, 0],
            [0, 1, 0],
            [1, 1, 0]]

Step 2: Scale by √d_k = √2 ≈ 1.41
  scores /= 1.41
  → [[0.71, 0,    0   ],
     [0,    0.71, 0   ],
     [0.71, 0.71, 0   ]]

Step 3: Softmax row-wise
  row 0: softmax([0.71, 0, 0])   ≈ [0.52, 0.24, 0.24]
  row 1: softmax([0,  0.71, 0])  ≈ [0.24, 0.52, 0.24]
  row 2: softmax([0.71, 0.71,0]) ≈ [0.39, 0.39, 0.22]

Step 4: Weighted sum of V
  output_cat = 0.52·V_cat + 0.24·V_sat + 0.24·V_on
             = 0.52·[0.9,0.1] + 0.24·[0.1,0.9] + 0.24·[0.5,0.5]
             = [0.59, 0.41]

"cat" updated its representation by blending in info from "sat" and "on".
```

### The N×N Attention Matrix

```
         K_cat  K_sat  K_on
Q_cat  [ 0.52   0.24   0.24 ]   ← cat mostly attends to itself
Q_sat  [ 0.24   0.52   0.24 ]   ← sat mostly attends to itself
Q_on   [ 0.39   0.39   0.22 ]   ← on splits between cat and sat

This N×N matrix is the computational bottleneck:
  Memory:  O(N²)
  Compute: O(N²·d)

For N=197 (ViT 16×16 patches):   197×197 = 38,809 entries per head
For N=4096 (long text):           4096×4096 = 16M entries per head
```

---

## Part 3 — Multi-Head Attention

### One Head Is Not Enough

A single attention head learns one type of relationship. But language and vision have many:

```
"The animal didn't cross the street because it was too tired."
  ↑ syntactic: "animal" ↔ "it" (coreference)
  ↑ semantic:  "cross" ↔ "street" (verb-object)
  ↑ positional: "because" ↔ "tired" (causal link)
```

Multi-head attention runs `h` attention functions in parallel, each in a lower-dimensional subspace.

### The Mechanism

```
d_model = 512,  h = 8  →  d_k = d_v = d_model/h = 64

For each head i:
  Qᵢ = X · Wqᵢ      Wqᵢ ∈ ℝᵈᵐˣᵈᵏ
  Kᵢ = X · Wkᵢ      Wkᵢ ∈ ℝᵈᵐˣᵈᵏ
  Vᵢ = X · Wvᵢ      Wvᵢ ∈ ℝᵈᵐˣᵈᵛ

  headᵢ = Attention(Qᵢ, Kᵢ, Vᵢ)   ∈ ℝᴺˣᵈᵛ

Concatenate all heads:
  MultiHead(X) = Concat(head₁, ..., headₕ) · Wₒ

  Wₒ ∈ ℝ⁽ʰ·ᵈᵛ⁾ˣᵈᵐ  (projects back to d_model)
```

Visually:

```
Input X  (N × d_model)
   │
   ├── Wq₁, Wk₁, Wv₁ → head₁  ──┐
   ├── Wq₂, Wk₂, Wv₂ → head₂  ──┤
   ├── Wq₃, Wk₃, Wv₃ → head₃  ──┤  Concat  →  (N × h·d_v)  →  Wₒ  →  Output
   ├── ...                     ──┤
   └── Wqₕ, Wkₕ, Wvₕ → headₕ  ──┘

Each head: (N × d_k) · (d_k × N) = (N × N) scores
Total params: h × (3·d_model·d_k) + d_model² ≈ 4·d_model²
```

### Parameter Count (ViT-Base)

```
d_model = 768,  h = 12,  d_k = d_v = 64

Per head: Wq, Wk, Wv each 768×64   →  3 × 768×64 = 147,456
All heads: 12 × 147,456            = 1,769,472
Output proj Wₒ: 768×768            =   589,824

Total MHA params: ~2.36M per block
```

---

## Part 4 — Position Encoding

Attention has no concept of order — shuffling tokens gives identical scores. Position encoding injects sequence position into the representation.

### Generation 1: Sinusoidal (Vaswani et al. 2017)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Add to embedding: x[pos] = embedding[pos] + PE[pos]
```

Fixed, no learned parameters. Works but doesn't extrapolate well.

### Generation 2: Learned Absolute (BERT, ViT)

```
pos_embed = nn.Embedding(max_len, d_model)   # trained table

x[pos] = embedding[pos] + pos_embed[pos]
```

Flexible but hard ceiling: can't generalize beyond `max_len` seen during training.

### Generation 3: RoPE — Rotary Position Embedding

RoPE doesn't *add* position — it **rotates** Q and K vectors. The rotation angle depends on position.

**Core idea:** for a 2D pair (x₁, x₂) at position m:

```
R(mθ) · [x₁, x₂]ᵀ = [x₁·cos(mθ) - x₂·sin(mθ),
                       x₁·sin(mθ) + x₂·cos(mθ)]
```

For d-dimensional Q/K, split into d/2 pairs. Each pair gets its own base frequency:

```
θᵢ = 1 / 10000^(2i/d)      i = 0, 1, ..., d/2 - 1

Pair 0: rotates fast   (high freq, captures fine position)
Pair 1: rotates slower
...
Pair d/2-1: rotates very slow (low freq, captures global structure)
```

**The key property:**

```
⟨R(m)·q, R(n)·k⟩ = qᵀ · R(m)ᵀ · R(n) · k
                  = qᵀ · R(n-m) · k          ← only (n-m) matters!

Proof: rotation matrices compose:  R(a)ᵀ · R(b) = R(b-a)
```

The dot product depends **only on relative distance**, not absolute positions. Token 5 attending to token 10 behaves identically to token 105 attending to token 110.

**Visual intuition:**

```
Absolute PE:         RoPE:
  pos 1 → [0.84, 0.54, ...]    pos 1 → rotates Q,K by angle 1·θᵢ
  pos 2 → [0.91, 0.41, ...]    pos 2 → rotates Q,K by angle 2·θᵢ
  (added to embedding)         (baked into attention scores)

  Attention sees absolute       Attention sees relative
  position of each token        distance between tokens
```

### Comparison

```
┌──────────────────┬────────────┬──────────┬─────────────────┬───────────────────────┐
│                  │ Params     │ Relative │ Extrapolation   │ Used in               │
├──────────────────┼────────────┼──────────┼─────────────────┼───────────────────────┤
│ Sinusoidal       │ None       │ No       │ Poor            │ Original Transformer  │
│ Learned Abs.     │ max_len×d  │ No       │ None (hard cap) │ BERT, ViT, GPT-2      │
│ Relative (Shaw)  │ Few        │ Yes      │ Moderate        │ T5, Longformer        │
│ RoPE             │ None       │ Yes      │ Good            │ LLaMA, Mistral, Gemma │
└──────────────────┴────────────┴──────────┴─────────────────┴───────────────────────┘
```

---

## Part 5 — The Transformer Encoder Block

This is the unit that is stacked N times (12× in ViT-Base, 32× in LLaMA-7B).

### Structure

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        self.attn  = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.mlp   = nn.Sequential(nn.Linear(embed_dim, mlp_dim),
                                   nn.GELU(),
                                   nn.Linear(mlp_dim, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # Pre-Norm attention + residual
        x = x + self.mlp(self.norm2(x))    # Pre-Norm FFN + residual
        return x
```

Data flow:

```
Input x  (B, N, d)
   │
   │  ┌──────────────────────────────────────────┐
   │  │  norm1(x)  →  MultiHeadAttn  →  output   │
   └──┤                                           ├──▶ x = x + output
      └──────────────────────────────────────────┘
   │
   │  ┌──────────────────────────────────────────┐
   │  │  norm2(x)  →  Linear → GELU → Linear     │
   └──┤                                           ├──▶ x = x + output
      └──────────────────────────────────────────┘
   │
Output x  (B, N, d)   ← same shape as input
```

### Pre-Norm vs Post-Norm

```
Post-Norm (original 2017):  x = LayerNorm(x + Sublayer(x))
  → Normalize AFTER adding residual
  → Gradient magnitudes uncontrolled deep in network
  → Hard to train beyond ~6 layers without warmup tricks

Pre-Norm (modern default):  x = x + Sublayer(LayerNorm(x))
  → Normalize BEFORE the sublayer
  → Residual path x is always unnormalized → gradient highway
  → Stable for 100+ layer models
```

**Why the residual (`x + ...`) matters:**

```
Without residual:
  Layer 1: x₁ = f₁(x₀)
  Layer 2: x₂ = f₂(x₁)        If f₁ or f₂ outputs near-zero → vanishing gradient

With residual:
  Layer 1: x₁ = x₀ + f₁(x₀)
  Layer 2: x₂ = x₁ + f₂(x₁)   ∂x₂/∂x₀ = 1 + ... (always at least 1)
                                → gradient always flows through the shortcut
```

### The FFN / MLP

```
Linear(d → 4d) → GELU → Linear(4d → d)

In ViT-Base: d=768 → 3072 → 768

Purpose: after attention mixes information ACROSS tokens,
         the FFN processes each token INDEPENDENTLY.

  Token 0: [768-dim] → expand to [3072] → activate → compress to [768]
  Token 1: same operation, different values, same weights
  ...
  Token N: same operation

Attention = communication   (tokens talk to each other)
FFN       = computation     (each token thinks on its own)
```

GELU (Gaussian Error Linear Unit) is the standard activation here — smoother than ReLU, differentiable everywhere, empirically better for transformers.

---

## Part 6 — Putting It Together: Vision Transformer

For ViT, the input is an image, not a token sequence. The patch embedding layer converts an image into tokens:

```
Image: (B, 3, 224, 224)

Step 1 — Patch Embedding (Conv2d):
  nn.Conv2d(3, 768, kernel_size=16, stride=16)
  → (B, 768, 14, 14)   ← 14×14 = 196 patches

  kernel_size=16 means: process 16×16 pixels at once
  stride=16 means: jump 16 pixels each step (no overlap)

Step 2 — Flatten spatial dims:
  (B, 768, 14, 14) → (B, 196, 768)

Step 3 — Prepend [CLS] token:
  cls_token = nn.Parameter(torch.zeros(1, 1, 768))
  x = cat([cls_token, patches], dim=1)
  → (B, 197, 768)

Step 4 — Add positional embedding:
  pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
  x = x + pos_embed

Step 5 — Pass through N Transformer Encoder Blocks:
  for block in self.blocks:
      x = block(x)

Step 6 — Classification from [CLS]:
  cls_output = x[:, 0]          # (B, 768)
  logits = self.head(cls_output) # (B, num_classes)
```

The CLS token attends to all 196 patch tokens across all layers and accumulates a global image representation.

---

## Part 7 — PyTorch Implementations

### Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
) -> torch.Tensor:
    """
    Q: (B, H, N, d_k)
    K: (B, H, N, d_k)
    V: (B, H, N, d_v)
    returns: (B, H, N, d_v)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)

    if dropout > 0.0:
        weights = F.dropout(weights, p=dropout)

    return torch.matmul(weights, V)
```

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, d = x.shape

        # Project to Q, K, V
        Q = self.Wq(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (B, num_heads, N, d_k)

        # Attention
        out = scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        # Concat heads: (B, num_heads, N, d_k) → (B, N, embed_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, d)

        return self.Wo(out)
```

### RoPE

```python
def build_rope_freqs(d_k: int, max_len: int = 2048) -> torch.Tensor:
    i = torch.arange(0, d_k, 2, dtype=torch.float)
    theta = 1.0 / (10000 ** (i / d_k))                  # (d_k/2,)
    pos = torch.arange(max_len, dtype=torch.float)       # (max_len,)
    freqs = torch.outer(pos, theta)                      # (max_len, d_k/2)
    return torch.cat([freqs, freqs], dim=-1)             # (max_len, d_k)

def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    x:     (B, H, N, d_k)
    freqs: (N, d_k)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = freqs[:, :half].cos()
    sin = freqs[:, :half].sin()
    rotated = torch.cat([x1 * cos - x2 * sin,
                         x1 * sin + x2 * cos], dim=-1)
    return rotated
```

### Transformer Encoder Block

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attn  = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### Vision Transformer (Full)

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)              # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int   = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int  = 768,
        depth: int      = 12,
        num_heads: int  = 12,
        mlp_dim: int    = 3072,
        dropout: float  = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        x = self.patch_embed(x)                             # (B, N, d)
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, d)
        x = torch.cat([cls, x], dim=1)                     # (B, N+1, d)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])                           # classify from CLS token


# Instantiate ViT-Base/16
model = VisionTransformer(
    img_size=224, patch_size=16, embed_dim=768,
    depth=12, num_heads=12, mlp_dim=3072
)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# Parameters: 86,567,656  (~86M, matches official ViT-Base/16)
```

---

## Part 8 — Attention Variants

```
┌────────────────────┬──────────────────────────────┬────────────────────────────┐
│ Variant            │ Key Change                   │ Use Case                   │
├────────────────────┼──────────────────────────────┼────────────────────────────┤
│ Self-Attention     │ Q=K=V from same source       │ All transformers           │
│ Cross-Attention    │ Q from decoder, K/V encoder  │ Translation, T5            │
│ Masked Self-Attn   │ Mask future tokens           │ GPT, LLaMA (generation)    │
│ Multi-Query (MQA)  │ Share K,V across heads       │ Falcon, PaLM (fast decode) │
│ Grouped-Query(GQA) │ Group heads for K,V sharing  │ LLaMA-2/3, Mistral         │
│ Flash Attention    │ Tile-based GPU-efficient impl│ All modern training        │
│ Sliding Window     │ Each token attends locally   │ Longformer, Mistral long   │
└────────────────────┴──────────────────────────────┴────────────────────────────┘
```

---

## Part 9 — Key Numbers to Remember

```
ViT-Base/16:
  Image         224×224×3
  Patch size    16×16
  Patches       196  (+1 CLS = 197 tokens)
  embed_dim     768
  Depth         12 blocks
  Heads         12
  mlp_dim       3072  (4× embed_dim)
  Parameters    ~86M

ViT-Large/16:
  embed_dim     1024,  depth 24,  heads 16  →  ~307M params

GPT-2 Small (decoder-only transformer):
  embed_dim     768,  depth 12,  heads 12,  context 1024 tokens  →  ~117M params

LLaMA-2 7B:
  embed_dim     4096,  depth 32,  heads 32  (GQA: 8 KV heads)  →  7B params
  Uses RoPE, RMSNorm, SwiGLU — no biases in attention
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  Transformer Building Blocks                                    │
│                                                                 │
│  Self-Attention   every token attends to every other token     │
│  Q, K, V         query/key/value projections of same input     │
│  Scale by √d_k   prevents vanishing gradients in softmax       │
│  Multi-Head       parallel attention with different subspaces  │
│  Position Enc     sinusoidal → learned → RoPE (relative)       │
│  Pre-Norm         normalize before sublayer for stable depth   │
│  Residual         gradient highway, allows 100+ layer training │
│  FFN              per-token computation after cross-token attn │
│  CLS token        aggregates global info for classification    │
└─────────────────────────────────────────────────────────────────┘
```

The transformer is elegant precisely because it contains so few distinct ideas — attention, projection, normalization, residual — repeated deeply. Everything from ViT to LLaMA-3 is a variation on this same skeleton.
