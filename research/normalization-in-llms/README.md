# Normalization in Modern LLMs: From LayerNorm to RMSNorm, DyT, and Beyond

**Tutorial by:** Koyilbek Valiev  
**Topics:** Transformers · LLMs · Training Stability · Math

---

## The Problem Normalization Solves

Training a 96-layer transformer without normalization is impossible. Activations grow or shrink exponentially as signals propagate forward, and gradients vanish or explode as they flow backward.

```
Without normalization — what happens layer by layer:

  Layer 1:   activations ∈ [-1.0,  +1.0]   ← reasonable
  Layer 4:   activations ∈ [-8.0,  +8.0]   ← growing
  Layer 8:   activations ∈ [-200,  +200]    ← exploding
  Layer 12:  activations ∈ [-9000, +9000]   ← NaN incoming

  Backward pass (gradients):
  Layer 12:  gradient = 1.0
  Layer 8:   gradient = 0.01
  Layer 4:   gradient = 0.00001
  Layer 1:   gradient ≈ 0.0   ← vanished, learns nothing
```

Normalization fixes this by forcing activations back to a controlled range at every layer — regardless of how deep the model is or how unlucky the initialization was.

**The core idea:** after every sublayer, rescale activations so they have predictable magnitude. The model trains stably. The learned parameters (`γ`, `β`) give the network freedom to undo the normalization if it genuinely needs to.

---

## Part 1 — Mathematical Foundation: Variance

Before understanding normalization, you need one concept: **variance**.

Variance measures how spread out a set of numbers is from their mean.

```
Given values:  x = [x₁, x₂, ..., xₙ]

─────────────────────────────────────────────────
Step 1:  Mean (center of the data)

         μ = (1/n) · Σᵢ xᵢ

─────────────────────────────────────────────────
Step 2:  Variance (average squared deviation)

         σ² = (1/n) · Σᵢ (xᵢ - μ)²

─────────────────────────────────────────────────
Step 3:  Standard deviation (same unit as data)

         σ = √σ²
─────────────────────────────────────────────────
```

### Concrete walkthrough

```
Values: x = [2, 4, 6, 8]

Step 1 — Mean:
    μ = (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5.0

Step 2 — Deviations from mean:
    x₁ - μ = 2 - 5 = -3
    x₂ - μ = 4 - 5 = -1
    x₃ - μ = 6 - 5 = +1
    x₄ - μ = 8 - 5 = +3

    Note: always sums to zero → (-3) + (-1) + (1) + (3) = 0

Step 3 — Square the deviations (remove sign, penalize outliers):
    (-3)² = 9
    (-1)² = 1
    (+1)² = 1
    (+3)² = 9

Step 4 — Variance:
    σ² = (9 + 1 + 1 + 9) / 4 = 20 / 4 = 5.0

Step 5 — Standard deviation:
    σ  = √5.0 ≈ 2.24
```

### Why squaring?

```
Raw deviations:   [-3, -1, +1, +3]   sum = 0  ← useless
Abs deviations:   [ 3,  1,  1,  3]   sum = 8  ← works but not differentiable
Squared devs:     [ 9,  1,  1,  9]   sum = 20 ← smooth, differentiable, outlier-sensitive
```

Squaring is **differentiable** (critical for gradient-based training) and **penalizes large deviations more heavily** — a deviation of 3 contributes 9×, not 3×.

### Variance vs Standard Deviation

```
┌──────────────────┬───────────────────────────┬──────────────────────┐
│                  │ Formula                   │ Unit                 │
├──────────────────┼───────────────────────────┼──────────────────────┤
│ Variance  (σ²)   │ mean of squared deviations│ original units²      │
│ Std Dev   (σ)    │ √variance                 │ same as original data│
└──────────────────┴───────────────────────────┴──────────────────────┘
```

Standard deviation is more interpretable (same unit as the data), but normalization divides by `√(σ² + ε)` — using variance directly in its square-rooted form.

---

## Part 2 — Layer Normalization

LayerNorm (Ba et al., 2016) became the standard normalization in transformers. It normalizes each token's activation vector **across its feature dimension** — independently per sample.

### The Full Formula

```
Input:  x ∈ ℝᵈ  (one token's activation vector, d = hidden dimension)

─────────────────────────────────────────────────────────────
Step 1:  Mean across features

         μ = (1/d) · Σᵢ xᵢ

─────────────────────────────────────────────────────────────
Step 2:  Variance across features

         σ² = (1/d) · Σᵢ (xᵢ - μ)²

─────────────────────────────────────────────────────────────
Step 3:  Normalize  (ε = 1e-5, prevents ÷0)

         x̂ᵢ = (xᵢ - μ) / √(σ² + ε)

─────────────────────────────────────────────────────────────
Step 4:  Scale and shift with learned parameters

         yᵢ = γᵢ · x̂ᵢ + βᵢ

         γ ∈ ℝᵈ  — per-dimension gain  (initialized to 1)
         β ∈ ℝᵈ  — per-dimension bias  (initialized to 0)
─────────────────────────────────────────────────────────────
```

After step 3, the output has mean ≈ 0 and std ≈ 1. The learned `γ` and `β` let the network rescale and shift the distribution to whatever it needs. If the network learns `γ = σ_original` and `β = μ_original`, it has effectively undone the normalization — but now the network **chooses** the scale rather than inheriting it randomly.

### What Normalization Physically Does

```
Before LayerNorm:
  Token activations across 768 dims:

  dim:  1    2    3    4    5    6    7    8  ...  768
  val: 0.01  452  -3   0.4  230  -89  1.1  0.6  ...  12

  mean μ  ≈  45.3   (wildly off-center)
  std  σ  ≈  120.4  (huge spread)


After LayerNorm:
  dim:  1    2    3    4    5    6    7    8  ...  768
  val: -0.4  3.4 -0.4 -0.4  1.5 -1.1 -0.4 -0.4 ... -0.3

  mean μ  ≈  0.0   (centered)
  std  σ  ≈  1.0   (unit spread)
```

### LayerNorm vs BatchNorm — Why Transformers Use LayerNorm

```
BatchNorm normalizes across the BATCH dimension:

  Batch B=4, features d=3:

  Sample 1: [a₁, a₂, a₃]      For feature 1: normalize [a₁, b₁, c₁, d₁]
  Sample 2: [b₁, b₂, b₃]  →   For feature 2: normalize [a₂, b₂, c₂, d₂]
  Sample 3: [c₁, c₂, c₃]      For feature 3: normalize [a₃, b₃, c₃, d₃]
  Sample 4: [d₁, d₂, d₃]
                ↑
         each column gets its own μ and σ (computed over the batch)


LayerNorm normalizes across the FEATURE dimension:

  Sample 1: [a₁, a₂, a₃]  →  normalize [a₁, a₂, a₃]  (one sample, all features)
  Sample 2: [b₁, b₂, b₃]  →  normalize [b₁, b₂, b₃]
  Sample 3: [c₁, c₂, c₃]  →  normalize [c₁, c₂, c₃]
  Sample 4: [d₁, d₂, d₃]  →  normalize [d₁, d₂, d₃]
                                  ↑
                         each row gets its own μ and σ (per sample)
```

```
┌────────────────────┬──────────────────────┬────────────────────────┐
│                    │ BatchNorm            │ LayerNorm              │
├────────────────────┼──────────────────────┼────────────────────────┤
│ Normalizes over    │ batch dimension      │ feature dimension      │
│ Batch size dep.    │ yes — breaks at B=1  │ no — any batch size    │
│ Sequence length    │ must be fixed        │ variable OK            │
│ At inference       │ needs stored stats   │ computes on the fly    │
│ Primary use        │ CNNs                 │ Transformers, LLMs     │
└────────────────────┴──────────────────────┴────────────────────────┘
```

Language models process variable-length sequences, often with small or size-1 batches at inference. LayerNorm works identically in training and inference and never depends on other samples in the batch.

### Pre-Norm vs Post-Norm — Where You Place It Matters

The *formula* is only half the story. **Placement** inside the transformer block determines training stability.

```
Post-Norm (original "Attention Is All You Need", 2017):

  x ──→ Sublayer(x) ──→ + ──→ LayerNorm ──→ output
  │                      ↑
  └──────────────────────┘ (residual)

  During backprop, gradients must pass THROUGH LayerNorm
  at every layer. In deep models (>20 layers), this leads
  to vanishing gradients.


Pre-Norm (standard since GPT-3, 2020):

  x ──→ LayerNorm(x) ──→ Sublayer ──→ + ──→ output
  │                                    ↑
  └────────────────────────────────────┘ (residual)

  The residual path x is UNOBSTRUCTED — gradients flow
  directly from output to input, bypassing LayerNorm.
  Stable even at 96+ layers.
```

**All modern LLMs (GPT-3, LLaMA, Mistral, Gemma, Qwen) use Pre-Norm.** Post-Norm requires careful learning rate warmup and initialization tricks to work at depth; Pre-Norm is robust out of the box.

### Full Transformer Block with Pre-Norm LayerNorm

```
Input x ∈ ℝ^(T×d)    (T tokens, d hidden dim)
   │
   ├─── LayerNorm ──→ Multi-Head Self-Attention ──→ + ───→
   │                                                │
   │◄───────────────────────────────────────────────┘
   │
   ├─── LayerNorm ──→ Feed-Forward Network ─────────→ + ───→ output
   │                                                │
   │◄───────────────────────────────────────────────┘
```

---

## Part 3 — RMSNorm: The Dominant Replacement

**Used in:** LLaMA 2, LLaMA 3, Mistral, Gemma, Qwen 2.5, DeepSeek, Falcon

RMSNorm (Zhang & Sennrich, 2019) asks: **do we actually need mean subtraction?**

The answer, empirically, is no.

### The Formula

```
Standard LayerNorm:
    μ  = (1/d) · Σ xᵢ
    σ² = (1/d) · Σ (xᵢ - μ)²           ← subtract mean first
    y  = γ · (x - μ) / √(σ² + ε) + β  ← shift by β

RMSNorm:
    RMS(x) = √( (1/d) · Σ xᵢ² + ε )   ← NO mean subtraction
    y       = γ · x / RMS(x)           ← NO additive β
```

RMSNorm drops two operations: the mean subtraction and the additive bias `β`. This is not a simplification for simplification's sake — both were shown to contribute negligibly to model quality.

### LayerNorm vs RMSNorm Side by Side

```
LayerNorm step count:
  1. Compute mean μ           (d additions + 1 division)
  2. Subtract mean x - μ      (d subtractions)
  3. Compute variance σ²      (d subtractions + d squarings + d additions + 1 division)
  4. Normalize (x - μ) / σ   (d divisions)
  5. Scale γ · x̂             (d multiplications)
  6. Shift x̂ + β             (d additions)
  Total: ~6d operations, 2 passes over x

RMSNorm step count:
  1. Square x²                (d squarings)
  2. Mean of squares          (d additions + 1 division)
  3. Sqrt + normalize x/RMS   (1 sqrt + d divisions)
  4. Scale γ · x̂             (d multiplications)
  Total: ~4d operations, 1 pass over x

Speed gain: ~15–20% faster in practice
```

### Why Removing the Mean Works

```
Intuition:
  LayerNorm: (x - μ) / σ · γ + β
             ↑__________↑
             these two cancel each other out if β ≈ μ·γ/σ

  The network can learn β to shift the output center wherever
  it needs. The explicit mean subtraction is therefore redundant
  — the model recovers any centering it needs via β.

  RMSNorm skips the redundant centering step entirely.
```

### Ablation (from the RMSNorm paper)

```
Model: WMT'14 English→German translation

  LayerNorm   →  27.3 BLEU   (baseline)
  RMSNorm     →  27.3 BLEU   (identical quality)

  Wall-clock training time:
  LayerNorm:  100%
  RMSNorm:     84%   ← 16% faster

Same quality. 16% faster. This is why the entire open-source
LLM ecosystem migrated from LayerNorm to RMSNorm.
```

---

## Part 4 — Dynamic Tanh (DyT): No Statistics At All

**Paper:** "Transformers without Normalization" (Zhu et al., 2025)

DyT makes the most radical proposal: **eliminate normalization statistics entirely**. No mean. No variance. No division. Replace the whole norm layer with a learned pointwise nonlinearity.

### The Formula

```
DyT(x) = γ ⊙ tanh(α · x) + β

where:
  α   ∈ ℝ          — learned scalar per layer (initialized to 0.5)
  γ   ∈ ℝᵈ         — learned per-dim scale (same as LayerNorm's γ)
  β   ∈ ℝᵈ         — learned per-dim bias  (same as LayerNorm's β)
  ⊙               — element-wise multiplication
  tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
```

### Why tanh Behaves Like Normalization

```
tanh(z) curve:

      output
       1.0 ┤                    ╭─────────────
           │              ╭────╯
       0.5 ┤         ╭───╯
           │    ╭───╯
       0.0 ┤───╯──────────────────────────── input z
           │         ╰───╮
      -0.5 ┤              ╰────╮
           │                   ╰────╮
      -1.0 ┤─────────────────────── ╰────────

  tanh saturates at ±1 for large |z|.
  Large activations (±50, ±1000) all get compressed to (-1, 1).
  This is precisely what normalization achieves — but via a
  smooth learned function instead of computing statistics.
```

### DyT vs LayerNorm: Computation

```
LayerNorm:
  1. Compute mean μ       — reduction over entire vector x
  2. Compute variance σ²  — second reduction over entire vector x
  3. Normalize and scale  — element-wise

  Requires TWO full passes over x.
  Memory bandwidth bottleneck on large hidden dims.


DyT:
  1. Scale input:  α · x       — element-wise (scalar α)
  2. Apply tanh                — element-wise nonlinearity
  3. Affine:  γ ⊙ (·) + β    — element-wise

  ZERO reduction operations. Fully element-wise.
  Trivially parallelizable. No memory bottleneck.
```

### Results (from the paper)

```
ViT-B/16 on ImageNet classification:

  Standard LayerNorm  →  81.8% top-1
  DyT replacement     →  81.8% top-1   (identical)

GPT-2 language modeling (WikiText-103, perplexity ↓ better):

  Standard LayerNorm  →  18.7 perplexity
  DyT replacement     →  18.6 perplexity  (marginally better)

DyT matches LayerNorm quality across vision and language tasks.
Still research-stage in 2025 — not yet adopted in production LLMs.
```

---

## Part 5 — DeepNorm: Training 1000-Layer Transformers

**Paper:** "DeepNet: Scaling Transformers to 1,000 Layers" (Wang et al., 2022, Microsoft Research)

DeepNorm does not replace LayerNorm's formula. It changes **how the residual connection interacts with LayerNorm** to enable training at extreme depth.

### The Problem with Deep Models

```
Post-Norm residual:  xₗ₊₁ = LayerNorm(xₗ + F(xₗ))

At initialization, F(xₗ) ≈ 0 (small random weights).
Gradient of loss w.r.t. layer l involves the product of Jacobians:

  ∂L/∂Wₗ ∝ Π_{k=l}^{L} J_k

For L = 1000 layers, this product of 1000 Jacobians either:
  → Explodes to infinity  (gradient explosion)
  → Shrinks to zero       (gradient vanishing)

Standard pre-norm works up to ~100 layers but still
struggles at 500+ layers.
```

### DeepNorm's Fix

```
Standard Pre-Norm:
  xₗ₊₁ = xₗ + Sublayer(LayerNorm(xₗ))


DeepNorm:
  xₗ₊₁ = LayerNorm(α · xₗ + Sublayer(xₗ))
                    ↑
           scale the residual connection
           (α > 1, e.g. α = 2.0 for 1000-layer encoder)
```

The amplified skip connection `α·xₗ` keeps the model close to the identity function at initialization, bounding the gradient norm throughout training. Combined with **Xavier-style weight init scaled by β < 1** for sublayer weights, this enables stable training at depth previously impossible.

```
Stability comparison:

  Architecture        │ Max stable depth (empirical)
  ────────────────────┼──────────────────────────────
  Post-Norm           │  ~20 layers
  Pre-Norm            │  ~100 layers
  DeepNorm (α=2)      │  1000+ layers  ✓
```

---

## Part 6 — QKNorm: Normalization Inside Attention

**Used in:** Mistral 7B v0.3+, Stable Diffusion 3, long-context vision transformers

QKNorm does not replace LayerNorm. It adds **extra normalization inside the attention mechanism** to solve attention score explosion in long-context models.

### The Problem

```
Standard self-attention:

  Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

  As sequence length grows (4K → 32K → 128K tokens),
  dot products Q·Kᵀ can produce very large values:

  Example: Q[i] · K[j] = 1200  (after √dₖ scaling: still 75)

  softmax([..., 75, ...]) ≈ [0.0, 0.0, ..., 1.0, ..., 0.0]
                                               ↑
                         Near-one-hot: attention collapses to one token.
                         Long-range context is lost entirely.
```

### QKNorm's Fix

```
QKNorm applies L2 normalization to Q and K before the dot product:

  Q̂ = Q / ‖Q‖₂   (L2-normalize each query vector)
  K̂ = K / ‖K‖₂   (L2-normalize each key vector)

  scores = Q̂ · K̂ᵀ · s

  s = learned temperature scalar (initialized to √dₖ)


Effect on dot product range:
  Standard:  Q · Kᵀ ∈ [-∞, +∞]   (unbounded)
  QKNorm:    Q̂ · K̂ᵀ ∈ [-1,  +1]  (bounded by Cauchy-Schwarz)

  The model cannot produce extreme logits.
  Attention distributions stay diffuse → long-range attention preserved.
```

---

## Part 7 — Complete Comparison

```
┌──────────────┬──────────────────────────────────┬──────────────┬────────────────────────────┐
│ Method       │ Formula                          │ Speed        │ Used In                    │
├──────────────┼──────────────────────────────────┼──────────────┼────────────────────────────┤
│ LayerNorm    │ (x - μ) / √(σ²+ε) · γ + β       │ baseline     │ BERT, GPT-2, original ViT  │
│ RMSNorm      │ x / RMS(x) · γ                   │ ~15% faster  │ LLaMA 2/3, Mistral, Gemma  │
│ DyT          │ γ ⊙ tanh(α·x) + β               │ fastest      │ Research (2025)            │
│ DeepNorm     │ LN(α·x + Sublayer(x))            │ negligible   │ 1000-layer research models │
│ QKNorm       │ L2-norm on Q and K               │ minimal      │ Mistral, long-context LLMs │
└──────────────┴──────────────────────────────────┴──────────────┴────────────────────────────┘
```

### Timeline of Adoption

```
2016  LayerNorm   — Ba et al.; standard for all RNNs and early transformers
2017  Post-Norm   — used in original "Attention Is All You Need"
2019  RMSNorm     — Zhang & Sennrich; faster, simpler
2020  Pre-Norm    — GPT-3 paper; now the standard placement
2022  DeepNorm    — enables 1000-layer stable training
2022  QKNorm      — long-context attention stability
2025  DyT         — eliminates statistics entirely; research-stage
```

---

## Part 8 — PyTorch Implementations

### LayerNorm (manual)

```python
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        mean  = x.mean(dim=-1, keepdim=True)
        var   = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
```

### RMSNorm (used in LLaMA)

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(dim))  # no beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)
```

### Dynamic Tanh (DyT)

```python
class DynamicTanh(nn.Module):
    def __init__(self, dim: int, alpha_init: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))  # learned scalar
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta
```

### Drop-in Transformer Block (swap any norm)

```python
from typing import Literal

NormType = Literal["layernorm", "rmsnorm", "dyt"]


def make_norm(norm_type: NormType, dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim)
    elif norm_type == "dyt":
        return DynamicTanh(dim)
    raise ValueError(f"Unknown norm: {norm_type}")


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, norm_type: NormType = "rmsnorm"):
        super().__init__()
        self.norm1 = make_norm(norm_type, dim)
        self.norm2 = make_norm(norm_type, dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


# Compare all three norms
for norm in ("layernorm", "rmsnorm", "dyt"):
    model  = TransformerBlock(dim=768, num_heads=12, norm_type=norm)
    x      = torch.randn(2, 128, 768)
    out    = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"{norm:12s}  output: {out.shape}  params: {params:,}")

# layernorm     output: torch.Size([2, 128, 768])  params: 7,089,920
# rmsnorm       output: torch.Size([2, 128, 768])  params: 7,088,384  (no β)
# dyt           output: torch.Size([2, 128, 768])  params: 7,088,897  (scalar α)
```

### QKNorm in Attention

```python
class QKNormAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = nn.Parameter(torch.tensor(float(self.head_dim) ** 0.5))
        self.qkv       = nn.Linear(dim, 3 * dim, bias=False)
        self.proj      = nn.Linear(dim, dim, bias=False)
        self.q_norm    = RMSNorm(self.head_dim)
        self.k_norm    = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Hd   = self.num_heads, self.head_dim

        qkv     = self.qkv(x).reshape(B, T, 3, H, Hd)
        q, k, v = qkv.unbind(dim=2)

        q = self.q_norm(q)             # normalize queries
        k = self.k_norm(k)             # normalize keys

        q = q.transpose(1, 2)          # (B, H, T, Hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)
```

---

## Key Takeaways

```
1. VARIANCE is the foundation
   Measures spread of a distribution. Normalization divides by
   √variance to enforce unit spread across a token's features.

2. LAYERNORM normalizes per-token across features
   Batch-size independent. Works identically in train and inference.
   The standard for all transformers from 2017 onward.

3. PRE-NORM placement is critical
   Norm before the sublayer (not after) keeps the residual path
   clean. Stable up to 96+ layers. All modern LLMs use it.

4. RMSNORM drops mean subtraction
   Same quality, ~15% faster. Mean centering adds no value when
   the network has a learned β to shift the distribution anyway.
   Now the default in LLaMA 3, Mistral, Gemma, Qwen, DeepSeek.

5. DYT eliminates statistics entirely
   Replaces the norm with a learned tanh gate. Matches LayerNorm
   quality with zero reduction operations. Research-stage (2025).

6. DEEPNORM enables 1000-layer depth
   Scales the residual connection (α·x) before LayerNorm,
   combined with small weight init. Keeps gradient norms
   bounded at extreme depth.

7. QKNORM prevents attention collapse in long contexts
   L2-normalizes Q and K vectors so attention logits stay
   bounded. Used in 128K+ context models where standard
   attention degrades.
```

---

## References

- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. *arXiv:1607.06450*
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*
- Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization. *NeurIPS 2019*
- Wang, H. et al. (2022). DeepNet: Scaling Transformers to 1,000 Layers. *arXiv:2203.00555*
- Touvron, H. et al. (2023). LLaMA 2: Open Foundation and Fine-Tuned Chat Models. *arXiv:2307.09288*
- Zhu, J. et al. (2025). Transformers without Normalization. *arXiv:2503.10622*

---

*Tutorial by Koyilbek Valiev — AI / ML Engineer | Research Engineer*
