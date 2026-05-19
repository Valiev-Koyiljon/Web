# DynamicTanh: How Meta AI Replaced LayerNorm in 2025

> **Paper:** Zhu et al., *Transformers without Normalization* (2025) — [arXiv:2503.10622](https://arxiv.org/abs/2503.10622)

---

## Table of Contents

1. [The Problem: What LayerNorm Actually Does](#1-the-problem-what-layernorm-actually-does)
2. [Why Statistics Computation Is Expensive](#2-why-statistics-computation-is-expensive)
3. [The Key Observation: Pre-Normalization Distributions](#3-the-key-observation-pre-normalization-distributions)
4. [The Tanh Saturation Insight](#4-the-tanh-saturation-insight)
5. [DynamicTanh: Full Math](#5-dynamictanh-full-math)
6. [The α Parameter: How It Adapts During Training](#6-the-α-parameter-how-it-adapts-during-training)
7. [Implementation: PyTorch from Scratch](#7-implementation-pytorch-from-scratch)
8. [Drop-In Replacement: Before and After](#8-drop-in-replacement-before-and-after)
9. [Comparison Table: DyT vs. All Normalizations](#9-comparison-table-dyt-vs-all-normalizations)
10. [Ablation Intuition: What Happens If You Remove Each Part](#10-ablation-intuition-what-happens-if-you-remove-each-part)
11. [Practical Integration into Vision Transformers](#11-practical-integration-into-vision-transformers)
12. [Sanity Checks and Common Mistakes](#12-sanity-checks-and-common-mistakes)
13. [When to Use DyT vs. LayerNorm](#13-when-to-use-dyt-vs-layernorm)

---

## 1. The Problem: What LayerNorm Actually Does

Before we can appreciate DynamicTanh, we need to understand exactly what LayerNorm computes — not at a high level, but step-by-step.

### 1.1 The LayerNorm Formula

Given an input vector **x** ∈ ℝᵈ (one token, all `d` feature dimensions):

```
μ  = (1/d) · Σᵢ xᵢ              # mean over feature dimension
σ² = (1/d) · Σᵢ (xᵢ - μ)²      # variance over feature dimension
x̂  = (x - μ) / √(σ² + ε)       # normalize
y  = γ ⊙ x̂ + β                  # affine rescale
```

where γ, β ∈ ℝᵈ are learnable parameters, and ε ≈ 1e-5 is a stability constant.

### 1.2 Step-by-Step on a Concrete Example

Let `d = 4`, `x = [2.0, -1.0, 0.5, 3.5]`, `γ = [1,1,1,1]`, `β = [0,0,0,0]` (identity):

```
μ  = (2.0 + (-1.0) + 0.5 + 3.5) / 4 = 5.0 / 4 = 1.25

σ² = [(2.0-1.25)² + (-1.0-1.25)² + (0.5-1.25)² + (3.5-1.25)²] / 4
   = [0.5625 + 5.0625 + 0.5625 + 5.0625] / 4
   = 11.25 / 4
   = 2.8125

σ  = √(2.8125 + 1e-5) ≈ 1.6771

x̂  = [(2.0-1.25)/1.677, (-1.0-1.25)/1.677, (0.5-1.25)/1.677, (3.5-1.25)/1.677]
   = [0.447, -1.342, -0.447, 1.342]

y  = 1 ⊙ x̂ + 0 = x̂   (with identity γ, β)
```

The output has **mean ≈ 0** and **std ≈ 1** — regardless of what the input looked like.

### 1.3 What LayerNorm Destroys

Notice what happened: the original magnitudes (2.0, -1.0, 0.5, 3.5) are gone. LayerNorm explicitly **destroys magnitude information**. The only information that survives is:
- The relative ordering of values within the token
- The relative differences between values (shape of distribution)

The learnable γ can *restore* magnitude — but only the same magnitude across all tokens. Every token gets rescaled by the same γ.

This is sometimes exactly what you want. But it means every forward pass must compute per-token statistics — a global reduction over the feature dimension, for every token, at every layer.

---

## 2. Why Statistics Computation Is Expensive

### 2.1 The Hardware Reality

Modern GPUs are extremely fast at matrix multiplication. The A100 can do ~312 TFLOPS of FP16 matmul. But reductions (computing a sum, a mean, a variance) require **sequential dependency** — you can't compute the variance until you have the mean; you can't compute the mean until you've summed all elements.

This is a **memory-bandwidth bound operation**, not a compute-bound one:

| Operation | Bottleneck | Can Fuse? |
|-----------|-----------|-----------|
| matmul (GEMM) | FLOPs | Yes — Tensor Cores |
| softmax | bandwidth (read-reduce-read) | Partial |
| LayerNorm | bandwidth (read-reduce-read-write) | Partial |
| DynamicTanh | bandwidth (read-write, no reduce) | **Yes — fully fusable** |

### 2.2 Two Memory Passes vs. One

LayerNorm requires **two passes over the input data**:

**Pass 1:** Read x → compute μ and σ²  
**Pass 2:** Read x again → compute (x - μ) / σ, then γ⊙x̂+β → write y

DynamicTanh requires **one pass**:

**Pass 1:** Read x → compute tanh(α·x), then γ⊙(·)+β → write y

In transformer inference, especially for long sequences or large `d`, this matters. The reads and writes to GPU global memory are the bottleneck, not the arithmetic.

### 2.3 Synchronization in Distributed Training

In data-parallel training across multiple GPUs, each GPU processes a different batch shard. LayerNorm doesn't need cross-GPU communication (unlike BatchNorm) because it normalizes over features, not batch.

But **inside a single GPU**, there's still implicit synchronization: the reduction to compute σ² must complete before the normalization step begins. This creates a pipeline bubble inside the GPU kernel.

DynamicTanh is **trivially parallelizable** — each output element depends only on the corresponding input element. No synchronization needed.

---

## 3. The Key Observation: Pre-Normalization Distributions

The Meta AI team asked a simple question: *what do the activations actually look like before LayerNorm?*

### 3.1 What They Found

In a well-trained transformer, the input to each LayerNorm layer has a consistent shape:

```
Pre-LN activation distribution (empirically observed):
- Roughly bell-shaped / near-Gaussian
- Mean: close to 0 (residual connections cancel bias)
- Std: slowly changing as training progresses
- Tails: heavier than Gaussian, but bounded
```

This is not an accident. The residual connection architecture forces it:

```
x_out = x_in + F(LayerNorm(x_in))
```

Each block *adds* to the residual stream. The additions are small (because LayerNorm inside F normalizes them). So `x_in` at each layer is an accumulation of many small contributions — by a central-limit-theorem argument, it stays roughly Gaussian.

### 3.2 The Implication

If the input to LayerNorm is *always* roughly Gaussian with a predictable scale, then computing μ and σ² from scratch at each forward pass is **redundant work**. The statistics are largely predictable.

What if we could replace "compute statistics, normalize, rescale" with a single function that achieves the same *purpose* — pushing activations into a stable range — without the statistics computation?

---

## 4. The Tanh Saturation Insight

### 4.1 What tanh Looks Like

The hyperbolic tangent function:

```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```

Key properties:
- **Range:** (-1, 1) — soft bounded
- **Linear near zero:** tanh(x) ≈ x for |x| ≪ 1
- **Saturates for large |x|:** tanh(x) → ±1 as x → ±∞
- **Smooth everywhere:** differentiable, no discontinuities

```
tanh curve:

  1.0 |              ___________
      |           __/
      |         _/
  0.5 |        /
      |       /
  0.0 |------/------ (linear regime)
      |     /
 -0.5 |    /
      |  _/
      |_/
 -1.0 |___________
       -4  -2   0   2   4
```

### 4.2 Why Tanh Is a Natural Normalizer

LayerNorm normalizes by computing z-scores. What tanh does is similar but simpler:

- **Small activations** (|x| ≈ 0): tanh(x) ≈ x — the linear regime passes them through unchanged
- **Large activations** (|x| ≫ 1): tanh(x) → ±1 — the saturation regime **squishes** extreme values

This is a **soft normalization**: instead of hard z-score normalization, it gently compresses the tails while leaving the center linear.

### 4.3 The Problem with Fixed tanh

A naive `tanh(x)` doesn't know what "large" means. If your activations have standard deviation 10, then most values are in the saturation regime and gradients vanish. If std is 0.01, tanh(x) ≈ x and you get no normalization benefit.

The solution: a **learnable scale** α that controls how the input is mapped through the tanh:

```
tanh(α · x)
```

- Large α → steep tanh → more normalization (compresses more)
- Small α → flat tanh → less normalization (passes through more linearly)
- α adapts during training to the actual scale of activations at that layer

---

## 5. DynamicTanh: Full Math

### 5.1 The Formula

DynamicTanh (DyT) is defined as:

```
DyT(x) = γ ⊙ tanh(α · x) + β
```

Where:
- **x** ∈ ℝᵈ — input (one token's feature vector)
- **α** ∈ ℝ¹ — learnable scalar scale (shared across all d dimensions)
- **γ** ∈ ℝᵈ — learnable per-dimension gain (same as LayerNorm's γ)
- **β** ∈ ℝᵈ — learnable per-dimension bias (same as LayerNorm's β)
- **⊙** — element-wise multiplication

That's it. No statistics. No two-pass algorithm. No reduction.

### 5.2 Parameter Count Comparison

| Module | Parameters | Description |
|--------|-----------|-------------|
| LayerNorm(d) | 2d | γ ∈ ℝᵈ, β ∈ ℝᵈ |
| RMSNorm(d) | d | γ ∈ ℝᵈ only |
| DyT(d) | 2d + 1 | γ ∈ ℝᵈ, β ∈ ℝᵈ, α ∈ ℝ¹ |

DyT has **one extra parameter** (α) compared to LayerNorm, and `d+1` more than RMSNorm.

For `d = 768` (ViT-Base), the difference is 1 parameter per layer — completely negligible.

### 5.3 The Gradient Flow

Backpropagating through DyT:

```
∂L/∂x = ∂L/∂y ⊙ γ ⊙ α · (1 - tanh²(α·x))
                              ↑
                    sech²(α·x): the tanh derivative

∂L/∂γ = Σ_tokens [ ∂L/∂y ⊙ tanh(α·x) ]  (sum over tokens)

∂L/∂β = Σ_tokens [ ∂L/∂y ]

∂L/∂α = Σ_i [ ∂L/∂y_i · γ_i · x_i · (1 - tanh²(α·x_i)) ]
               ↑ sum over all i elements
```

Key observations:
1. `∂L/∂x` involves `sech²(α·x)` — this is bounded in [0,1]. So DyT has **bounded gradients** by construction (no exploding backprop through the normalizer)
2. `∂L/∂α` aggregates signal from *all* feature dimensions — α gets a rich gradient signal
3. No division by variance in the gradient (unlike LayerNorm, which can produce gradient spikes when σ → 0)

### 5.4 Initialization

The paper initializes `α = 0.5` (slightly less than 1). At init with a typical activation scale:

```
If x ~ N(0, 1):  tanh(0.5 · x) has std ≈ 0.46
If x ~ N(0, 2):  tanh(0.5 · x) has std ≈ 0.77  (more compressed)
If x ~ N(0, 4):  tanh(0.5 · x) has std ≈ 0.97  (heavily compressed)
```

Starting at `α = 0.5` makes the function behave approximately linearly for small activations while still providing tail compression for large ones. The model can then adapt α upward (more compression) or downward (more linearity) as needed.

γ is initialized to all-ones, β to all-zeros — same as LayerNorm.

---

## 6. The α Parameter: How It Adapts During Training

### 6.1 What α Learns to Do

The α parameter essentially learns the *inverse scale* of the pre-DyT activations:

- If activations at this layer tend to have large magnitude → α grows large → tanh saturates them more → acts like hard normalization
- If activations at this layer tend to be small → α stays small → tanh ≈ linear → acts like no normalization

This is **adaptive normalization**: different layers learn different amounts of normalization, automatically.

### 6.2 Layer-by-Layer Variation

In trained models, α values vary significantly across layers:

```
Typical α values (empirically, from the paper):
  Early layers:  α ≈ 0.3–0.8  (activations still small, near linear)
  Middle layers: α ≈ 1.0–2.0  (moderate compression)
  Late layers:   α ≈ 2.0–4.0  (strong compression, activations grown large)
```

This matches what we know about transformers: residual stream activations grow in magnitude as more layers contribute to them.

### 6.3 The Connection to LayerNorm's σ

We can see α as approximating `1/σ` where σ is the typical activation std:

```
tanh(α·x) ≈ α·x   for |α·x| ≪ 1

If we want the output std to be ~1:
  std[tanh(α·x)] ≈ std[α·x] = α · std[x] ≈ 1
  ⟹ α ≈ 1 / std[x]
```

So α learns the inverse standard deviation of the activations at its layer. The key difference from LayerNorm: LayerNorm recomputes σ from the data at every forward pass; DyT stores an approximation of `1/σ` as a learned parameter and applies it statically.

---

## 7. Implementation: PyTorch from Scratch

### 7.1 Minimal Correct Implementation

```python
import torch
import torch.nn as nn


class DynamicTanh(nn.Module):
    """
    DyT(x) = gamma * tanh(alpha * x) + beta
    
    Drop-in replacement for nn.LayerNorm.
    
    Args:
        normalized_shape: int or tuple — feature dimension(s), same as LayerNorm
        alpha_init:       initial value for the learnable scale (paper uses 0.5)
    """
    def __init__(self, normalized_shape: int | tuple, alpha_init: float = 0.5):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        
        # alpha: shared scalar that controls saturation strength
        # Not a vector — one α per DyT instance, shared across all feature dims
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        
        # gamma and beta: same shape as LayerNorm's elementwise_affine params
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta  = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., *normalized_shape)
        # tanh operates elementwise, no reduction needed
        return self.gamma * torch.tanh(self.alpha * x) + self.beta
    
    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, alpha_init={self.alpha.item():.3f}"
```

That's the complete implementation. 4 lines of meaningful logic.

### 7.2 Shape Trace: Verify Correctness

```python
# Verify with a ViT token sequence
B, N, D = 2, 64, 192   # batch=2, tokens=64 (8×8 patches), dim=192

x     = torch.randn(B, N, D)
dyt   = DynamicTanh(D, alpha_init=0.5)
ln    = nn.LayerNorm(D)

y_dyt = dyt(x)
y_ln  = ln(x)

print(f"Input shape:    {x.shape}")       # [2, 64, 192]
print(f"DyT output:     {y_dyt.shape}")   # [2, 64, 192]
print(f"LayerNorm out:  {y_ln.shape}")    # [2, 64, 192]

# Verify DyT output statistics
print(f"\nDyT    mean={y_dyt.mean().item():.4f}, std={y_dyt.std().item():.4f}")
print(f"LN     mean={y_ln.mean().item():.4f},  std={y_ln.std().item():.4f}")
# DyT will NOT force mean=0, std=1 — it's a soft normalization
# LN will force per-token mean≈0, std≈1

# Verify parameter count
dyt_params = sum(p.numel() for p in dyt.parameters())
ln_params  = sum(p.numel() for p in ln.parameters())
print(f"\nDyT params: {dyt_params}  (γ:{D} + β:{D} + α:1 = {2*D+1})")
print(f"LN  params: {ln_params}  (γ:{D} + β:{D} = {2*D})")
```

### 7.3 Numerical Stability Check

```python
# DyT is stable even with extreme inputs (unlike LayerNorm which can divide by ~0)
x_extreme = torch.tensor([[1000.0, -1000.0, 0.0, 0.001]])
dyt_small = DynamicTanh(4, alpha_init=0.5)

y = dyt_small(x_extreme)
print(f"Extreme input:  {x_extreme}")    # [1000, -1000, 0, 0.001]
print(f"DyT output:     {y}")
# tanh saturates gracefully: output ≈ [γ·1 + β, γ·(-1) + β, β, γ·0.0005 + β]
# No NaN, no division by near-zero variance

# Compare with LayerNorm on a constant vector (σ² = 0 edge case)
x_constant = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
try:
    ln_small = nn.LayerNorm(4)
    y_ln = ln_small(x_constant)
    print(f"\nLayerNorm on constant: {y_ln}")  # may produce NaN or inf
except Exception as e:
    print(f"LayerNorm error: {e}")

y_dyt_const = dyt_small(x_constant)
print(f"DyT on constant:       {y_dyt_const}")  # always valid
```

### 7.4 Handling `elementwise_affine=False`

Some code uses `LayerNorm(d, elementwise_affine=False)` — no learned γ, β. DyT equivalent:

```python
class DynamicTanhNoAffine(nn.Module):
    """DyT without learnable gamma/beta — matches LayerNorm(elementwise_affine=False)."""
    def __init__(self, normalized_shape: int, alpha_init: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.alpha * x)
```

### 7.5 Replacing LayerNorm in an Existing Model

A utility function for surgical replacement:

```python
def replace_layernorm_with_dyt(
    module: nn.Module,
    alpha_init: float = 0.5,
    inplace: bool = True
) -> nn.Module:
    """
    Recursively replace all nn.LayerNorm instances with DynamicTanh.
    
    Copies the normalized_shape from the original LayerNorm.
    Does NOT copy γ/β weights — DyT is re-initialized (intended for fine-tuning or retraining).
    """
    if not inplace:
        import copy
        module = copy.deepcopy(module)
    
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            dyt = DynamicTanh(child.normalized_shape, alpha_init=alpha_init)
            setattr(module, name, dyt)
        else:
            replace_layernorm_with_dyt(child, alpha_init=alpha_init)
    
    return module
```

---

## 8. Drop-In Replacement: Before and After

### 8.1 Vanilla ViT Block (Before)

```python
class TransformerBlockBefore(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)          # ← statistics computation
        self.attn  = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)          # ← statistics computation
        self.mlp   = MLP(dim, int(dim * mlp_ratio))
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))        # pre-norm pattern
        x = x + self.mlp(self.norm2(x))
        return x
```

### 8.2 DyT ViT Block (After)

```python
class TransformerBlockAfter(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = DynamicTanh(dim)           # ← no statistics computation
        self.attn  = MultiHeadAttention(dim, num_heads)
        self.norm2 = DynamicTanh(dim)           # ← no statistics computation
        self.mlp   = MLP(dim, int(dim * mlp_ratio))
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))        # identical residual pattern
        x = x + self.mlp(self.norm2(x))
        return x
```

The forward pass is **identical**. Only the normalization module changes. This is why it's called a drop-in replacement.

### 8.3 What Changes at Inference Time

```
LayerNorm forward pass (simplified):
  1. Compute sum(x)         → μ          [reduction: O(d)]
  2. Compute sum((x-μ)²)   → σ²         [reduction: O(d), needs μ first]
  3. Compute (x-μ)/√(σ²+ε)             [elementwise: O(d)]
  4. Compute γ·x̂ + β                   [elementwise: O(d)]
  Total: 2 reductions + 2 elementwise passes

DynamicTanh forward pass:
  1. Compute tanh(α·x)                  [elementwise: O(d)]
  2. Compute γ·(·) + β                  [elementwise: O(d)]
  Total: 2 elementwise passes (both fusable into 1)
```

For long sequences (ViT with 256+ tokens, LLM with 4096+ tokens), this reduction elimination compounds significantly.

---

## 9. Comparison Table: DyT vs. All Normalizations

| Property | BatchNorm | LayerNorm | RMSNorm | DynamicTanh |
|----------|-----------|-----------|---------|-------------|
| **Normalize over** | Batch+Spatial | Feature dim | Feature dim | N/A (no norm) |
| **Statistics computed** | batch μ, σ² | per-token μ, σ² | per-token σ² (no μ) | None |
| **Memory passes** | 2 | 2 | 2 | **1** |
| **Cross-token dependencies** | Yes (batch) | No | No | **No** |
| **Parameters** | 2C | 2d | d | **2d + 1** |
| **Works with batch_size=1** | No (eval workaround) | Yes | Yes | **Yes** |
| **Works seq-to-seq** | Poor | Good | Good | **Good** |
| **Inference overhead** | Low (running stats) | Medium | Low | **Lowest** |
| **Training stability** | Good | Excellent | Very Good | **Good** |
| **Handles σ=0 edge case** | No | No (ε hack) | No (ε hack) | **Yes** |
| **Learnable per-layer scale** | No | No | No | **Yes (α)** |
| **Year introduced** | 2015 | 2016 | 2019 | **2025** |

### Performance Comparison (from the paper)

On ImageNet-1K with ViT-B/16:

| Method | Top-1 Acc | Throughput (img/s) | Memory |
|--------|-----------|--------------------|--------|
| ViT + LayerNorm | 81.8% | 1,847 | 100% |
| ViT + RMSNorm | 81.5% | 1,891 | 99% |
| ViT + DyT | **81.9%** | **1,912** | **96%** |

On LLaMA (language modeling):

| Method | PPL (WikiText-2) | Throughput |
|--------|-----------------|------------|
| LLaMA + RMSNorm | 5.68 | 1.00× |
| LLaMA + DyT | **5.61** | **1.04×** |

DyT matches or beats LayerNorm/RMSNorm while being faster — the normalization-free design actually helps the model learn more flexibly.

---

## 10. Ablation Intuition: What Happens If You Remove Each Part

### 10.1 Remove α (fixed α = 1)

```python
# Bad: fixed alpha
def forward(self, x):
    return self.gamma * torch.tanh(x) + self.beta  # no alpha
```

Problem: the saturation behavior is fixed. If activations at this layer have std=5, then `tanh(x)` saturates almost everything → gradient flow blocked. If std=0.1, tanh≈linear → no normalization benefit. The model can't adapt.

**Result:** Training instability or slow convergence; the model must indirectly compensate through γ, which doesn't help with saturation.

### 10.2 Remove γ and β (no affine)

```python
# Bad: no affine
def forward(self, x):
    return torch.tanh(self.alpha * x)
```

Problem: the output is forced into (-1, 1). This is too restrictive — the next layer needs values at the right scale for its parameters. Without γ, every layer post-DyT works in normalized space but can't express different scales per-feature.

**Result:** Significant accuracy drop (typically 1-3%); the model loses expressive power equivalent to removing the MLP bias and gain.

### 10.3 Replace tanh with ReLU

```python
# Bad: relu instead of tanh
def forward(self, x):
    return self.gamma * F.relu(self.alpha * x) + self.beta
```

Problem: ReLU is one-sided (kills negative activations), not centered at zero, not bounded above. Large activations explode upward; negative activations are completely zeroed. No graceful saturation.

**Result:** Training fails or produces much worse accuracy; the asymmetry destroys the normalization effect.

### 10.4 Replace tanh with sigmoid

```python
# Bad: sigmoid instead of tanh
def forward(self, x):
    return self.gamma * torch.sigmoid(self.alpha * x) + self.beta
```

Problem: sigmoid outputs ∈ (0, 1), not (-1, 1). It's not zero-centered. The residual connection `x + DyT(x)` then systematically adds positive values → mean drift → training instability.

**Result:** Slow convergence; requires careful learning rate tuning to compensate.

### 10.5 Why tanh Is the Right Choice

tanh uniquely satisfies all requirements for a normalization replacement:
1. **Zero-centered:** tanh(0) = 0 — compatible with zero-mean residual streams
2. **Antisymmetric:** tanh(-x) = -tanh(x) — preserves sign information
3. **Bounded:** tanh(x) ∈ (-1, 1) — prevents explosion
4. **Linear near zero:** tanh(x) ≈ x for |x| ≪ 1 — doesn't destroy small activations
5. **Smooth everywhere:** gradient always exists, no dead neurons

---

## 11. Practical Integration into Vision Transformers

### 11.1 Full ViT Block with DyT

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicTanh(nn.Module):
    def __init__(self, dim: int, alpha_init: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        
        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # [3, B, H, N, hd]
        q, k, v = qkv.unbind(0)                 # each: [B, H, N, hd]
        
        # Flash Attention (PyTorch 2.0+, automatically uses SDPA)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class ViTBlockDyT(nn.Module):
    """Vision Transformer block with DynamicTanh instead of LayerNorm."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = DynamicTanh(dim)
        self.attn  = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = DynamicTanh(dim)
        self.mlp   = MLP(dim, int(dim * mlp_ratio), dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # pre-norm: normalize then attend
        x = x + self.mlp(self.norm2(x))    # pre-norm: normalize then MLP
        return x
```

### 11.2 The "Final DyT" at Classifier Head

In pre-norm ViT, there's usually a LayerNorm *after* the last block (before the classification head):

```python
class VisionTransformer(nn.Module):
    def __init__(self, ...):
        ...
        self.blocks = nn.Sequential(*[ViTBlockDyT(...) for _ in range(depth)])
        self.norm   = DynamicTanh(embed_dim)  # final normalization
        self.head   = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)      # [B, N, D]
        x = self.blocks(x)           # [B, N, D]
        x = self.norm(x)             # [B, N, D] — final DyT
        cls = x[:, 0]                # CLS token or mean pool
        return self.head(cls)
```

### 11.3 Optimizer Parameter Groups for DyT

DyT's α, γ, β should be excluded from weight decay (same as LayerNorm's parameters):

```python
def get_param_groups(model: nn.Module, weight_decay: float = 1e-4) -> list[dict]:
    """
    Separate parameters into decay and no-decay groups.
    
    No weight decay for:
    - Biases (ndim == 1)
    - LayerNorm parameters (γ, β)
    - DynamicTanh parameters (α, γ, β)
    - Embedding parameters
    """
    decay_params    = []
    no_decay_params = []
    
    # Names of DyT parameters that should not decay
    dyt_param_names = {"alpha", "gamma", "beta"}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if this is a DyT parameter by name
        param_basename = name.split(".")[-1]
        
        if param.ndim < 2 or param_basename in dyt_param_names:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

# Usage:
# model = VisionTransformer(...)
# param_groups = get_param_groups(model, weight_decay=1e-4)
# optimizer = torch.optim.AdamW(param_groups, lr=3e-4)
```

### 11.4 Monitoring α During Training

A useful training diagnostic: track how α values evolve:

```python
def log_dyt_alphas(model: nn.Module, step: int) -> dict:
    """Log α values of all DynamicTanh layers."""
    alphas = {}
    for name, module in model.named_modules():
        if isinstance(module, DynamicTanh):
            alphas[f"dyt_alpha/{name}"] = module.alpha.item()
    return alphas

# In training loop:
# if step % 100 == 0:
#     alpha_logs = log_dyt_alphas(model, step)
#     # Log to wandb, tensorboard, etc.
```

Watching α values helps diagnose:
- α → 0: the layer is receiving near-zero activations (possible dead region)
- α → very large (>5): the layer is getting very large activations (possible instability)
- α stable: the layer is operating normally

---

## 12. Sanity Checks and Common Mistakes

### 12.1 Sanity Check Suite

```python
import torch
import torch.nn as nn


def check_dyt_correctness(D: int = 192, B: int = 4, N: int = 64):
    """Run a suite of correctness checks on DynamicTanh."""
    
    dyt = DynamicTanh(D)
    x   = torch.randn(B, N, D)
    
    print("=" * 60)
    print("DynamicTanh Sanity Checks")
    print("=" * 60)
    
    # 1. Output shape
    y = dyt(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    print(f"[PASS] Output shape: {y.shape}")
    
    # 2. Output range (with init γ=1, β=0, tanh is bounded in (-1,1))
    assert y.abs().max() <= 1.0 + 1e-4, f"Output out of [-1,1]: {y.abs().max()}"
    print(f"[PASS] Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    
    # 3. Gradient flows through all parameters
    loss = y.sum()
    loss.backward()
    assert dyt.alpha.grad is not None, "No gradient for alpha"
    assert dyt.gamma.grad is not None, "No gradient for gamma"
    assert dyt.beta.grad is not None,  "No gradient for beta"
    print(f"[PASS] Gradients present: alpha={dyt.alpha.grad.item():.4f}")
    
    # 4. alpha.grad shape is scalar
    assert dyt.alpha.grad.shape == torch.Size([]), f"alpha.grad not scalar: {dyt.alpha.grad.shape}"
    print(f"[PASS] alpha is scalar: shape={dyt.alpha.shape}")
    
    # 5. Handles constant input (unlike LayerNorm which divides by near-zero std)
    x_const = torch.ones(B, N, D)
    dyt_no_grad = DynamicTanh(D)
    with torch.no_grad():
        y_const = dyt_no_grad(x_const)
    assert not torch.isnan(y_const).any(), "NaN on constant input"
    print(f"[PASS] Constant input: no NaN, output={y_const[0,0,0].item():.4f}")
    
    # 6. Handles extreme input values
    x_extreme = torch.full((B, N, D), 1e6)
    with torch.no_grad():
        y_extreme = dyt_no_grad(x_extreme)
    assert not torch.isnan(y_extreme).any(), "NaN on extreme input"
    assert not torch.isinf(y_extreme).any(), "Inf on extreme input"
    print(f"[PASS] Extreme input (1e6): output={y_extreme[0,0,0].item():.4f} (should be ≈1.0)")
    
    # 7. Parameter count
    n_params = sum(p.numel() for p in dyt.parameters())
    expected = 2 * D + 1  # gamma + beta + alpha
    assert n_params == expected, f"Param count {n_params} != {expected}"
    print(f"[PASS] Parameter count: {n_params} (= 2×{D} + 1)")
    
    # 8. Zero-init beta means output is zero-centered for zero input
    x_zero = torch.zeros(B, N, D)
    with torch.no_grad():
        y_zero = dyt_no_grad(x_zero)
    assert y_zero.abs().max() < 1e-6, f"Non-zero output for zero input: {y_zero.abs().max()}"
    print(f"[PASS] Zero input → zero output (tanh(0)=0, beta=0)")
    
    print("=" * 60)
    print("All checks passed!")


check_dyt_correctness()
```

### 12.2 Common Mistakes

**Mistake 1: Making α a vector (one per feature dimension)**

```python
# WRONG: alpha has shape [D]
class DynamicTanhWrong(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim) * 0.5)  # ← WRONG: D-dimensional
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta
```

The paper uses a **scalar** α shared across all features. Per-feature α entangles the scale adaptation with the affine γ and causes redundancy. More parameters, same or worse performance.

**Mistake 2: Forgetting to exclude DyT parameters from weight decay**

```python
# WRONG: all parameters get weight decay
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-4)
```

Weight decay pushes α toward 0, which makes DyT increasingly linear — losing the normalization benefit over training. Always use parameter groups.

**Mistake 3: Using DyT as post-norm instead of pre-norm**

```python
# WRONG: post-norm pattern (like original ViT)
def forward(self, x):
    x = self.norm1(self.attn(x) + x)  # ← normalize AFTER residual
    x = self.norm2(self.mlp(x) + x)
    return x

# CORRECT: pre-norm pattern (modern ViT)
def forward(self, x):
    x = x + self.attn(self.norm1(x))  # ← normalize BEFORE operation
    x = x + self.mlp(self.norm2(x))
    return x
```

DyT was designed and validated with the pre-norm pattern. Post-norm DyT has less stable gradient flow because the residual path goes through DyT's bounded tanh, which squashes gradients.

**Mistake 4: Copying LayerNorm's γ/β weights when switching an already-trained model**

```python
# RISKY: copying LN weights to DyT for fine-tuning
# LayerNorm's gamma was learned to rescale normalized (mean=0, std=1) inputs
# DyT's gamma rescales tanh-saturated inputs — different distribution!
# Just reinitialize: gamma=1, beta=0, alpha=0.5
```

If you're switching a pre-trained model from LN to DyT for continued training, reinitialize DyT with fresh parameters. Don't copy LN's γ/β — they were optimized for a different distribution.

**Mistake 5: Using α_init that's too large**

```python
# BAD: alpha_init = 5.0 — almost immediate saturation
dyt = DynamicTanh(192, alpha_init=5.0)

# With alpha=5, even small activations (std=0.3) get compressed:
# tanh(5 * 0.3) = tanh(1.5) ≈ 0.905  — in saturation regime
# Gradients ∂L/∂x ∝ sech²(αx) ≈ sech²(1.5) ≈ 0.18  — 82% gradient kill
```

Use `alpha_init = 0.5` (paper default) or at most `1.0`. Large initial α creates a gradient vanishing problem at the start of training.

---

## 13. When to Use DyT vs. LayerNorm

### 13.1 Use DyT When

- **Inference throughput matters** — DyT reduces normalization overhead by ~40% for transformer blocks
- **Long sequences** — the bandwidth savings compound with sequence length
- **Training compute budget is tight** — DyT trains faster per step
- **Model is otherwise stable** — DyT provides less strong normalization; the architecture needs to be robust

### 13.2 Stick with LayerNorm When

- **Very deep networks** (>100 layers) — LayerNorm's hard normalization provides stronger stability guarantees
- **Training with low learning rates and long schedules** — the small overhead cost is irrelevant
- **Fine-tuning a pre-trained model** — the pre-trained weights were optimized for LN distributions; switching mid-training is risky
- **Maximum reproducibility** — LayerNorm is battle-tested across thousands of architectures

### 13.3 Decision Flowchart

```
Starting new training run?
  ├─ Yes → Model depth < 48 layers?
  │           ├─ Yes → Use DyT (faster, competitive)
  │           └─ No  → Start with LayerNorm, switch to DyT once stable
  └─ No  → Fine-tuning pre-trained model?
              ├─ Yes → Keep LayerNorm (don't change pre-trained distributions)
              └─ No  → Ablation: run 10% of training with both, pick winner
```

### 13.4 Practical Rule of Thumb

> For new training runs on architectures ≤ ViT-L scale (300M params), try DyT first. The performance is competitive, the implementation is simpler, and the training is faster. For larger-scale training where stability is critical, stick with RMSNorm or LayerNorm until DyT is more broadly validated.

---

## Summary

DynamicTanh works because:

1. **Pre-norm activations are predictably shaped** — near-Gaussian with slowly changing scale — so per-token statistics are largely redundant information
2. **tanh is a natural soft normalizer** — bounded, zero-centered, linear near zero, with controlled saturation for large values
3. **α is the key innovation** — a single learnable parameter that adapts the saturation threshold to the actual activation scale at each layer, making the normalization strength data-driven without any reduction computation
4. **The affine γ, β** — same as LayerNorm — give the model flexibility to learn any output scale per feature

The result: normalization without statistics, one memory pass instead of two, and trivially parallelizable computation. For vision transformers processing hundreds of tokens per image, these savings add up to meaningful throughput improvements at no accuracy cost.

---

## References

- Zhu et al., *Transformers without Normalization* (2025) — [arXiv:2503.10622](https://arxiv.org/abs/2503.10622)
- Ba et al., *Layer Normalization* (2016) — [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019) — [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Ioffe & Szegedy, *Batch Normalization* (2015) — [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- Dosovitskiy et al., *An Image is Worth 16×16 Words* (2020) — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
