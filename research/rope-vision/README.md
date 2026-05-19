# 2D Rotary Position Embeddings for Vision: Complete Math and Implementation

**Tutorial by:** Koyilbek Valiev  
**Topics:** Position Embeddings · Rotary Encodings · Vision Transformers · Linear Algebra  
**Paper:** [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)

---

## Why Position Encoding Exists

Transformers process tokens as a **set**, not a sequence. Internally, attention computes:

```
score(i, j) = qᵢ · kⱼᵀ
```

This dot product is identical regardless of whether token `i` is at position 3 or position 300. Shuffle the entire sequence and you get the same output. The model has no idea what order anything is in.

For text, this is disastrous — "dog bites man" ≠ "man bites dog".  
For images, it's equally bad — patch position defines spatial structure.

Position encoding injects position information so the model can distinguish "patch at (0,0)" from "patch at (7,7)".

The question is: **what is the right way to encode position?**

---

## Part 1 — What We Had Before: Learned Positional Embeddings

ViT's original approach: a learnable matrix `E ∈ ℝ^(N+1, D)`.

```python
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
```

You add `E[i]` to the `i`-th token before the first transformer block:

```
Token embedding at position 5:   [0.3, -0.1, 0.8, ...]
+ Positional embedding for 5:  + [0.1,  0.4, -0.2, ...]
= Input to transformer:        = [0.4,  0.3, 0.6, ...]
```

### What's wrong with this?

**Problem 1 — It encodes absolute position, not relative position.**

Attention is fundamentally a function of how tokens **relate to each other**. Whether patch A is close to patch B or far from patch B matters. Whether A is at absolute index 5 or 50 usually does not.

The learned embedding tells the model "you are at position 5." What the model actually needs is "you are 3 positions to the right of this other token."

```
What learned PE gives you:
  Token at position 5:  "I am at position 5"
  Token at position 8:  "I am at position 8"
  
What attention needs:
  score(5, 8) should reflect that they are 3 positions apart
  score(5, 100) should reflect that they are 95 positions apart
  
Learned PE encodes absolute position — relative distance
must be inferred indirectly by the model.
```

**Problem 2 — It wastes parameters.**

For ViT-Tiny (N=65, D=192): 65 × 192 = **12,480 parameters** that must be learned from scratch every training run. RoPE achieves better position encoding with zero parameters.

**Problem 3 — Fixed to training resolution.**

Trained on 32×32 images (64 patches)? The embedding matrix has shape (65, 192). It cannot be used on 64×64 images (256 patches) without retraining. The embedding matrix simply does not have entries for positions 65–257.

---

## Part 2 — The Mathematical Foundation: Complex Rotations

RoPE's foundation is a fact from complex number theory.

### Complex Numbers as 2D Rotations

A complex number `z = a + bi` can be represented as a 2D vector `[a, b]`. Multiplying two complex numbers corresponds to **rotating one by the angle of the other**:

```
z₁ = r₁ · e^(iθ₁)   (magnitude r₁, angle θ₁)
z₂ = r₂ · e^(iθ₂)   (magnitude r₂, angle θ₂)

z₁ · z₂ = r₁r₂ · e^(i(θ₁+θ₂))   ← angles ADD
```

In matrix form, rotating a 2D vector `[x, y]` by angle `θ`:

```
[x'] = [cos θ   -sin θ] [x]
[y']   [sin θ    cos θ] [y]
```

### The Key Property

The dot product of two rotated vectors depends only on **the difference of their angles**:

```
If  u = R(θ₁) · v₀   and   w = R(θ₂) · v₀

Then:  u · w = v₀ᵀ · R(θ₁)ᵀ · R(θ₂) · v₀
             = v₀ᵀ · R(θ₂ - θ₁) · v₀

The result depends only on (θ₂ - θ₁), not on θ₁ or θ₂ individually.
```

This is the insight RoPE exploits: if we set `θ = m · θ_base` where `m` is position, then:

```
score(position m, position n) = q_m · k_n
                               depends on (n·θ - m·θ) = (n-m)·θ
                               = relative position (n - m) only
```

The model can only perceive relative distances. It literally **cannot** encode absolute position.

---

## Part 3 — RoPE in 1D (Language Modeling)

### Setup

For a sequence of N tokens, each token at position `m` gets its query and key vectors rotated by an angle proportional to `m`.

But a D-dimensional vector has D dimensions — how do you rotate it?

**Answer:** pair up the dimensions and apply independent rotations to each pair.

For head dimension D, we have D/2 pairs: `(0,1), (2,3), (4,5), ..., (D-2, D-1)`.

Each pair gets its own frequency `θᵢ`:

```
Pair (2i, 2i+1) gets rotated by angle:  m · θᵢ

where θᵢ = 1 / 10000^(2i/D)
```

### Frequency Intuition

```
θᵢ = 1 / 10000^(2i/D)

i=0  (first pair):   θ₀ = 1/10000⁰ = 1.0        ← rotates FAST (period = 2π steps)
i=D/4:              θ = 1/10000^0.5 ≈ 0.01       ← medium rotation
i=D/2-1 (last pair): θ ≈ 1/10000¹ = 0.0001      ← rotates VERY SLOW (period ≈ 62,832 steps)

┌─────────────────────────────────────────────────────────────────────┐
│                    Frequency Spectrum                               │
│                                                                     │
│  Fast (θ ≈ 1.0)                         Slow (θ ≈ 0.0001)         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ dim 0  dim 2  dim 4  dim 6  ....  dim D-4  dim D-2         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Fast dims: complete a full rotation in a few positions            │
│  → sensitive to NEARBY tokens, encode SHORT-RANGE structure        │
│                                                                     │
│  Slow dims: take thousands of positions for a full rotation        │
│  → sensitive to DISTANT tokens, encode LONG-RANGE structure        │
└─────────────────────────────────────────────────────────────────────┘
```

The model can use fast dimensions to tell if two tokens are adjacent, and slow dimensions to tell if two tokens are in the same paragraph. All learned automatically.

### The Rotation Matrix

For a full D-dimensional vector at position `m`:

```
q̃ = R_m · q

where R_m is a block-diagonal rotation matrix:

        ┌ cos(m·θ₀)  -sin(m·θ₀)                              ┐
        │ sin(m·θ₀)   cos(m·θ₀)                              │
R_m =   │              cos(m·θ₁)  -sin(m·θ₁)                 │
        │              sin(m·θ₁)   cos(m·θ₁)                 │
        │                              ⋱                      │
        └                           cos(m·θ_{D/2-1}) ...     ┘
```

### Efficient Computation (No Matrix Multiply)

You don't actually build this matrix. Instead, apply rotations elementwise:

```
For dimension pair (2i, 2i+1) at position m:
  angle = m · θᵢ

  q̃[2i]   =  q[2i] · cos(angle) - q[2i+1] · sin(angle)
  q̃[2i+1] =  q[2i] · sin(angle) + q[2i+1] · cos(angle)
```

Vectorized as:

```python
q_even  = q[..., 0::2]              # (B, heads, N, D/2)
q_odd   = q[..., 1::2]              # (B, heads, N, D/2)

q_rotated_even = q_even * cos - q_odd * sin
q_rotated_odd  = q_even * sin + q_odd * cos

# Interleave back
q_rotated = torch.stack([q_rotated_even, q_rotated_odd], dim=-1).flatten(-2)
```

Or equivalently (the form used in our ViT):

```python
# Rotate: [x₀, x₁] → [x₀·cos - x₁·sin, x₀·sin + x₁·cos]
#         equivalently: rotate using [-x₁, x₀] as the perpendicular component
x1      = x[..., 0::2]
x2      = x[..., 1::2]
rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
output  = x * cos + rotated * sin
```

### Why This Encodes Relative Position

Attention score between token at position `m` (query) and position `n` (key):

```
score = q̃_m · k̃_n
      = (R_m · q) · (R_n · k)
      = qᵀ · R_mᵀ · R_n · k
      = qᵀ · R_{n-m} · k         ← depends ONLY on (n - m)
```

This is the fundamental property. The attention score between two tokens depends only on their relative distance, not their absolute positions. The model has no way to distinguish "position 5 attending to position 8" from "position 105 attending to position 108" — both have the same relative distance of 3.

---

## Part 4 — From 1D to 2D: Vision Requires Spatial Coordinates

Language has one dimension of position (token index). Images have two — row and column.

Patch at grid position `(r, c)` in an 8×8 patch grid needs to encode both spatial coordinates.

### The Naive Approach (Wrong)

Flatten the 2D grid into a 1D sequence and use 1D RoPE:

```
8×8 grid → [P00, P01, ..., P07, P10, P11, ..., P77]  (row-major order)

Patch (0,7) gets position 7.
Patch (1,0) gets position 8.

Distance in 1D: |7 - 8| = 1  ← looks adjacent!
Distance in 2D: they are in the same row: 7 cols apart horizontally, 1 row apart

Patch (0,0) gets position 0.
Patch (7,7) gets position 63.
1D distance: 63  ← looks far
2D distance: √(7² + 7²) ≈ 9.9 (in patch units)
```

1D RoPE on flattened tokens gives incorrect spatial distances. Patches that are far apart in 2D look close in 1D and vice versa.

### The Correct Approach: 2D RoPE

Assign each patch its actual grid coordinates `(row, col)`. Encode them independently using different halves of the head dimension:

```
Head dimension D = 64

Dims 0–31:   encode row coordinate   (vertical position)
Dims 32–63:  encode col coordinate   (horizontal position)

Patch (row=3, col=5):
  Dims 0–31:  rotated by 3 · θᵢ    (row frequencies)
  Dims 32–63: rotated by 5 · θᵢ    (col frequencies)

Patch (row=3, col=6):
  Dims 0–31:  rotated by 3 · θᵢ    (same row → same row rotation)
  Dims 32–63: rotated by 6 · θᵢ    (different col → different col rotation)
```

Attention score between patches `(r₁, c₁)` and `(r₂, c₂)`:

```
score = q̃ · k̃
      = (row part): depends on (r₂ - r₁)
      + (col part): depends on (c₂ - c₁)

The model sees: "these patches are 2 rows apart and 1 column apart"
```

---

## Part 5 — Complete Implementation

### Step 1: Build the Rotation Cache

```python
import torch

def build_2d_rope_cache(num_patches_side: int, head_dim: int, device):
    """
    Precompute cos/sin rotation matrices for all positions in the 2D patch grid.

    Args:
        num_patches_side: grid side length (8 for 32×32 image with patch_size=4)
        head_dim:         attention head dimension (64 for embed_dim=192, heads=3)
        device:           torch device

    Returns:
        cos: (num_patches, head_dim)  — num_patches = num_patches_side²
        sin: (num_patches, head_dim)
    """
    # Each of (row, col) uses head_dim/2 dimensions
    # Each half uses alternating pairs → quarter = head_dim/4 frequency values
    quarter = head_dim // 4

    # Frequency bands: θᵢ = 1 / 10000^(i/quarter)  for i in 0..quarter-1
    # Shape: (quarter,)
    theta = 1.0 / (10000 ** (
        torch.arange(0, quarter, device=device).float() / quarter
    ))

    # Grid coordinates: 0, 1, 2, ..., num_patches_side-1
    coords     = torch.arange(num_patches_side, device=device).float()
    # Build 2D grid: rows[i,j]=i, cols[i,j]=j
    rows, cols = torch.meshgrid(coords, coords, indexing='ij')

    # Flatten to 1D sequence (row-major): shape (num_patches,)
    rows = rows.flatten()   # [0,0,...,0, 1,1,...,1, ..., 7,7,...,7]
    cols = cols.flatten()   # [0,1,...,7, 0,1,...,7, ..., 0,1,...,7]

    # Outer product: position × frequency → angles matrix
    # row_freqs[i, j] = rows[i] * theta[j]
    row_freqs = torch.outer(rows, theta)   # (num_patches, quarter)
    col_freqs = torch.outer(cols, theta)   # (num_patches, quarter)

    # Concatenate: [row_freqs | col_freqs] → (num_patches, head_dim/2)
    freqs = torch.cat([row_freqs, col_freqs], dim=-1)  # (64, 32) for our case

    # cos/sin of the frequencies — each duplicated for the two rotation components
    # Shape: (num_patches, head_dim)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

    return cos, sin
```

### Concrete shape trace for our ViT:

```
num_patches_side = 8    (32×32 image, patch_size=4 → 8×8 grid)
head_dim         = 64   (embed_dim=192, num_heads=3 → 192/3=64)

quarter = 64 // 4 = 16

theta: shape (16,)
  [1.0, 0.681, 0.464, ..., 0.0001]   ← 16 frequency values

rows, cols after meshgrid: each (8, 8)
rows.flatten(): (64,)  [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, ..., 7,7,7,7,7,7,7,7]
cols.flatten(): (64,)  [0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7, ..., 0,1,2,3,4,5,6,7]

row_freqs = outer(rows, theta): (64, 16)
col_freqs = outer(cols, theta): (64, 16)

freqs = cat([row_freqs, col_freqs], dim=-1): (64, 32)

cos = cat([freqs.cos(), freqs.cos()], dim=-1): (64, 64) ✓
sin = cat([freqs.sin(), freqs.sin()], dim=-1): (64, 64) ✓
```

### Step 2: Apply RoPE to Q and K

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply precomputed 2D RoPE rotations to a query or key tensor.

    Args:
        x:   (B, num_heads, seq_len, head_dim)  — includes CLS at index 0
        cos: (num_patches, head_dim)             — from build_2d_rope_cache
        sin: (num_patches, head_dim)

    Returns:
        Rotated tensor, same shape as x.

    Note: CLS token (index 0) receives identity rotation — cos=1, sin=0.
    """
    seq_len = x.shape[2]

    # Prepend identity rotation for CLS token
    # Identity rotation: cos=1 (no scaling), sin=0 (no rotation)
    ones  = torch.ones(1,  cos.shape[-1], device=x.device)
    zeros = torch.zeros(1, sin.shape[-1], device=x.device)

    # Shape after cat: (num_patches+1, head_dim)
    # Slice to seq_len in case of any mismatch
    cos_full = torch.cat([ones,  cos], dim=0)[:seq_len]   # (seq_len, head_dim)
    sin_full = torch.cat([zeros, sin], dim=0)[:seq_len]

    # Expand for broadcasting: (1, 1, seq_len, head_dim)
    cos_full = cos_full.unsqueeze(0).unsqueeze(0)
    sin_full = sin_full.unsqueeze(0).unsqueeze(0)

    # Rotation: pair up even/odd dimensions
    # Even dims:  x[0], x[2], x[4], ...
    # Odd  dims:  x[1], x[3], x[5], ...
    x_even = x[..., 0::2]   # (B, heads, seq_len, head_dim/2)
    x_odd  = x[..., 1::2]   # (B, heads, seq_len, head_dim/2)

    # Interleave [-x_odd, x_even] to form the "perpendicular" vector
    # This is the rotation formula: rotate(x) = x*cos + perp(x)*sin
    # where perp([x₀, x₁]) = [-x₁, x₀]
    rotated = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)
    # Shape: (B, heads, seq_len, head_dim)

    return x * cos_full + rotated * sin_full
```

### Step 3: Use in the Model

```python
class VisionTransformer(nn.Module):
    def __init__(self, ...):
        ...
        # Register as buffers — not parameters, but move with .to(device)
        num_patches = (img_size // patch_size) ** 2   # 64
        head_dim    = embed_dim // num_heads           # 64
        self.register_buffer('rope_cos', torch.empty(num_patches, head_dim))
        self.register_buffer('rope_sin', torch.empty(num_patches, head_dim))

    def build_rope(self, device):
        """Call ONCE after model.to(device)."""
        cos, sin = build_2d_rope_cache(
            num_patches_side=self.patch_grid,   # 8
            head_dim=self.rope_cos.shape[-1],   # 64
            device=device
        )
        self.rope_cos.copy_(cos)
        self.rope_sin.copy_(sin)

    def forward(self, x):
        ...
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        ...

class AttentionBlock(nn.Module):
    def forward(self, x, rope_cos=None, rope_sin=None):
        ...
        q, k, v = ...
        if rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin)   # rotate Q
            k = apply_rope(k, rope_cos, rope_sin)   # rotate K
            # V is NOT rotated — only Q and K affect attention scores
        ...
```

**Why not rotate V?** RoPE encodes position in the attention scores `Q·Kᵀ`. The values `V` are the content retrieved after attention — they don't need position encoding. Rotating V would distort the retrieved content without providing positional information.

---

## Part 6 — Visualizing What 2D RoPE Encodes

### Attention Score as a Function of Relative Position

```
Two patches: A at (row_A, col_A) and B at (row_B, col_B)

Attention score (simplified, single dimension pair):
  score = cos((row_B - row_A) · θ_row) + cos((col_B - col_A) · θ_col)

When A and B are at the same position:
  score = cos(0) + cos(0) = 1 + 1 = 2   ← maximum

When A and B are 1 row apart, same column:
  score ≈ cos(θ_row) + cos(0) = cos(θ_row) + 1

When A and B are far apart:
  cos terms oscillate and tend to cancel → lower average score
```

### What the Model Learns

Because attention scores decrease (on average) as relative distance increases, the model naturally develops **local attention** in early layers without any explicit constraint:

```
Layer 1 attention from center patch (4,4):

Using 2D RoPE:                    Using learned PE:
  ░░░░░░░░                          ▒▒▒▒▒▒▒▒
  ░░▒▒▒░░░                          ▒▓▓▒▒▒▒▒
  ░▒███▒░░                          ▒▓█▓▒▒▒▒
  ░▒███▒░░   ← local by default     ▒▒▒▒▓▒▒▒   ← random global
  ░░▒▒▒░░░                          ▒▒▒▒▒▒▒▒
  ░░░░░░░░                          ▒▒▒▒▒▒▒▒
  ░░░░░░░░                          ▒▒▒▒▒▒▒▒
  ░░░░░░░░                          ▒▒▒▒▒▒▒▒

2D RoPE gives the model a spatial prior for free.
Learned PE must discover spatial structure from data.
```

---

## Part 7 — RoPE vs Learned PE: Full Comparison

```
┌──────────────────────────┬────────────────────────┬─────────────────────────┐
│ Property                 │ Learned PE             │ 2D RoPE                 │
├──────────────────────────┼────────────────────────┼─────────────────────────┤
│ Parameters               │ (N+1) × D = 12,480     │ 0                       │
│ Position type            │ Absolute               │ Relative                │
│ Encodes distance         │ Indirectly             │ Directly                │
│ Fixed to training size   │ Yes (N must match)     │ No (works at any N)     │
│ Spatial prior            │ None (learned from data│ Built-in (rotation ∝ d) │
│ CLS handling             │ Special learned vector │ Identity rotation       │
│ Transfer to new sizes    │ Requires interpolation │ Works natively          │
│ Used in                  │ Original ViT, BERT     │ LLaMA, GPT-NeoX,        │
│                          │                        │ LlamaGen, SD3, our ViT  │
└──────────────────────────┴────────────────────────┴─────────────────────────┘
```

---

## Part 8 — Common Mistakes and How to Avoid Them

### Mistake 1: Forgetting the CLS token

```python
# WRONG — cos/sin has shape (64, 64) for 64 patches
# But sequence has 65 tokens (CLS + 64 patches)
q = apply_rope(q, cos, sin)   # ← shape mismatch or silent error

# CORRECT — prepend identity rotation for CLS
ones  = torch.ones(1, head_dim, device=device)
zeros = torch.zeros(1, head_dim, device=device)
cos_full = torch.cat([ones, cos], dim=0)    # (65, head_dim)
sin_full = torch.cat([zeros, sin], dim=0)   # (65, head_dim)
```

### Mistake 2: Using the same RoPE for all heads

Each head has dimension `head_dim = embed_dim / num_heads`. You must build the RoPE cache for `head_dim`, not `embed_dim`:

```python
# WRONG
cos, sin = build_2d_rope_cache(8, embed_dim=192, device)  # 192-dim rotation

# CORRECT
head_dim = embed_dim // num_heads   # = 64
cos, sin = build_2d_rope_cache(8, head_dim=64, device)    # 64-dim rotation
```

### Mistake 3: Rotating V

Only rotate Q and K. V should never be rotated — it contains the value content, not the position keys.

```python
# WRONG
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
v = apply_rope(v, cos, sin)   # ← incorrect, distorts retrieved content

# CORRECT
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
# v unchanged
```

### Mistake 4: Rebuilding the cache every forward pass

```python
# WRONG — very slow
def forward(self, x):
    cos, sin = build_2d_rope_cache(...)   # recomputed every call
    ...

# CORRECT — precompute once, store as buffer
def build_rope(self, device):
    cos, sin = build_2d_rope_cache(...)
    self.rope_cos.copy_(cos)   # stored in buffer
    self.rope_sin.copy_(sin)

def forward(self, x):
    # Use precomputed buffers
    q = apply_rope(q, self.rope_cos, self.rope_sin)
```

---

## Part 9 — Sanity Checks

```python
import torch

# Build cache
cos, sin = build_2d_rope_cache(num_patches_side=8, head_dim=64, device='cpu')

# Shape check
assert cos.shape == (64, 64), f"Expected (64, 64), got {cos.shape}"
assert sin.shape == (64, 64)

# Trig identity: cos²(θ) + sin²(θ) = 1 for all entries
assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-6)

# CLS patch should have zero rotation (sin=0)
# After prepending: first row of sin_full should be zeros
ones  = torch.ones(1, 64)
zeros = torch.zeros(1, 64)
cos_full = torch.cat([ones, cos], dim=0)
sin_full = torch.cat([zeros, sin], dim=0)
assert torch.allclose(sin_full[0], torch.zeros(64), atol=1e-10)
assert torch.allclose(cos_full[0], torch.ones(64),  atol=1e-10)
print("CLS identity rotation: OK")

# Apply rotation to a query tensor
q = torch.randn(2, 3, 65, 64)   # (B, heads, seq_len, head_dim)
q_rotated = apply_rope(q, cos, sin)
assert q_rotated.shape == q.shape
assert not torch.isnan(q_rotated).any()
print("apply_rope shape: OK")

# Rotation preserves vector norms (rotation never changes magnitude)
q_norms = q.norm(dim=-1)
qr_norms = q_rotated.norm(dim=-1)
assert torch.allclose(q_norms, qr_norms, atol=1e-5), "Rotation changed vector norm!"
print("Norm preservation: OK")

# Relative position property:
# Attention score between q at pos 5 and k at pos 8
# should equal score between q at pos 0 and k at pos 3
# (both have relative distance 3)
# This is hard to test exactly but we can verify the rotation difference
cos0, sin0 = build_2d_rope_cache(4, 16, 'cpu')   # small example
# Patch (0,3) has index 3, Patch (1,0) has index 4
# Rotation for patch (0,3): cos[3], sin[3]
# Rotation for patch (1,0): cos[4], sin[4]
print("All sanity checks passed.")
```

---

## Summary

```
Key ideas:

1. Complex rotations:
   Multiplying by e^(iθ) rotates a 2D vector by θ.
   Dot products of rotated vectors depend only on rotation DIFFERENCE.

2. From rotation to position:
   Set rotation angle = position × frequency.
   Dot product → depends on position difference → relative position.

3. Frequency bands:
   Fast frequencies (large θ) → encode short-range structure.
   Slow frequencies (small θ) → encode long-range structure.
   Together: multi-scale spatial awareness.

4. 1D → 2D:
   Split head dimension in half.
   First half encodes row coordinate.
   Second half encodes column coordinate.
   Each patch gets (row, col) frequencies independently.

5. Implementation:
   Precompute cos/sin cache once per model (stored as buffer).
   Apply to Q and K only (never V).
   CLS token gets identity rotation.
   No parameters required.
```

---

## Citation

```bibtex
@article{su2021roformer,
  title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author={Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha
          and Bo Wen and Yunfeng Liu},
  journal={arXiv:2104.09864},
  year={2021}
}
```

---

*Tutorial by Koyilbek Valiev — AI / ML Research Engineer*
