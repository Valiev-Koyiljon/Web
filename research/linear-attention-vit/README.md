# Linear Attention in Vision Transformers: Breaking the O(N²) Barrier

**Tutorial by:** Koyilbek Valiev  
**Topics:** Vision Transformers · Efficient Attention · Linear Algebra · PyTorch  
**Paper:** [The Linear Attention Resurrection in Vision Transformer](https://arxiv.org/abs/2501.16182) — L2ViT (2025)

---

## The Problem: Attention Scales Quadratically

Every Vision Transformer contains this operation at its core:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) · V
```

For N tokens and embedding dimension D, the cost is:

```
QKᵀ        →   N × N matrix   ← THIS IS THE BOTTLENECK
softmax(·)  →   N × N matrix
· V         →   N × D output

Time:   O(N² · D)
Memory: O(N²)
```

For CIFAR-10 with 64 patches, N=65. That's fine. But watch what happens as resolution grows:

```
Image    Patch  Tokens (N)   Attention matrix   vs Linear
─────────────────────────────────────────────────────────
32×32    p=4       64         64×64   = 4k        1×
64×64    p=4      256        256×256  = 65k        16×
128×128  p=4     1024       1024×1024 = 1M        256×
256×256  p=4     4096       4096×4096 = 16M      4096×
512×512  p=4    16384      16384×16384= 268M    65536×
```

At 512×512, the attention matrix is **268 million values per head per layer**. For a 12-layer model with 3 heads, that's 9.6 billion floats — just for the attention matrix.

This is why high-resolution ViT inference is impossible without modification.

---

## The Key Insight: Matrix Associativity

Standard softmax attention cannot be factored. But what if we replaced `softmax` with a function that **can** be decomposed as a dot product?

If we can write the similarity between query `q` and key `k` as:

```
sim(q, k) = φ(q) · φ(k)ᵀ
```

where `φ` is any function (called a **feature map**), then:

```
Standard attention:
  output = (φ(Q) · φ(K)ᵀ) · V
           └─────────────┘
                N × N           ← must form this first

Linear attention (reordered):
  output = φ(Q) · (φ(K)ᵀ · V)
                   └─────────┘
                     D × D      ← compute this first!
```

This is just the **associativity of matrix multiplication**: `A(BC) = (AB)C`.

The result is identical — but the order of operations completely changes the complexity.

---

## The Math: Why Reordering Works

Let's trace the shapes carefully.

```
Inputs:
  Q: (N, D)   — N queries, each D-dimensional
  K: (N, D)   — N keys
  V: (N, D)   — N values

After feature map:
  φ(Q): (N, D)
  φ(K): (N, D)

Option A — Standard order:
  Step 1: φ(Q) · φ(K)ᵀ     (N,D) × (D,N) = (N,N)   ← quadratic!
  Step 2: (N,N) · V          (N,N) × (N,D) = (N,D)
  
  Total: O(N²D)

Option B — Reordered:
  Step 1: φ(K)ᵀ · V         (D,N) × (N,D) = (D,D)   ← constant in N!
  Step 2: φ(Q) · (D,D)       (N,D) × (D,D) = (N,D)
  
  Total: O(ND²)
```

**The N×N matrix never forms.** The `(D,D)` context matrix is computed once and shared across all N queries.

---

## Complexity Comparison

```
N = sequence length
D = feature/embedding dimension

Standard softmax attention:   O(N²·D)
Linear attention:             O(N·D²)

Cross-over point: N² = D²  →  N = D
For our ViT (D=192): linear attention wins when N > 192

Our token counts:
  CIFAR-10 32×32, patch=4:  N=64    → N < D, linear slower (but negligible)
  64×64    image, patch=4:  N=256   → N > D, linear 1.3× faster
  128×128  image, patch=4:  N=1024  → linear 5.3× faster
  256×256  image, patch=4:  N=4096  → linear 21× faster
  512×512  image, patch=4:  N=16384 → linear 85× faster
```

For our CIFAR-10 use case, the raw speedup is small. But implementing linear attention **now** means the architecture scales to larger images with no modification — the same code that trains on 32×32 can run on 512×512.

---

## The Feature Map: What is φ(x)?

The feature map replaces softmax. It must satisfy two requirements:

1. **Non-negative outputs** — ensures the denominator (normalizer) is always positive
2. **Approximates the exponential kernel** — `φ(q)·φ(k)ᵀ ≈ exp(q·kᵀ/√d)`

The standard choice from Katharopoulos et al. (2020):

```
φ(x) = elu(x) + 1

where elu(x) = x         if x > 0
               exp(x) - 1 if x ≤ 0

So φ(x) = x + 1          if x > 0   (linear, always ≥ 1)
           exp(x)          if x ≤ 0   (positive, ≥ 0)
```

Visual comparison of feature maps:

```
Value  softmax  elu+1    relu    relu²+ε
─────────────────────────────────────────
-3.0   0.004    0.050    0.000   0.001
-1.0   0.034    0.368    0.000   0.001
 0.0   0.100    1.000    0.000   0.001
+1.0   0.271    2.000    1.000   1.001
+2.0   0.736    3.000    2.000   4.001
+3.0   2.000    4.000    3.000   9.001

Key: all non-negative ✓
     positive for all inputs (elu+1 only) ✓ ← safest
```

---

## The Normalizer: Avoiding Division by Zero

Standard softmax normalizes automatically — rows sum to 1. Linear attention needs explicit normalization:

```
           φ(qᵢ) · (Σⱼ φ(kⱼ)ᵀ vⱼ)
output_i = ─────────────────────────
              φ(qᵢ) · (Σⱼ φ(kⱼ))
```

The denominator `φ(qᵢ) · (Σⱼ φ(kⱼ))` can approach zero for unusual input distributions. Always clamp it:

```python
denominator = denominator.clamp(min=1e-6)
```

Without clamping, linear attention produces NaN values during training.

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearGlobalAttention(nn.Module):
    """
    O(N·D²) attention via kernel feature map φ(x) = elu(x) + 1.
    Reorders computation to φ(Q) · (φ(K)ᵀ · V) — never forms N×N matrix.
    Includes learnable KV scale for numerical stability.
    L2ViT (2025): arXiv:2501.16182
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.qkv       = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.drop      = nn.Dropout(dropout)
        # Learnable scale controls variance of KV product
        self.kv_scale  = nn.Parameter(torch.ones(1) * 0.1)

    @staticmethod
    def phi(x):
        """Feature map: elu(x) + 1. Non-negative, smooth, approximates exp."""
        return F.elu(x) + 1

    def forward(self, x):
        B, N, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                  # each: (B, N, heads, head_dim)
        q = q.transpose(1, 2)                     # (B, heads, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply feature map (replaces softmax)
        q = self.phi(q)   # (B, heads, N, D)
        k = self.phi(k)   # (B, heads, N, D)

        # ── KEY STEP ─────────────────────────────────────────────────
        # Compute D×D context matrix FIRST — O(N·D²) total
        # φ(K)ᵀ · V: (B, heads, D, D)  ← constant size, no matter how large N is
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v) * self.kv_scale

        # Apply each query to the shared context: φ(Q) · (φ(K)ᵀV)
        # (B, heads, N, D)
        numerator = torch.einsum('bhnd,bhdm->bhnm', q, kv)

        # Normalisation: φ(Q) · Σφ(K) — prevents softmax-like normalization
        k_sum       = k.sum(dim=2)                          # (B, heads, D)
        denominator = torch.einsum('bhnd,bhd->bhn', q, k_sum)
        denominator = denominator.clamp(min=1e-6).unsqueeze(-1)

        out = (numerator / denominator).transpose(1, 2).reshape(B, N, C)
        return self.proj(self.drop(out))
```

---

## The Problem with Pure Linear Attention

Linear attention produces a fundamentally different attention pattern than softmax:

```
Softmax attention (concentrated):
  Token A attends to:
    Token B:  0.02
    Token C:  0.89   ← focused, selective
    Token D:  0.01
    Token E:  0.08

Linear attention (distributed):
  Token A attends to:
    Token B:  0.24   ← blurry, uniform
    Token C:  0.26
    Token D:  0.22
    Token E:  0.28
```

Softmax is a **peaky** distribution — it can focus almost entirely on one token. The kernel feature map φ produces a **flat** distribution — every token gets roughly equal weight.

This is the core limitation of linear attention. It has global reach (sees all tokens) but no local focus.

---

## The Fix: Local Concentration Module (LCM)

**L2ViT (2025)** introduces the **Local Concentration Module** — a lightweight depthwise 3×3 convolution applied after linear attention to restore local spatial precision:

```
Linear attention output:   blurry global context
         +
LCM (depthwise 3×3 conv): sharp local features
         =
Final output:              global context + local precision
```

A depthwise convolution operates independently on each channel, applying a 3×3 spatial filter. At patch resolution (8×8 for CIFAR-10), this looks at a 3×3 neighborhood of patches — exactly the local context that linear attention misses.

```python
class LocalConcentrationModule(nn.Module):
    """
    Depthwise 3×3 conv on the spatial patch grid.
    Restores local precision lost by the global kernel in linear attention.
    L2ViT (2025): arXiv:2501.16182
    """
    def __init__(self, embed_dim, patch_grid=8):
        super().__init__()
        self.H = self.W = patch_grid
        self.dw_conv = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3,
            padding=1, groups=embed_dim, bias=False   # depthwise
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) — patch tokens only (no CLS)
        B, N, C = x.shape
        # Reshape token sequence back to spatial grid for convolution
        x_2d = x.transpose(1, 2).reshape(B, C, self.H, self.W)
        x_2d = self.dw_conv(x_2d)                    # spatial 3×3 filter
        return self.norm(x_2d.flatten(2).transpose(1, 2))
```

**Why groups=embed_dim?** A regular Conv2d mixes channels — each output channel sees all input channels. A depthwise conv (`groups=embed_dim`) processes each channel independently — it's a spatial filter only, with 9 parameters per channel instead of 9×C². For C=192: `9 × 192 = 1,728` parameters vs `9 × 192² = 331,776`.

---

## Flash Attention: A Different Kind of Efficiency

Flash Attention (Dao et al., 2022) doesn't reduce the mathematical complexity — it's still O(N²D). But it dramatically reduces **memory bandwidth** by rewriting the algorithm to be IO-aware.

```
Standard attention memory access pattern:
  1. Write QKᵀ to HBM (slow, 16MB for N=1024)    ← slow
  2. Read QKᵀ from HBM
  3. Apply softmax, write back to HBM             ← slow
  4. Read again, multiply by V                    ← slow

Flash Attention memory access pattern:
  1. Load a TILE of Q, K, V into SRAM (fast)
  2. Compute partial attention IN SRAM
  3. Write only the final output to HBM           ← one write
  
  Never materializes the full N×N matrix.
```

In PyTorch 2.0+, Flash Attention is available as a single function call:

```python
class FlashAttentionBlock(nn.Module):
    """
    IO-aware softmax attention. Mathematically identical to standard attention
    but never writes the N×N matrix to slow GPU memory (HBM).
    Built into PyTorch 2.0+ as F.scaled_dot_product_attention.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.qkv       = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.drop_p    = dropout

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        # Flash Attention — tiled SRAM computation, no N×N in HBM
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.drop_p if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(B, N, C))
```

---

## L2ViT: The Alternating Design

The key insight of **L2ViT** (2025): alternate linear and softmax attention across encoder layers.

```
Layer 0  →  LinearGlobalAttention + LCM    cheap global context
Layer 1  →  FlashAttention                 precise local attention
Layer 2  →  LinearGlobalAttention + LCM    cheap global context
Layer 3  →  FlashAttention                 precise local attention
...×12 layers total
```

This gives you the best of both:

```
┌────────────────────────────────────────────────────────────────┐
│                    Attention Type Comparison                   │
├──────────────────────┬──────────────────┬──────────────────────┤
│                      │ Linear (even)    │ Flash (odd)          │
├──────────────────────┼──────────────────┼──────────────────────┤
│ Complexity           │ O(N·D²)          │ O(N²·D) IO-aware     │
│ Memory               │ O(D²) constant   │ O(N) tiled SRAM      │
│ Attention pattern    │ Global, blurry   │ Sharp, selective     │
│ Local precision      │ Low (fixed by LCM│ High                 │
│ Scales with N        │ Linearly ✓       │ Quadratically        │
│ Used for             │ Long-range deps  │ Local precision      │
└──────────────────────┴──────────────────┴──────────────────────┘
```

**Result:** L2ViT achieves **84.4% Top-1 on ImageNet** without extra data, matching ViT-Base at a fraction of the computation for large images.

---

## Complete Alternating Block

```python
class AlternatingTransformerBlock(nn.Module):
    """
    Even block_idx  →  LinearGlobalAttention + LocalConcentrationModule
    Odd  block_idx  →  FlashAttentionBlock
    Both use DynamicTanh normalization and an MLP with dropout.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, block_idx,
                 dropout=0.1, patch_grid=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        if block_idx % 2 == 0:
            self.attn = LinearGlobalAttention(embed_dim, num_heads, dropout)
            self.lcm  = LocalConcentrationModule(embed_dim, patch_grid)
        else:
            self.attn = FlashAttentionBlock(embed_dim, num_heads, dropout)
            self.lcm  = None

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Attention sub-layer
        attn_out = self.attn(self.norm1(x))

        # LCM on patch tokens (skip CLS at index 0) for even blocks
        if self.lcm is not None:
            patch_out = self.lcm(attn_out[:, 1:, :])
            attn_out  = torch.cat([attn_out[:, :1, :], patch_out], dim=1)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## Attention Pattern: Visualizing the Difference

```
Image: 8×8 patch grid (64 patches), all three attention types

Patch (4,4) — center of image

Softmax attention map:
  ░░░░░░░░
  ░░▒▒░░░░
  ░▒████▒░    ← focused on center region
  ░▒████▒░
  ░░▒▒░░░░
  ░░░░░░░░
  ░░░░░░░░
  ░░░░░░░░

Linear attention map:
  ▒▒▒▒▒▒▒▒
  ▒▒▒▒▒▒▒▒    ← nearly uniform (blurry)
  ▒▒▒▒▒▒▒▒
  ▒▒▒▒▒▒▒▒
  ▒▒▒▒▒▒▒▒
  ▒▒▒▒▒▒▒▒

Linear + LCM map:
  ░░░░░░░░
  ░░▒▒░░░░
  ░▒███▒░░    ← restored local focus
  ░▒████▒░
  ░░░▒▒░░░
  ░░░░░░░░
```

---

## When to Use Each Attention Type

```
Your image resolution:

< 128×128 (e.g. CIFAR-10 32×32, ImageNet 64px):
  → Standard softmax or Flash Attention
  → N is small, quadratic cost is acceptable
  → Linear attention adds code complexity for minimal gain

128×128 to 512×512:
  → L2ViT alternating design recommended
  → Even blocks save ~40-60% compute
  → LCM critical to maintain quality

> 512×512 (medical, satellite, document images):
  → Pure linear attention or window-based attention
  → Softmax attention becomes infeasible in memory
  → Consider sliding window attention (Longformer style)
```

---

## Citations

```bibtex
@article{zheng2025linear,
  title={The Linear Attention Resurrection in Vision Transformer},
  author={Chuanyang Zheng et al.},
  journal={arXiv:2501.16182},
  year={2025}
}
@article{katharopoulos2020transformers,
  title={Transformers are RNNs: Fast Autoregressive Transformers with 
         Linear Attention},
  author={Katharopoulos et al.},
  year={2020}
}
@article{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention 
         with IO-Awareness},
  author={Tri Dao et al.},
  year={2022}
}
```

---

*Tutorial by Koyilbek Valiev — AI / ML Research Engineer*
