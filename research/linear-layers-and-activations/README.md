# Linear Layers & Activation Functions: The Building Blocks of Neural Networks

**Tutorial by:** Koyilbek Valiev  
**Topics:** Deep Learning · Neural Networks · Math · Transformers

---

## Why These Two Primitives Matter

Every neural network — from a two-layer perceptron to a 70-billion parameter LLM — is built from exactly two primitives repeated over and over:

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   Input  →  [Linear Layer]  →  [Activation]  →  Output        │
│                  ↑                   ↑                         │
│          mixes & rescales      breaks linearity                │
│          information           adds expressive power           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

Understand these two and you understand the computational core of transformers, CNNs, and LLMs.

---

## Part 1 — Linear Layers

### The Formula

A linear layer (also called a **fully connected** or **dense** layer) applies a learned linear transformation:

```
Given:
  x ∈ ℝⁿ   — input vector   (n input features)
  W ∈ ℝᵐˣⁿ — weight matrix  (m outputs, n inputs)
  b ∈ ℝᵐ   — bias vector

Output:
  y = W · x + b    →    y ∈ ℝᵐ

Each output neuron yᵢ is a weighted sum of ALL inputs plus a bias:

  y₁ = W₁₁·x₁ + W₁₂·x₂ + ... + W₁ₙ·xₙ + b₁
  y₂ = W₂₁·x₁ + W₂₂·x₂ + ... + W₂ₙ·xₙ + b₂
  ⋮
  yₘ = Wₘ₁·x₁ + Wₘ₂·x₂ + ... + Wₘₙ·xₙ + bₘ
```

### Concrete Walkthrough

```
Input:   x = [2, 3]     (n=2 features)

Weights: W = [[ 1,  0],
              [ 0,  1],
              [ 1,  1]]  (m=3 outputs, n=2 inputs)

Bias:    b = [1, 0, -1]

Compute each output:
  y[0] = 1·2 + 0·3 + 1  =  3
  y[1] = 0·2 + 1·3 + 0  =  3
  y[2] = 1·2 + 1·3 - 1  =  4

Result:  y = [3, 3, 4]    (shape: m=3)

The layer projected a 2D input into 3D output space.
```

### What It Does Geometrically

```
A linear layer performs three geometric operations:

  1. ROTATION    — W rotates the input in space
  2. SCALING     — W stretches or compresses along axes
  3. TRANSLATION — b shifts the result by a fixed offset

  Input space ℝⁿ  ─────── W·x + b ──────▶  Output space ℝᵐ

  Dimensionality can change:
  ┌─────────────────────────────────────────────────────┐
  │  Project UP:   n=2   → m=512   (embed into richer   │
  │                                 representation)     │
  │  Project DOWN: n=512 → m=64    (compress, bottleneck│
  │                                 distill)            │
  │  Stay same:    n=768 → m=768   (rotate/mix features)│
  └─────────────────────────────────────────────────────┘
```

### Why Bias Matters

```
Without bias (b = 0):
  y = W · x

  The output is always 0 when x = 0.
  Every hyperplane learned by the layer passes through the origin.
  The model cannot shift its decision boundaries freely.

With bias:
  y = W · x + b

  The model can shift the output by any amount.
  Decision boundaries can be placed anywhere in space.
  This is the same reason we add β in LayerNorm.
```

### Counting Parameters

```
Linear(n, m) has:
  Weight: m × n  parameters
  Bias:   m      parameters
  Total:  m × (n + 1)

Examples from a GPT-2 style model (d=768):

  Q projection   Linear(768, 768)  →  768 × 769  =  590,592  params
  FFN expand     Linear(768, 3072) →  3072 × 769 = 2,362,368 params
  FFN compress   Linear(3072, 768) →  768 × 3073 = 2,360,064 params

In a 7B LLM (d=4096, FFN=14336):
  One FFN layer: 4096 × 14336 × 2 ≈ 117M parameters
  × 32 layers    ≈ 3.7B params — most of the model lives here
```

### Stacking Linear Layers Alone Does Nothing

```
Two linear layers in sequence:

  h = W₁ · x + b₁
  y = W₂ · h + b₂
    = W₂ · (W₁ · x + b₁) + b₂
    = (W₂W₁) · x + (W₂b₁ + b₂)
       ↑___________↑
       still just ONE linear layer

No matter how many linear layers you stack without anything
between them, the result is always equivalent to a single
linear layer. Depth buys nothing.

This is exactly why activation functions exist.
```

---

## Part 2 — Activation Functions

An activation function is a **nonlinear function** applied element-wise after a linear layer. It is the only source of nonlinearity in a neural network — and nonlinearity is what gives deep networks the ability to approximate any function.

```
With activations:
  h = σ(W₁ · x + b₁)
  y = W₂ · h + b₂

  Now W₂ · σ(W₁ · x + b₁)  ≠  (W₂W₁) · x + const
  The composition is genuinely nonlinear.
  Deep networks can now approximate arbitrarily complex functions.
  (Universal Approximation Theorem)
```

---

### Sigmoid

```
Formula:    σ(x) = 1 / (1 + e⁻ˣ)

Output range: (0, 1)

      1.0 ┤                ╭──────────────
      0.5 ┤           ╭───╯
      0.0 ┤──────────╯
          └──────────────────────────── x
               -5    0    +5

Derivative:   σ'(x) = σ(x) · (1 - σ(x))
              Max value = 0.25 (at x=0)
              At x=±5: σ'≈ 0.007  ← nearly zero
```

**Problem — Vanishing Gradients:**

```
During backpropagation, gradients are multiplied layer by layer.
Sigmoid's derivative is at most 0.25 everywhere.

In a 10-layer network:
  gradient ≈ 0.25¹⁰ = 0.000001   ← vanished

Early layers learn essentially nothing.
```

**Use today:** Output layer for binary classification only (`P(class=1)`).

---

### Tanh

```
Formula:    tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)

Output range: (-1, 1)

       1.0 ┤               ╭─────────────
       0.0 ┤──────────────╱──────────────
      -1.0 ┤─────────────╯
           └──────────────────────────── x
                -5    0    +5

Derivative:   tanh'(x) = 1 - tanh²(x)
              Max value = 1.0 (at x=0)
              At x=±3: tanh'≈ 0.01  ← saturates here too
```

**Improvement over Sigmoid:**

```
  Sigmoid:  output ∈ (0, 1)   — not zero-centered
            gradients always positive → zig-zag gradient updates

  Tanh:     output ∈ (-1, 1)  — zero-centered
            gradients can be positive or negative → cleaner updates

Still saturates at extremes. Largely replaced by ReLU for hidden layers.
```

**Use today:** RNNs (LSTM cell state gates), DyT normalization layer.

---

### ReLU (Rectified Linear Unit)

```
Formula:    ReLU(x) = max(0, x)

Output range: [0, ∞)

      out ┤              ╱
          ┤             ╱
          ┤            ╱
      0.0 ┤───────────╱
          └──────────────────────────── x
                      0

Derivative:   ReLU'(x) = 1  if x > 0
                        = 0  if x ≤ 0
```

**Why ReLU dominated (2012–2020):**

```
  1. No saturation for x > 0  →  gradients don't vanish
  2. Computationally trivial   →  just a max(0, x)
  3. Sparse activations        →  ~50% neurons output 0
                                  sparse = efficient
```

**Problem — Dying ReLU:**

```
If a neuron's pre-activation is always negative:
  ReLU(x) = 0  always
  gradient = 0  always
  weights never update
  neuron is permanently "dead"

Can affect 10–40% of neurons in poorly initialized networks.
```

**Variants to fix dying ReLU:**

```
  Leaky ReLU:   f(x) = max(0.01x, x)      ← small slope for x<0
  ELU:          f(x) = x        if x ≥ 0
                       α(eˣ - 1) if x < 0  ← smooth negative
  PReLU:        f(x) = max(αx, x)          ← α is learned
```

**Use today:** Still common in CNNs. Mostly replaced by GELU in transformers.

---

### GELU (Gaussian Error Linear Unit)

```
Formula:    GELU(x) = x · Φ(x)

  Φ(x) = CDF of the standard normal distribution
        = (1/2) · [1 + erf(x / √2)]

Fast approximation:
  GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))

Output range: (-0.17, ∞)

      out ┤               ╱
          ┤              ╱
      0.0 ┤─────────────╱──────────────
          ┤      ╲     ╱
     -0.17┤       ╰───╯
          └──────────────────────────── x
               -3   0   +3

Derivative: smooth everywhere, no dead neurons.
```

**Why GELU replaced ReLU in transformers:**

```
  ReLU:  hard gate — either 0 or pass through
         not smooth at x=0 (kink in the curve)

  GELU:  soft gate — weights input by how likely it is
         to be positive (under a normal distribution)
         smooth everywhere — better gradient flow

  Intuition: GELU "stochastically" zeroes inputs proportional
  to how small they are. Small activations are suppressed
  softly, not hard-clipped to 0.
```

**Ablation (BERT pre-training, from the paper):**

```
  Activation   │  GLUE Score
  ─────────────┼─────────────
  ReLU         │  82.1
  ELU          │  82.3
  GELU         │  82.8  ← best

Used in: BERT, GPT-2, GPT-3, ViT, RoBERTa, T5 — standard
for nearly all transformer architectures 2018–2023.
```

---

### SwiGLU

```
Formula:    SwiGLU(x, W, V) = Swish(xW) ⊙ (xV)

  Swish(x)  = x · σ(x)    (sigmoid-weighted linear)
  ⊙          = element-wise multiplication (gating)

Expanded:
  gate   = Swish(x · W₁ + b₁)   ← "what to pass through"
  value  = x · W₂ + b₂          ← "the actual values"
  output = gate ⊙ value
```

**What gating means:**

```
Standard FFN:
  h = GELU(x · W₁ + b₁)
  y = h · W₂ + b₂

  Every feature flows through; GELU decides how much.


SwiGLU FFN:
  gate  = Swish(x · Wg + bg)   ← learned per-feature gate [0, 1]
  value = x · Wv + bv
  h     = gate ⊙ value          ← gate controls information flow
  y     = h · W₂ + b₂

  The network learns WHICH features to pass and WHICH to suppress.
  Dynamic, input-dependent gating. Richer than a fixed nonlinearity.
```

**Why the FFN needs three matrices instead of two:**

```
Standard FFN with GELU:       SwiGLU FFN:
  Linear(d, 4d)   ← 1 up      Linear(d, 8d/3)  ← gate proj
  GELU                         Linear(d, 8d/3)  ← value proj
  Linear(4d, d)   ← 1 down    Swish + multiply
                               Linear(8d/3, d)  ← down proj

SwiGLU needs 3 matrices, but uses 8d/3 instead of 4d width
to keep parameter count equal to the standard FFN.
```

**Ablation (PaLM paper):**

```
  Activation    │  Perplexity (↓ better)
  ──────────────┼────────────────────────
  ReLU          │  8.4
  GELU          │  8.1
  SwiGLU        │  7.6  ← best

Used in: LLaMA 2, LLaMA 3, PaLM, Gemma, Mistral, Qwen, DeepSeek
— the standard activation for all modern open-source LLMs.
```

---

## Part 3 — Linear + Activation in a Full Transformer FFN

The Feed-Forward Network inside every transformer block is exactly two linear layers with an activation between them:

```
┌──────────────────────────────────────────────────────────────┐
│                FEED-FORWARD NETWORK (FFN)                    │
│                                                              │
│  Input x ∈ ℝᵈ   (d = hidden dim, e.g. 4096 for LLaMA-7B)   │
│      │                                                       │
│      ▼                                                       │
│  ┌───────────────────────────────────┐                       │
│  │  Linear(d → 4d)  +  bias          │  expand               │
│  │  y = W₁ · x + b₁                 │  d=4096 → 16384       │
│  └──────────────────┬────────────────┘                       │
│                     │                                        │
│                     ▼                                        │
│  ┌───────────────────────────────────┐                       │
│  │  Activation Function              │  nonlinearity         │
│  │  GELU(y)  or  SwiGLU(y)          │                       │
│  └──────────────────┬────────────────┘                       │
│                     │                                        │
│                     ▼                                        │
│  ┌───────────────────────────────────┐                       │
│  │  Linear(4d → d)  +  bias          │  compress             │
│  │  z = W₂ · h + b₂                 │  16384 → 4096         │
│  └──────────────────┬────────────────┘                       │
│                     │                                        │
│  Output z ∈ ℝᵈ     ▼                                        │
└──────────────────────────────────────────────────────────────┘

Why expand then compress?
  The 4× wider hidden space lets the network represent richer
  combinations of features. The compression forces it to distill
  those into the most useful d-dimensional representation.
  This bottleneck structure is the "thinking space" of the model.
```

---

## Part 4 — PyTorch Implementations

### Manual Linear Layer

```python
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # Xavier uniform initialization — keeps variance stable across layers
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        out = x @ self.weight.T          # matrix multiply
        if self.bias is not None:
            out = out + self.bias
        return out


# Verify shapes
layer = Linear(4, 8)
x = torch.randn(2, 10, 4)   # batch=2, seq=10, features=4
print(layer(x).shape)        # (2, 10, 8)
```

### All Activation Functions

```python
import torch
import torch.nn.functional as F


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# Test all on the same input
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

print("Input:  ", x.tolist())
print("Sigmoid:", [f"{v:.3f}" for v in sigmoid(x).tolist()])
print("Tanh:   ", [f"{v:.3f}" for v in tanh(x).tolist()])
print("ReLU:   ", [f"{v:.3f}" for v in relu(x).tolist()])
print("GELU:   ", [f"{v:.3f}" for v in gelu(x).tolist()])
print("Swish:  ", [f"{v:.3f}" for v in swish(x).tolist()])

# Input:   [-3.0, -1.0, 0.0, 1.0, 3.0]
# Sigmoid: [0.047, 0.269, 0.500, 0.731, 0.953]
# Tanh:    [-0.995, -0.762, 0.000, 0.762, 0.995]
# ReLU:    [0.000, 0.000, 0.000, 1.000, 3.000]
# GELU:    [-0.004, -0.159, 0.000, 0.841, 2.996]
# Swish:   [-0.142, -0.269, 0.000, 0.731, 2.858]
```

### Standard FFN (GELU — used in BERT, GPT-2, ViT)

```python
class FFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)          # expand: d → 4d
        x = F.gelu(x)            # nonlinearity
        x = self.fc2(x)          # compress: 4d → d
        return x


model = FFN(dim=768)
x   = torch.randn(2, 128, 768)
out = model(x)
print(out.shape)   # (2, 128, 768)

params = sum(p.numel() for p in model.parameters())
print(f"FFN params: {params:,}")   # FFN params: 4,722,432
```

### SwiGLU FFN (used in LLaMA, Gemma, Mistral)

```python
class SwiGLUFFN(nn.Module):
    """
    LLaMA-style FFN with SwiGLU activation.
    Uses hidden_dim = 8/3 * dim (rounded) to match parameter count
    of a standard 4x-expansion FFN.
    """
    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = int(8 * dim / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)  # round to 256

        self.gate_proj  = nn.Linear(dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj  = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate  = F.silu(self.gate_proj(x))   # SiLU ≡ Swish
        value = self.value_proj(x)
        x     = gate * value                # element-wise gate
        return self.down_proj(x)


model = SwiGLUFFN(dim=768)
x   = torch.randn(2, 128, 768)
out = model(x)
print(out.shape)   # (2, 128, 768)

params = sum(p.numel() for p in model.parameters())
print(f"SwiGLU FFN params: {params:,}")    # SwiGLU FFN params: 4,718,592
```

### Complete Transformer Block

```python
class TransformerBlock(nn.Module):
    """
    Single transformer block with Pre-RMSNorm and SwiGLU FFN —
    the architecture used by LLaMA 3, Mistral, Gemma.
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn   = SwiGLUFFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm attention + residual
        h, _ = self.attn(*[self.norm1(x)] * 3)
        x    = x + h

        # Pre-Norm FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x


block = TransformerBlock(dim=768, num_heads=12)
x     = torch.randn(2, 128, 768)
out   = block(x)
print(out.shape)   # (2, 128, 768)
```

---

## Part 5 — Complete Comparison

### Activation Functions

```
┌──────────┬────────────────┬────────────────┬──────────────────────────────┐
│ Function │ Output range   │ Key weakness   │ Used in                      │
├──────────┼────────────────┼────────────────┼──────────────────────────────┤
│ Sigmoid  │ (0, 1)         │ vanishing grad │ output layer (binary classif)│
│ Tanh     │ (-1, 1)        │ vanishing grad │ RNNs, LSTM gates, DyT        │
│ ReLU     │ [0, ∞)         │ dying neurons  │ CNNs, older networks         │
│ GELU     │ (-0.17, ∞)     │ —              │ BERT, GPT-2, ViT             │
│ SwiGLU   │ gated          │ 3 matrices     │ LLaMA, Gemma, Mistral, Qwen  │
└──────────┴────────────────┴────────────────┴──────────────────────────────┘
```

### Timeline of Adoption

```
1986  Sigmoid     — backprop paper; first widely used activation
1990s Tanh        — zero-centered sigmoid; used in RNNs
2012  ReLU        — AlexNet; solved vanishing gradients for deep CNNs
2016  ELU         — smooth negative region; short-lived improvement
2018  GELU        — BERT; smooth + probabilistic gating; replaced ReLU
2020  Swish/SiLU  — self-gated; equivalent to SwiGLU's gate component
2022  SwiGLU      — PaLM, LLaMA; learned gating; dominates modern LLMs
```

---

## Key Takeaways

```
1. LINEAR LAYER = learned weighted sum
   y = Wx + b transforms inputs across dimensions.
   W mixes features; b shifts the output.
   Bias is essential — without it, all hyperplanes pass through origin.

2. STACKING LINEAR LAYERS ALONE IS USELESS
   Two linear layers = one linear layer.
   Activation functions are what make depth meaningful.

3. ACTIVATION = nonlinearity
   Applied element-wise after each linear layer.
   Breaks the linear collapse and enables universal approximation.

4. SIGMOID AND TANH SATURATE
   Derivatives approach zero at extremes → vanishing gradients.
   Do not use in deep hidden layers.

5. RELU IS SIMPLE AND FAST
   No saturation for positive inputs.
   But dying neurons are a real problem at scale.

6. GELU IS THE TRANSFORMER DEFAULT (pre-2022)
   Smooth, probabilistic soft-gating.
   Standard in BERT, GPT-2, GPT-3, ViT.

7. SWIGLU DOMINATES MODERN LLMS
   Learned input-dependent gating.
   Three matrices but richer representations.
   LLaMA, Mistral, Gemma, Qwen, DeepSeek all use it.
```

---

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Nair, V. & Hinton, G. (2010). Rectified Linear Units Improve Restricted Boltzmann Machines. *ICML 2010*
- Hendrycks, D. & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). *arXiv:1606.08415*
- Ramachandran, P. et al. (2017). Searching for Activation Functions (Swish). *arXiv:1710.05941*
- Noam, S. et al. (2020). GLU Variants Improve Transformer (SwiGLU). *arXiv:2002.05202*
- Touvron, H. et al. (2023). LLaMA 2. *arXiv:2307.09288*

---

*Tutorial by Koyilbek Valiev — AI / ML Engineer | Research Engineer*
