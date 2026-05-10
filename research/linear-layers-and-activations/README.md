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

Two properties every activation must have:

```
  1. NONLINEAR  — otherwise stacking layers is pointless
  2. DIFFERENTIABLE (or mostly so) — gradients must flow during backprop
```

---

### 1. Sigmoid

**Introduced:** 1986 (Rumelhart et al., backpropagation paper)

```
Formula:
  σ(x) = 1 / (1 + e⁻ˣ)

Output range: (0, 1)   — strictly between 0 and 1, never reaches either

Derivative:
  σ'(x) = σ(x) · (1 - σ(x))
         = e⁻ˣ / (1 + e⁻ˣ)²
```

**Output curve and derivative:**

```
  Output σ(x):                    Derivative σ'(x):

  1.0 ┤              ╭─────       0.25┤     ╭───╮
  0.8 ┤          ╭──╯             0.20┤   ╭╯   ╰╮
  0.5 ┤─────────╱                 0.10┤ ╭╯       ╰╮
  0.2 ┤    ╭───╯                  0.0 ┤╯           ╰──
  0.0 ┤───╯                           └────────────────
      └──────────────── x                  -5  0  +5
           -5  0  +5
```

**Numerical values:**

```
  x      σ(x)    σ'(x)
  ─────────────────────
  -5     0.007   0.007   ← near-zero gradient
  -3     0.047   0.045
  -1     0.269   0.197
   0     0.500   0.250   ← max gradient = 0.25
  +1     0.731   0.197
  +3     0.953   0.045
  +5     0.993   0.007   ← near-zero gradient
```

**Problem — Vanishing Gradients:**

```
During backpropagation, gradients multiply across layers:

  ∂L/∂W₁ = ∂L/∂y · σ'(layer_n) · σ'(layer_{n-1}) · ... · σ'(layer_1)

  Max gradient per layer = 0.25

  In a 10-layer network:
    gradient ≈ 0.25¹⁰ = 0.0000001   ← essentially zero
    Layer 1 never learns.

  In a 20-layer network:
    gradient ≈ 0.25²⁰ ≈ 10⁻¹²      ← completely dead
```

**Problem — Not Zero-Centered:**

```
  σ(x) always outputs positive values (0 to 1).
  Gradients flowing back are always the same sign.
  This causes zig-zag weight updates — slower convergence.

  Example: to move weights both up and down simultaneously,
  the optimizer must take a zig-zag path instead of a direct one.
```

**Use today:** Output layer only — `P(class=1)` in binary classification.

---

### 2. Tanh (Hyperbolic Tangent)

**Introduced:** 1990s — widely adopted for RNNs

```
Formula:
  tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)

Relationship to sigmoid:
  tanh(x) = 2 · σ(2x) - 1   ← just a rescaled sigmoid

Output range: (-1, 1)   — zero-centered

Derivative:
  tanh'(x) = 1 - tanh²(x)
```

**Output curve and derivative:**

```
  Output tanh(x):                 Derivative tanh'(x):

  1.0 ┤            ╭──────        1.0 ┤    ╭─╮
  0.5 ┤        ╭──╯               0.5 ┤  ╭╯  ╰╮
  0.0 ┤───────╱───────────        0.0 ┤╭╯      ╰──────
 -0.5 ┤   ╰──╮                       └────────────────
 -1.0 ┤──────╯                             -3  0  +3
      └──────────────── x
           -3  0  +3
```

**Numerical values:**

```
  x      tanh(x)   tanh'(x)
  ──────────────────────────
  -3     -0.995    0.010   ← near-zero gradient
  -2     -0.964    0.071
  -1     -0.762    0.420
   0      0.000    1.000   ← max gradient = 1.0
  +1      0.762    0.420
  +2      0.964    0.071
  +3      0.995    0.010   ← near-zero gradient
```

**Tanh vs Sigmoid:**

```
  ┌──────────────┬───────────────────┬───────────────────────┐
  │              │ Sigmoid           │ Tanh                  │
  ├──────────────┼───────────────────┼───────────────────────┤
  │ Output range │ (0, 1)            │ (-1, 1)               │
  │ Zero-centered│ No                │ Yes                   │
  │ Max gradient │ 0.25              │ 1.0                   │
  │ Saturates    │ Yes (x > ±5)      │ Yes (x > ±3)          │
  └──────────────┴───────────────────┴───────────────────────┘

  Tanh has 4× larger gradients at center → faster learning.
  But still saturates → still causes vanishing gradients in deep nets.
```

**Use today:** LSTM and GRU gates, DyT normalization (as a compression function).

---

### 3. ReLU (Rectified Linear Unit)

**Introduced:** 2010 (Nair & Hinton); popularized by AlexNet 2012

```
Formula:
  ReLU(x) = max(0, x)

Equivalent to:
  ReLU(x) = x   if x > 0
           = 0   if x ≤ 0

Output range: [0, ∞)

Derivative:
  ReLU'(x) = 1   if x > 0
            = 0   if x ≤ 0
            = undefined at x=0 (in practice, set to 0 or 0.5)
```

**Output curve and derivative:**

```
  Output ReLU(x):                 Derivative ReLU'(x):

  3.0 ┤              ╱           1.0 ┤         ──────────
  2.0 ┤             ╱            0.5 ┤
  1.0 ┤            ╱             0.0 ┤─────────
  0.0 ┤───────────╱                  └────────────────────
      └──────────────── x                   0
            -3  0  +3
```

**Numerical values:**

```
  x      ReLU(x)   ReLU'(x)
  ──────────────────────────
  -3      0.000    0         ← dead zone
  -1      0.000    0
   0      0.000    0
  +1      1.000    1
  +2      2.000    1
  +3      3.000    1         ← full gradient, no saturation
```

**Why ReLU solved the vanishing gradient problem:**

```
  For x > 0:  gradient = 1 exactly, no matter how large x is.
  No shrinkage. Gradients flow perfectly through positive neurons.

  10-layer network with all positive activations:
    gradient = 1¹⁰ = 1.0   ← no vanishing!

  This is why AlexNet (2012) was a breakthrough — first deep
  network to train reliably without vanishing gradients.
```

**Problem — Dying ReLU:**

```
If a neuron receives negative input consistently:
  pre-activation = Wx + b < 0  always
  ReLU output    = 0  always
  gradient       = 0  always
  ΔW             = 0  always   ← weights never update

The neuron is "dead" — permanently stuck, contributing nothing.
Can affect 10–40% of neurons with bad initialization or high LR.
```

**ReLU Variants:**

```
  Leaky ReLU:  f(x) = max(0.01x, x)
               Small negative slope → neurons can recover
               Gradient is never exactly 0

  PReLU:       f(x) = max(αx, x)
               α is a learned parameter
               Network decides how "leaky" to be

  ELU:         f(x) = x           if x ≥ 0
                    = α(eˣ - 1)   if x < 0
               Smooth at x=0, mean output closer to 0

  ┌─────────────┬──────────────────────────────────────────────┐
  │ Variant     │ Negative region behaviour                    │
  ├─────────────┼──────────────────────────────────────────────┤
  │ ReLU        │ exactly 0 — neurons can die                  │
  │ Leaky ReLU  │ 0.01x — tiny gradient, neurons survive       │
  │ PReLU       │ αx (learned) — adaptive                      │
  │ ELU         │ α(eˣ-1) — smooth, saturates to -α           │
  └─────────────┴──────────────────────────────────────────────┘
```

**Use today:** Standard in CNNs (ResNet, VGG, EfficientNet). Largely replaced by GELU in transformers.

---

### 4. GELU (Gaussian Error Linear Unit)

**Introduced:** Hendrycks & Gimpel, 2016. Adopted widely in 2018 with BERT.

```
Formula:
  GELU(x) = x · Φ(x)

  Φ(x) = CDF of the standard normal distribution
        = P(X ≤ x)  where X ~ N(0, 1)
        = (1/2) · [1 + erf(x / √2)]

  Interpretation: multiply x by the probability that a standard
  normal random variable is ≤ x. Small x gets suppressed
  proportionally to how "unlikely" it is to be active.

Fast approximation (used in practice):
  GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))

Output range: approximately (-0.17, ∞)
  Minimum at x ≈ -0.75:  GELU(-0.75) ≈ -0.17
```

**Output curve and derivative:**

```
  Output GELU(x):                 Derivative GELU'(x):

  3.0 ┤               ╱          1.1 ┤           ╭──────
  2.0 ┤              ╱           1.0 ┤        ╭──╯
  1.0 ┤           ╭─╱            0.5 ┤   ╭───╯
  0.0 ┤──────────╱               0.0 ┤───╯
 -0.1 ┤       ╰──╯                   └────────────────────
      └──────────────── x                  -3   0   +3
            -3   0   +3
```

**Numerical values:**

```
  x      GELU(x)   GELU'(x)
  ──────────────────────────
  -3     -0.004    0.020
  -2     -0.045    0.085
  -1     -0.159    0.317
  -0.75  -0.170    0.254   ← minimum output
   0      0.000    0.500
  +1      0.841    1.083
  +2      1.954    1.086
  +3      2.996    1.013
```

**GELU vs ReLU — the key difference:**

```
  Input x = -0.5:
    ReLU(-0.5)  = 0.000   ← hard clipped to zero
    GELU(-0.5)  = -0.154  ← small negative, not fully suppressed

  Input x = 0.5:
    ReLU(0.5)   = 0.500   ← passes through exactly
    GELU(0.5)   = 0.346   ← slightly suppressed (38% chance of being negative)

  Input x = 2.0:
    ReLU(2.0)   = 2.000   ← passes through exactly
    GELU(2.0)   = 1.954   ← passes through almost fully (95% chance positive)

  ReLU:  binary gate — 0 or identity
  GELU:  soft probabilistic gate — suppresses proportionally to negativity
```

**Why this matters for transformers:**

```
  Transformers process language where subtle numerical differences
  between features carry meaning. Hard-clipping small activations
  to zero (ReLU) loses this information.

  GELU preserves small signals through soft suppression — better
  for the nuanced feature interactions in attention-based models.
```

**Ablation (BERT pre-training):**

```
  Activation   │  GLUE Score  │  Training stability
  ─────────────┼──────────────┼─────────────────────
  ReLU         │  82.1        │  occasional instability
  ELU          │  82.3        │  stable
  GELU         │  82.8  ←     │  stable

Used in: BERT, RoBERTa, GPT-2, GPT-3, ViT, T5, CLIP
— the transformer standard from 2018 to 2022.
```

---

### 5. Swish (SiLU)

**Introduced:** Ramachandran et al., 2017 (discovered via neural architecture search)  
Also called **SiLU** (Sigmoid Linear Unit) — same function, different name.

```
Formula:
  Swish(x) = x · σ(x)
           = x / (1 + e⁻ˣ)

  SiLU(x)  = x · sigmoid(x)   ← identical to Swish

Output range: approximately (-0.28, ∞)
  Minimum at x ≈ -1.28:  Swish(-1.28) ≈ -0.28

Derivative:
  Swish'(x) = σ(x) + x · σ(x) · (1 - σ(x))
             = σ(x) · (1 + x · (1 - σ(x)))
             = Swish(x) + σ(x) · (1 - Swish(x))
```

**Output curve and derivative:**

```
  Output Swish(x):                Derivative Swish'(x):

  3.0 ┤               ╱          1.1 ┤           ╭──────
  2.0 ┤              ╱           1.0 ┤        ╭──╯
  1.0 ┤           ╭─╱            0.5 ┤  ╭────╯
  0.0 ┤──────────╱               0.0 ┤──╯
 -0.2 ┤      ╰───╯              -0.1 ┤
      └──────────────── x             └──────────────────
            -3   0   +3                    -3   0   +3
```

**Numerical values:**

```
  x       Swish(x)   σ(x)    x·σ(x)
  ────────────────────────────────────
  -3      -0.142     0.047   ← nearly zero, slight suppression
  -2      -0.238     0.119
  -1.28   -0.278     0.217   ← minimum of Swish
  -1      -0.269     0.269
   0       0.000     0.500
  +1       0.731     0.731
  +2       1.762     0.881
  +3       2.858     0.953
```

**Swish vs ReLU vs GELU:**

```
  x = -1.0:
    ReLU:   0.000   ← hard zero
    GELU:  -0.159   ← slight suppression
    Swish: -0.269   ← stronger suppression (self-gated by sigmoid)

  x = 0.0:
    ReLU:   0.000
    GELU:   0.000
    Swish:  0.000   ← all zero at origin

  x = 1.0:
    ReLU:   1.000
    GELU:   0.841
    Swish:  0.731   ← most suppressive of the three for small positives

  x = 3.0:
    ReLU:   3.000
    GELU:   2.996
    Swish:  2.858   ← all nearly identical for large positives
```

**The self-gating intuition:**

```
  Swish(x) = x · σ(x)
              ↑   ↑
           value gate

  The input x gates itself: σ(x) ∈ (0,1) acts as a smooth
  learnable gate determined by the value itself.
  Positive values open the gate; negative values suppress it.

  Unlike ReLU which uses a fixed threshold (0),
  Swish uses the value's own sign as a soft continuous gate.
```

**Use today:** Core component of SwiGLU. Used standalone in EfficientNet and MobileNetV3.

---

### 6. SwiGLU

**Introduced:** Noam Shazeer, 2020 ("GLU Variants Improve Transformer")

SwiGLU is not a simple activation — it is a **gated FFN variant** that uses Swish as its gating mechanism. Understanding Swish first makes SwiGLU clear.

```
Formula:
  SwiGLU(x, Wg, Wv) = Swish(x · Wg) ⊙ (x · Wv)

  ⊙ = element-wise multiplication

Expanded into the full FFN:
  gate  = Swish(x · Wg + bg)   ← gate branch: what to suppress
  value = x · Wv + bv          ← value branch: the raw features
  h     = gate ⊙ value         ← element-wise: gate controls flow
  out   = h · Wd + bd          ← down-projection back to d
```

**Gating — what it means:**

```
  Standard FFN (GELU):               SwiGLU FFN:

  x ──→ Linear(d, 4d) ──→ GELU      x ──→ Linear(d, 8d/3) → Swish → gate
        ──→ Linear(4d, d)             x ──→ Linear(d, 8d/3) ──────── value
                                            gate ⊙ value
                                      ──→ Linear(8d/3, d)

  GELU: single projection; fixed nonlinearity decides suppression.

  SwiGLU: TWO projections; one becomes the gate, the other the value.
          The gate is input-dependent — different inputs generate
          different gates. Richer than a fixed nonlinearity.
```

**Concrete example:**

```
  Suppose d=4, hidden=4 (toy example):

  Input x = [1.0, -0.5, 2.0, 0.3]

  After gate projection + Swish:
    gate = Swish([0.8, -0.2, 1.5, 0.1]) = [0.66, -0.10, 1.24, 0.05]

  After value projection:
    value = [0.5, 1.2, -0.3, 0.9]

  Output = gate ⊙ value:
    h = [0.66·0.5, (-0.10)·1.2, 1.24·(-0.3), 0.05·0.9]
      = [0.33, -0.12, -0.37, 0.045]

  The gate selectively amplified feature 0 and feature 2,
  and nearly zeroed feature 3 — a dynamic, input-dependent filter.
```

**Why 8d/3 instead of 4d:**

```
  Standard FFN params (d=768, expansion=4):
    Linear(768, 3072) + Linear(3072, 768) = 2 × 768 × 3072 = 4,718,592

  SwiGLU FFN params (d=768, hidden=2048):
    Linear(768, 2048) × 2 (gate + value)
    + Linear(2048, 768)
    = 2×768×2048 + 2048×768 = 3×768×2048 = 4,718,592

  8d/3 ≈ 2048 for d=768. Same total parameters, three matrices.
```

**Ablation (PaLM paper):**

```
  Activation    │  Perplexity (↓ better)
  ──────────────┼────────────────────────
  ReLU          │  8.4
  GELU          │  8.1
  SwiGLU        │  7.6  ← best

Used in: LLaMA 2, LLaMA 3, PaLM, Gemma, Mistral, Qwen, DeepSeek
— the default FFN activation for all modern open-source LLMs.
```

---

### All Five Functions Side by Side

```
  Input:  x = [-3, -1, 0, 1, 3]

  ┌─────────┬────────┬────────┬────────┬────────┬────────┐
  │  x      │ Sigmoid│  Tanh  │  ReLU  │  GELU  │  Swish │
  ├─────────┼────────┼────────┼────────┼────────┼────────┤
  │  -3.0   │  0.047 │ -0.995 │  0.000 │ -0.004 │ -0.142 │
  │  -1.0   │  0.269 │ -0.762 │  0.000 │ -0.159 │ -0.269 │
  │   0.0   │  0.500 │  0.000 │  0.000 │  0.000 │  0.000 │
  │  +1.0   │  0.731 │  0.762 │  1.000 │  0.841 │  0.731 │
  │  +3.0   │  0.953 │  0.995 │  3.000 │  2.996 │  2.858 │
  └─────────┴────────┴────────┴────────┴────────┴────────┘

  Key observations:
  - Sigmoid and Tanh: bounded outputs, saturate at extremes
  - ReLU: exact zeros for negatives, identity for positives
  - GELU and Swish: smooth, small negative dip, near-identity for large positives
  - Swish is more suppressive than GELU for moderate negatives (-1 to 0)
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
