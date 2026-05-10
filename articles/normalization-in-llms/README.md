# Normalization in Modern LLMs: LayerNorm, RMSNorm, and Beyond

**By:** Koyilbek Valiev  
**Topics:** Deep Learning · Transformers · LLMs · Math

---

## The Problem Normalization Solves

Training deep neural networks is unstable. As gradients flow backward through dozens of layers, activations can explode to enormous values or shrink to near-zero. Both kill training.

Normalization layers fix this by keeping activations in a well-behaved range — regardless of what happened in the layers before.

Before understanding normalization itself, we need one foundational concept: **variance**.

---

## Part 1 — Variance: The Foundation

**Variance** measures how spread out a set of numbers is from their mean.

Given values x₁, x₂, ..., xₙ:

```
Step 1: Compute the mean
        μ = (1/n) · (x₁ + x₂ + ... + xₙ)

Step 2: Compute variance
        σ² = (1/n) · Σ (xᵢ - μ)²

Step 3: Standard deviation (same unit as the data)
        σ  = √σ²
```

### Concrete Example

```
Values: [2, 4, 6, 8]

Mean:       μ  = (2 + 4 + 6 + 8) / 4 = 5

Deviations: [2-5, 4-5, 6-5, 8-5] = [-3, -1, +1, +3]

Squared:    [9, 1, 1, 9]

Variance:   σ² = (9 + 1 + 1 + 9) / 4 = 5.0

Std Dev:    σ  = √5.0 ≈ 2.24
```

### Why do we square the deviations?

- Raw deviations always sum to zero (positives and negatives cancel out)
- Squaring removes the sign and **penalizes large deviations more heavily**
- Taking the square root at the end (std dev) brings us back to the original units

### Variance vs Standard Deviation

```
┌─────────────────┬────────────────────────────┬─────────────────────┐
│                 │ Formula                    │ Units               │
├─────────────────┼────────────────────────────┼─────────────────────┤
│ Variance        │ σ² = mean((xᵢ - μ)²)      │ units² (e.g. cm²)  │
│ Std Deviation   │ σ  = √variance             │ same as data (cm)  │
└─────────────────┴────────────────────────────┴─────────────────────┘
```

Standard deviation is just the square root of variance. It is more interpretable because it shares the unit of the original data — but under the hood, normalization uses the variance.

---

## Part 2 — Layer Normalization

LayerNorm, introduced by Ba et al. (2016), normalizes across the **feature dimension** of a single sample. This is the default normalization in nearly every transformer built between 2017 and 2023.

### The Formula

```
Given: a vector x = [x₁, x₂, ..., xᵈ]  (one token's activations, d dimensions)

Step 1 — Mean across features:
         μ = (1/d) · Σ xᵢ

Step 2 — Variance across features:
         σ² = (1/d) · Σ (xᵢ - μ)²

Step 3 — Normalize:
         x̂ᵢ = (xᵢ - μ) / √(σ² + ε)

Step 4 — Scale and shift (learned parameters γ, β):
         yᵢ = γ · x̂ᵢ + β
```

`ε` (epsilon) is a tiny constant (typically 1e-5) to prevent division by zero when variance is near zero.

`γ` and `β` are **learned per-dimension** — the network can undo the normalization if it needs to.

### Visualizing What Normalization Does

```
Before LayerNorm:                After LayerNorm:
                                 
  activation values              activation values
  ┌─────────────────┐            ┌─────────────────┐
  │ ●               │            │      ●  ●       │
  │       ●         │            │   ●     ●  ●    │
  │                 │            │ ●          ●    │
  │ 0.001  to  500  │            │  -2.0  to  2.0  │
  └─────────────────┘            └─────────────────┘
  
  Spread: uncontrolled           Mean ≈ 0, Std ≈ 1
```

### LayerNorm vs BatchNorm

The key alternative before LayerNorm was Batch Normalization. They normalize over different dimensions:

```
┌───────────────┬────────────────────────────┬────────────────────────────┐
│               │ BatchNorm                  │ LayerNorm                  │
├───────────────┼────────────────────────────┼────────────────────────────┤
│ Normalizes    │ across the batch (B)       │ across features (d)        │
│ over          │ for each feature           │ for each sample            │
├───────────────┼────────────────────────────┼────────────────────────────┤
│ Batch size    │ depends on it              │ completely independent      │
│ dependency    │ (breaks at batch size=1)   │ (works with any size)      │
├───────────────┼────────────────────────────┼────────────────────────────┤
│ At inference  │ needs stored running stats │ computes on the fly        │
├───────────────┼────────────────────────────┼────────────────────────────┤
│ Best for      │ CNNs, image models         │ Transformers, LLMs, RNNs   │
└───────────────┴────────────────────────────┴────────────────────────────┘
```

LayerNorm dominated transformers because language models often process sequences of varying length and batch size — BatchNorm breaks in those conditions.

### Where LayerNorm Lives in a Transformer

```
Input x
   │
   ├─── LayerNorm ──→ Multi-Head Self-Attention ──→ + ──→ (residual)
   │                                                │
   │◄───────────────────────────────────────────────┘
   │
   ├─── LayerNorm ──→ Feed-Forward Network ──────→ + ──→ (residual)
   │                                                │
   │◄───────────────────────────────────────────────┘
   │
Output
```

This is called **Pre-Norm** — applying normalization *before* the sublayer. Modern LLMs all use this. (The original "Attention Is All You Need" paper used Post-Norm, which is harder to train deep.)

---

## Part 3 — Alternatives to LayerNorm in Modern LLMs

LayerNorm works well, but researchers have found faster, simpler, or more stable alternatives. Here is the complete landscape.

---

### 1. RMSNorm (Root Mean Square Normalization)

**Used in:** LLaMA 2, LLaMA 3, Mistral, Gemma, Qwen, DeepSeek, Falcon

RMSNorm is the most widely adopted LayerNorm replacement today. The key insight: **mean-centering is unnecessary**. Only the scale matters.

```
Step 1 — Root Mean Square:
         RMS(x) = √( (1/d) · Σ xᵢ² + ε )
                  (no mean subtraction — just average of squares)

Step 2 — Normalize:
         x̂ᵢ = xᵢ / RMS(x)

Step 3 — Scale with learned γ:
         yᵢ = γ · x̂ᵢ
               (no additive β — also removed)
```

**LayerNorm vs RMSNorm side by side:**

```
LayerNorm:   y = γ · (x - μ) / √(σ² + ε)  +  β
                        ↑                      ↑
                  mean subtraction         bias shift
                  (2 extra ops)          (1 extra param)

RMSNorm:     y = γ · x / √( mean(x²) + ε )
                  (no mean, no bias — cleaner and faster)
```

**Why it works:** Empirically, the mean subtraction in LayerNorm contributes almost nothing to training quality. Removing it saves ~15% compute with no measurable loss in model quality.

---

### 2. Dynamic Tanh (DyT)

**Paper:** "Transformers without Normalization" (Zhu et al., 2025)  
**Used in:** Research stage — demonstrated parity with LayerNorm

The most radical alternative: **no normalization statistics at all**. Replace the entire norm layer with a learned element-wise tanh:

```
DyT(x) = γ ⊙ tanh(α · x) + β

where:
  α   = a learned scalar (one per layer, initialized to 0.5)
  γ   = learned per-dimension scale (same as LayerNorm's γ)
  β   = learned per-dimension bias (same as LayerNorm's β)
  ⊙   = element-wise multiplication
```

**Why tanh works like normalization:**

```
tanh input-output curve:
                         1.0 ─────────────────────────
                        /
          0.5 ────────/
                     /
          0.0 ──────/──────────────── (input)
                   /
        -0.5 ────/
                /
        -1.0 ──────────────────────── 
        
  Large activations (±10, ±100) all get compressed to (-1, 1).
  This is exactly what normalization achieves, but via a 
  smooth learned function instead of statistics.
```

**Key advantage:** No mean/variance computation — zero statistical overhead. Just a multiply and a tanh.

---

### 3. DeepNorm

**Paper:** "DeepNet: Scaling Transformers to 1,000 Layers" (Wang et al., 2022, Microsoft)  
**Used in:** Research; enables ultra-deep transformers

DeepNorm does not replace LayerNorm — it changes **how the residual connection is scaled** around it:

```
Standard Pre-Norm residual:
    output = x + Sublayer(LayerNorm(x))

DeepNorm residual:
    output = LayerNorm(α · x + Sublayer(x))
                       ↑
              scale the skip connection
              (α > 1, e.g. α = 2.0 for 1000-layer model)
```

Combined with a **very small initialization** of weights (scaled by β, where β < 1), this keeps gradients stable across 1000+ layers.

```
┌────────────────────────────────────────────────────────┐
│  Standard Transformer: starts failing around 20 layers  │
│  Pre-Norm Transformer: works up to ~100 layers          │
│  DeepNorm Transformer: stable at 1000 layers            │
└────────────────────────────────────────────────────────┘
```

---

### 4. QKNorm

**Used in:** Mistral, some diffusion models (SDXL, Stable Diffusion 3)

QKNorm doesn't replace LayerNorm — it adds **extra normalization inside the attention mechanism** to prevent attention score explosion in long contexts.

```
Standard attention:
    scores = Q · Kᵀ / √dₖ

QKNorm attention:
    Q̂ = Q / ‖Q‖₂    (L2-normalize each query vector)
    K̂ = K / ‖K‖₂    (L2-normalize each key vector)
    scores = Q̂ · K̂ᵀ · s    (s = learned temperature scalar)
```

**Why it matters:** As sequence length grows (8K, 32K, 128K tokens), dot products between Q and K can become extremely large, causing softmax to saturate and attention to collapse to near-one-hot distributions. QKNorm prevents this.

```
Without QKNorm (long context):     With QKNorm:
  Q·Kᵀ values: [-300, +850]         Q̂·K̂ᵀ values: [-1.0, +1.0]
  softmax: ≈ [0.00, 1.00]           softmax: more uniform distribution
  → attention collapses             → attention stays healthy
```

---

### 5. Pre-Norm vs Post-Norm (Placement Matters)

The *formula* of the norm is only half the story. **Where** you place it has a large effect on training stability:

```
Post-Norm (original "Attention is All You Need"):
    output = LayerNorm( x + Sublayer(x) )
    
    ┌──────────────────────────────────────────────┐
    │ x ──→ Sublayer ──→ + ──→ LayerNorm ──→ out  │
    │        ↑____________↑                        │
    └──────────────────────────────────────────────┘
    
    Problem: gradients must pass through LayerNorm at every
    layer during backprop — can vanish in deep models.

Pre-Norm (modern standard):
    output = x + Sublayer( LayerNorm(x) )
    
    ┌──────────────────────────────────────────────┐
    │ x ──→ LayerNorm ──→ Sublayer ──→ + ──→ out  │
    │ ↑_________________________________↑          │
    └──────────────────────────────────────────────┘
    
    Benefit: the residual path x is unobstructed — gradients
    flow directly without passing through LayerNorm.
    Much more stable for deep (24–96+ layer) models.
```

**All modern LLMs use Pre-Norm.** Post-Norm is only seen in shallow models or older architectures like the original BERT.

---

## Part 4 — Complete Comparison

```
┌─────────────┬──────────────────────────────┬──────────┬────────────────────────────┐
│ Method      │ Formula (simplified)         │ Speed    │ Used In                    │
├─────────────┼──────────────────────────────┼──────────┼────────────────────────────┤
│ LayerNorm   │ (x - μ) / σ · γ + β         │ baseline │ BERT, GPT-2, original ViT  │
│ RMSNorm     │ x / RMS(x) · γ              │ ~15% faster│ LLaMA 2/3, Mistral, Gemma│
│ DyT         │ γ · tanh(α·x) + β           │ fastest  │ Research (2025)            │
│ DeepNorm    │ LN(α·x + Sublayer(x))       │ same     │ Deep research models       │
│ QKNorm      │ L2-norm on Q and K          │ minimal  │ Mistral, long-context LLMs │
└─────────────┴──────────────────────────────┴──────────┴────────────────────────────┘
```

**The current industry default:** Pre-Norm + RMSNorm. That is what LLaMA 3, Mistral 7B, Gemma, Qwen 2.5, and DeepSeek all use.

---

## Key Takeaways

- **Variance** measures spread — normalization divides by standard deviation to enforce unit spread
- **LayerNorm** normalizes per-sample across features; works regardless of batch size, making it ideal for transformers
- **RMSNorm** drops mean subtraction — ~15% faster, no quality loss, now the dominant choice
- **DyT** removes all statistics and uses a learned tanh — promising but still research-stage
- **DeepNorm** keeps LayerNorm but modifies residual scaling to enable 1000-layer models
- **QKNorm** normalizes Q and K vectors in attention to prevent score explosion in long contexts
- **Pre-Norm placement** (norm before sublayer) is critical for training stability in deep models

---

*If you found this useful, reach out on [LinkedIn](https://www.linkedin.com/in/koyiljonvaliev2003/) or [GitHub](https://github.com/Valiev-Koyiljon).*
