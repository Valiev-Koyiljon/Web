# DeiT — Data-Efficient Image Transformers & Distillation Through Attention

**Paper:** [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)  
**Authors:** Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou (Facebook AI Research)  
**Venue:** ICML 2021  
**Tutorial by:** Koyilbek Valiev

---

## The Problem DeiT Solves

ViT showed Transformers can work for vision, but with a catch:

```
ViT's dirty secret:

  ViT trained on ImageNet-1K (1.3M images)   →  77.9% top-1  (WORSE than ResNet)
  ViT trained on JFT-300M (300M images)       →  88.5% top-1  (SOTA)

Without hundreds of millions of images, ViT fails.
Most researchers don't have access to JFT-300M.
```

**DeiT's question:** Can we train ViT to match or beat CNNs using ONLY ImageNet-1K?

**DeiT's answer:** Yes — with the right training recipe + knowledge distillation.

```
DeiT-B trained on ImageNet-1K ONLY  →  83.4% top-1  🏆
  vs. ViT-B on ImageNet-1K          →  77.9% top-1
  vs. EfficientNet-B7               →  84.3% top-1

DeiT closes the gap WITHOUT needing extra data.
```

---

## Architecture Overview

DeiT uses the **exact same architecture as ViT**, with one addition: a **distillation token**.

```
┌──────────────────────────────────────────────────────────────┐
│                      INPUT IMAGE                             │
│                    (224 × 224 × 3)                           │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    PATCH EMBEDDING                            │
│         Split into 16×16 patches → 196 patch tokens          │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│     PREPEND [CLS] TOKEN + [DIST] TOKEN + POSITION EMB       │
│                                                              │
│  ┌─────┬──────┬────┬────┬────┬─────┬────┐                   │
│  │[CLS]│[DIST]│ P1 │ P2 │ P3 │ ... │P196│   198 tokens     │
│  └─────┴──────┴────┴────┴────┴─────┴────┘                   │
│     +     +     +    +    +    +     +                       │
│  ┌─────┬──────┬────┬────┬────┬─────┬────┐                   │
│  │ E0  │ E1   │ E2 │ E3 │ E4 │ ... │E197│   Position embeds │
│  └─────┴──────┴────┴────┴────┴─────┴────┘                   │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              TRANSFORMER ENCODER (×12 layers)                │
│                                                              │
│     ┌──────────────────────────────────────────┐             │
│     │  LayerNorm → MHSA → Residual            │             │
│     │  LayerNorm → MLP  → Residual            │             │
│     └──────────────────────────────────────────┘ × 12        │
│                                                              │
│  All 198 tokens attend to each other (full self-attention)   │
└───────────────────────────┬──────────────────────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                    ▼               ▼
┌──────────────────────┐  ┌──────────────────────┐
│    [CLS] OUTPUT      │  │   [DIST] OUTPUT      │
│         │            │  │         │            │
│    Classification    │  │    Distillation      │
│    Head (MLP)        │  │    Head (MLP)        │
│         │            │  │         │            │
│   Class logits       │  │   Teacher logits     │
└──────────────────────┘  └──────────────────────┘
            │                       │
            └───────┬───────────────┘
                    │
          Average at inference
                    │
              Final prediction
```

### The Key Difference: [DIST] Token

ViT has one special token: `[CLS]`. DeiT adds a second: `[DIST]` (distillation token).

```
ViT:   [CLS] [P1] [P2] ... [P196]  →  [CLS] output → classification
DeiT:  [CLS] [DIST] [P1] [P2] ... [P196]  →  [CLS] → classification
                                               [DIST] → matches teacher
```

- **[CLS]** learns to predict the true label (standard cross-entropy)
- **[DIST]** learns to match the CNN teacher's predictions (distillation loss)
- Both tokens attend to all patches and to each other through self-attention
- At inference: average the two heads

---

## Knowledge Distillation — How It Works

### The Idea

A large, pre-trained CNN (the "teacher") has already learned good visual features. We can transfer this knowledge to the Transformer (the "student") by making the student mimic the teacher's outputs.

```
┌─────────────────┐         ┌─────────────────┐
│  CNN TEACHER     │         │  DeiT STUDENT   │
│  (RegNet-Y-16GF)│         │  (ViT-B/16)     │
│                  │         │                  │
│  Input image ────┼────────▶│  Input image     │
│        │         │         │        │         │
│        ▼         │         │   ┌────┴────┐    │
│  Teacher logits  │         │ [CLS]    [DIST]  │
│   (frozen)  ─────┼────┐    │   │        │     │
│                  │    │    │   ▼        ▼     │
└─────────────────┘    │    │  L_CE    L_dist  │
                       │    │   │        │     │
                       │    └───┼────────┤     │
                       │        │        │     │
                       └────────┼───▶ Match!   │
                                │              │
                            L_total = L_CE + L_dist
```

### Hard Distillation vs. Soft Distillation

**Soft distillation** (traditional): Student matches the teacher's full probability distribution.

```
L_soft = (1-λ) · CE(student, true_label) + λ · KL(student_soft, teacher_soft)

Where soft = softmax(logits / temperature)
```

**Hard distillation** (DeiT's finding): Student simply matches the teacher's top prediction.

```
L_hard = 0.5 · CE(cls_output, true_label) + 0.5 · CE(dist_output, teacher_argmax)

teacher_argmax = argmax(teacher_logits)   ← hard label, not distribution
```

**Surprising result:** Hard distillation works **better** than soft distillation for DeiT.

```
Distillation Results (DeiT-B, ImageNet top-1):

  No distillation          →  81.8%
  Soft distillation        →  83.1%
  Hard distillation        →  83.4%  🏆

Hard labels from the teacher provide a stronger learning signal.
```

---

## The Training Recipe — Why DeiT Works

DeiT's success isn't just distillation. It's a carefully tuned **training recipe** that regularizes the Transformer heavily:

```
┌──────────────────────────────────────────────────────────┐
│                  DeiT TRAINING RECIPE                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  DATA AUGMENTATION:                                      │
│  ├── RandAugment (N=2, M=9)                             │
│  ├── Mixup (α=0.8)                                      │
│  ├── CutMix (α=1.0)                                     │
│  ├── Random Erasing (p=0.25)                            │
│  └── Repeated Augmentation (3 repeats)                  │
│                                                          │
│  REGULARIZATION:                                         │
│  ├── Stochastic Depth (drop rate=0.1)                   │
│  ├── Label Smoothing (ε=0.1)                            │
│  └── Weight Decay (0.05)                                │
│                                                          │
│  OPTIMIZER:                                              │
│  ├── AdamW (β₁=0.9, β₂=0.999)                         │
│  ├── Learning Rate: 5e-4 × batch_size/512              │
│  ├── Cosine LR Schedule                                 │
│  ├── 5 epochs warmup                                    │
│  └── 300 total epochs                                   │
│                                                          │
│  HARDWARE:                                               │
│  ├── 4× NVIDIA V100 GPUs (32GB)                        │
│  ├── Batch size: 1024                                   │
│  └── Training time: ~53 hours for DeiT-B               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Why Each Augmentation Matters

```
Ablation — removing one augmentation at a time (DeiT-B):

  Full recipe                        →  81.8%
  Remove RandAugment                 →  80.3%  (−1.5%)  ⚠️ biggest drop
  Remove Mixup                       →  81.2%  (−0.6%)
  Remove CutMix                      →  81.0%  (−0.8%)
  Remove Random Erasing              →  81.4%  (−0.4%)
  Remove Stochastic Depth            →  81.0%  (−0.8%)
  Remove Repeated Augmentation       →  81.3%  (−0.5%)

Every component contributes. RandAugment is the most critical.
Transformers overfit much more easily than CNNs on small datasets.
```

---

## PyTorch Implementation

### DeiT Model with Distillation Token

```python
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)             # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth — randomly drops entire residual branches during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class DeiT(nn.Module):
    """
    Data-efficient Image Transformer with distillation token.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # ← DeiT's addition
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, embed_dim))  # +2 for cls & dist

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Two separate heads
        self.head = nn.Linear(embed_dim, num_classes)       # classification head
        self.head_dist = nn.Linear(embed_dim, num_classes)  # distillation head

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                              # (B, 196, 768)

        cls_tokens = self.cls_token.expand(B, -1, -1)        # (B, 1, 768)
        dist_tokens = self.dist_token.expand(B, -1, -1)      # (B, 1, 768)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)   # (B, 198, 768)
        x = x + self.pos_embed                                # (B, 198, 768)

        x = self.blocks(x)                                    # (B, 198, 768)
        x = self.norm(x)

        cls_out = x[:, 0]    # [CLS] token output
        dist_out = x[:, 1]   # [DIST] token output

        cls_logits = self.head(cls_out)          # (B, num_classes)
        dist_logits = self.head_dist(dist_out)   # (B, num_classes)

        if self.training:
            return cls_logits, dist_logits  # both needed for loss
        else:
            return (cls_logits + dist_logits) / 2  # average at inference


# Create DeiT-Base
model = DeiT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
total = sum(p.numel() for p in model.parameters())
print(f"DeiT-Base: {total / 1e6:.1f}M parameters")
# Output: DeiT-Base: 87.3M parameters (slightly more than ViT-B due to dist token + head)
```

### Training Loop with Hard Distillation

```python
import torch.nn.functional as F

def train_one_step(student, teacher, images, labels, optimizer):
    """
    Hard distillation training step.
    teacher: frozen CNN (e.g., RegNet-Y-16GF)
    student: DeiT model
    """
    # Get teacher predictions (no gradient)
    with torch.no_grad():
        teacher_logits = teacher(images)
        teacher_labels = teacher_logits.argmax(dim=-1)  # hard labels

    # Student forward
    cls_logits, dist_logits = student(images)

    # Loss = 0.5 * CE(cls, true) + 0.5 * CE(dist, teacher_hard)
    loss_cls = F.cross_entropy(cls_logits, labels)
    loss_dist = F.cross_entropy(dist_logits, teacher_labels)
    loss = 0.5 * loss_cls + 0.5 * loss_dist

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

---

## Model Variants

```
┌───────────┬────────┬───────────┬───────┬────────┬─────────────────────┐
│  Model    │ Layers │ Embed Dim │ Heads │ Params │ ImageNet Top-1      │
├───────────┼────────┼───────────┼───────┼────────┼─────────────────────┤
│ DeiT-Ti   │   12   │    192    │   3   │   5M   │ 72.2%              │
│ DeiT-S    │   12   │    384    │   6   │  22M   │ 79.8%              │
│ DeiT-B    │   12   │    768    │  12   │  87M   │ 81.8%              │
│ DeiT-B ⚗  │   12   │    768    │  12   │  87M   │ 83.4% (distilled)  │
│ DeiT-B ↑384│  12   │    768    │  12   │  87M   │ 85.2% (fine-tuned) │
└───────────┴────────┴───────────┴───────┴────────┴─────────────────────┘

⚗ = with distillation from RegNet teacher
↑384 = fine-tuned at 384×384 resolution
```

---

## Results Comparison

```
Method              │ Extra Data │ Params │ Top-1 │ Top-5
────────────────────┼────────────┼────────┼───────┼──────
ResNet-50           │     No     │  25M   │ 76.2% │ 93.0%
ResNet-152          │     No     │  60M   │ 78.3% │ 94.2%
EfficientNet-B3     │     No     │  12M   │ 81.6% │ 95.7%
EfficientNet-B7     │     No     │  66M   │ 84.3% │ 97.0%
────────────────────┼────────────┼────────┼───────┼──────
ViT-B/16            │     No     │  86M   │ 77.9% │ 93.9%  ← struggles!
ViT-B/16            │  JFT-300M  │  86M   │ 84.2% │ 97.2%  ← needs data
ViT-L/16            │  JFT-300M  │ 307M   │ 87.8% │ 98.1%
────────────────────┼────────────┼────────┼───────┼──────
DeiT-B              │     No     │  87M   │ 81.8% │ 95.6%  ← without distill
DeiT-B ⚗            │     No     │  87M   │ 83.4% │ 96.5%  ← with distill 🏆
DeiT-B ⚗ ↑384       │     No     │  87M   │ 85.2% │ 97.2%  ← fine-tuned 🏆🏆

DeiT matches ViT-B trained on JFT-300M, using only ImageNet-1K.
```

---

## Key Insights

### 1. Why a CNN Teacher?

```
Teacher type experiment (DeiT-B distillation):

  Teacher = DeiT-B (Transformer)    → 82.6% student
  Teacher = RegNet-Y-16GF (CNN)     → 83.4% student  🏆

CNN teacher > Transformer teacher!

Why? The CNN has different inductive biases (locality, translation
equivariance). Distillation transfers these complementary biases
to the Transformer, giving it the best of both worlds.
```

### 2. [CLS] and [DIST] Learn Different Things

```
Cosine similarity between [CLS] and [DIST] representations:

  At initialization:  ~0.06 (random, uncorrelated)
  After training:     ~0.93 (highly correlated but NOT identical)

Despite high correlation, they specialize:
  [CLS]  → better at fine-grained features (from true labels)
  [DIST] → better at structural features (from CNN teacher)

Averaging both gives better results than either alone:
  [CLS] only   → 82.0%
  [DIST] only  → 82.7%
  Average      → 83.4%  🏆
```

### 3. Transformers Need More Regularization Than CNNs

```
The same augmentation recipe applied to ResNet-50 vs DeiT-S:

  ResNet-50 + full recipe  → 78.5% (+2.3% improvement)
  DeiT-S + full recipe     → 79.8% (+6.3% improvement)

Transformers benefit MORE from regularization because they
have no built-in inductive bias — they can memorize training
data more easily, so they need stronger regularization to
generalize.
```

### 4. Fine-tuning at Higher Resolution

```
DeiT-B training:  224×224  →  81.8%
DeiT-B fine-tune: 384×384  →  85.2%  (+3.4%!)

How it works:
1. Train at 224×224 for 300 epochs
2. Interpolate position embeddings to 384×384
3. Fine-tune for ~30 epochs at higher resolution

The model sees more detail per patch at higher resolution,
significantly boosting performance with minimal extra compute.
```

---

## DeiT vs ViT — Summary

```
┌────────────────────┬──────────────────┬──────────────────┐
│                    │       ViT        │       DeiT       │
├────────────────────┼──────────────────┼──────────────────┤
│ Architecture       │ Standard ViT     │ ViT + [DIST]     │
│ Training data      │ JFT-300M needed  │ ImageNet-1K only │
│ Distillation       │ None             │ Hard distillation│
│ Teacher            │ N/A              │ CNN (RegNet)     │
│ Augmentation       │ Light            │ Heavy            │
│ Regularization     │ Minimal          │ Stochastic Depth │
│ Training epochs    │ ~90              │ 300              │
│ Key insight        │ Scale is all     │ Recipe matters   │
│                    │ you need         │ more than scale  │
└────────────────────┴──────────────────┴──────────────────┘
```

---

## What Came After DeiT

```
2020  ViT          — Transformers work for vision (with lots of data)
2021  DeiT         — Make ViT data-efficient via distillation 🏆
2021  DeiT III     — Improved training recipe (3 augmentations only)
2021  Swin         — Hierarchical windows for efficiency
2021  CaiT         — Class-Attention in Image Transformers
2022  BEiT         — BERT-style pre-training for ViT
2022  MAE          — Masked Autoencoders (75% masking!)
2023  DINOv2       — Self-supervised ViT at scale
```

---

## Citation

```bibtex
@inproceedings{touvron2021training,
  title={Training data-efficient image transformers 
         \& distillation through attention},
  author={Touvron, Hugo and Cord, Matthieu and 
          Douze, Matthijs and Massa, Francisco and 
          Sablayrolles, Alexandre and J{\'e}gou, Herv{\'e}},
  booktitle={International Conference on Machine Learning},
  pages={10347--10357},
  year={2021},
  organization={PMLR}
}
```

---

*Tutorial by Koyilbek Valiev — AI / ML Engineer | Research Engineer*
