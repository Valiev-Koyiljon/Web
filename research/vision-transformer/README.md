# Vision Transformer (ViT) — Deep Dive Tutorial

**Paper:** [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  
**Authors:** Dosovitskiy, Beyer, Kolesnikov et al. (Google Brain)  
**Venue:** ICLR 2021  
**Tutorial by:** Koyilbek Valiev

---

## Why This Paper Matters

Before ViT, computer vision was dominated by CNNs (ResNet, EfficientNet). The core question ViT asks:

> *Can we apply a standard Transformer — the architecture that revolutionized NLP — directly to images, with minimal modifications?*

The answer is **yes**, and it changed everything. ViT proved that with enough data, Transformers match or beat CNNs in vision, opening the door to unified multimodal architectures (CLIP, DALL-E, GPT-4V).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT IMAGE                          │
│                  (224 × 224 × 3)                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               PATCH EMBEDDING                           │
│                                                         │
│   Split image into 16×16 patches → 196 patches          │
│   Each patch: 16 × 16 × 3 = 768 pixels                 │
│   Linear projection: 768 → D (embedding dim)           │
│                                                         │
│   ┌────┬────┬────┬────┬─────┬────┐                     │
│   │ P1 │ P2 │ P3 │ P4 │ ... │P196│   196 patch tokens  │
│   └────┴────┴────┴────┴─────┴────┘                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│          PREPEND [CLS] TOKEN + POSITION EMB             │
│                                                         │
│   ┌─────┬────┬────┬────┬─────┬────┐                    │
│   │[CLS]│ P1 │ P2 │ P3 │ ... │P196│   197 tokens       │
│   └─────┴────┴────┴────┴─────┴────┘                    │
│      +     +    +    +    +     +                       │
│   ┌─────┬────┬────┬────┬─────┬────┐                    │
│   │ E0  │ E1 │ E2 │ E3 │ ... │E196│   Position embeds  │
│   └─────┴────┴────┴────┴─────┴────┘                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│            TRANSFORMER ENCODER (×L layers)              │
│                                                         │
│   ┌───────────────────────────────────────────┐         │
│   │  Layer Norm                               │         │
│   │       ↓                                   │         │
│   │  Multi-Head Self-Attention (MHSA)         │         │
│   │       ↓                                   │         │
│   │  + Residual Connection                    │         │
│   │       ↓                                   │         │
│   │  Layer Norm                               │         │
│   │       ↓                                   │         │
│   │  MLP (Linear → GELU → Linear)            │         │
│   │       ↓                                   │         │
│   │  + Residual Connection                    │         │
│   └───────────────────────────────────────────┘         │
│                    × L layers                           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              CLASSIFICATION HEAD                        │
│                                                         │
│   Take [CLS] token output → Layer Norm → MLP → Classes │
│                                                         │
│   Output: [num_classes] logits                          │
└─────────────────────────────────────────────────────────┘
```

---

## Step 1: Patch Embedding — Turning Pixels into Tokens

The key insight: treat an image like a sentence. Just as NLP tokenizes text into words, ViT tokenizes an image into **patches**.

```
Input Image (224 × 224 × 3)
         │
         ▼
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │   14 × 14 = 196 patches
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤   Each: 16 × 16 × 3 = 768 values
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

### Math

Given image **x** ∈ ℝ^(H×W×C) and patch size P:

- Number of patches: **N = (H × W) / P²** = (224 × 224) / 16² = **196**
- Each patch flattened: **x_p** ∈ ℝ^(P²·C) = ℝ^768
- Linear projection: **z = x_p · E**, where E ∈ ℝ^(768 × D)

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 196
        
        # Conv2d with kernel_size=stride=patch_size is equivalent
        # to splitting into patches + linear projection
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.projection(x)    # (B, 768, 14, 14)
        x = x.flatten(2)          # (B, 768, 196)
        x = x.transpose(1, 2)     # (B, 196, 768)
        return x
```

**Why Conv2d instead of manual reshaping?** It's mathematically identical but more efficient on GPU. The convolution with `kernel_size=stride=16` extracts non-overlapping 16×16 patches and projects them in one operation.

---

## Step 2: [CLS] Token and Position Embeddings

### [CLS] Token

Borrowed from BERT. A learnable vector prepended to the sequence. After passing through the Transformer, this token's output serves as the **image representation** for classification.

```
Before:  [P1, P2, P3, ..., P196]         ← 196 tokens
After:   [[CLS], P1, P2, P3, ..., P196]  ← 197 tokens
```

### Position Embeddings

Unlike CNNs, Transformers have **no notion of spatial order**. Without position embeddings, shuffling the patches would produce the same output. We add learnable 1D position embeddings:

```
Token:     [CLS]   P1    P2    P3   ...  P196
              +     +     +     +    +     +
Position:   E_0   E_1   E_2   E_3  ...  E_196
```

**Key finding from the paper:** 1D position embeddings work just as well as 2D. The model learns spatial structure from data.

### PyTorch Implementation

```python
class ViTEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2  # 196
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, 196, 768)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)    # (B, 1, 768)
        x = torch.cat([cls_tokens, x], dim=1)            # (B, 197, 768)
        x = x + self.pos_embed                           # (B, 197, 768)
        return x
```

---

## Step 3: Transformer Encoder — The Heart of ViT

Each Transformer layer consists of two sub-blocks:

```
            Input (B, 197, D)
                │
       ┌────────┤
       │   Layer Norm
       │        │
       │   Multi-Head Self-Attention
       │        │
       └───(+)──┘  ← Residual connection
                │
       ┌────────┤
       │   Layer Norm
       │        │
       │   MLP (D → 4D → D)
       │        │
       └───(+)──┘  ← Residual connection
                │
            Output (B, 197, D)
```

### Multi-Head Self-Attention (MHSA)

Every patch attends to every other patch. This is how ViT captures global relationships — even patch 1 (top-left) can attend to patch 196 (bottom-right) in layer 1.

```
Q, K, V = Linear(x), Linear(x), Linear(x)

Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

**Complexity:** O(N² · D) where N=197 — this is why ViT is expensive for high-resolution images.

### PyTorch Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 64
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, D = x.shape
        
        # Compute Q, K, V in one projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Weighted sum of values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # Pre-norm + residual
        x = x + self.mlp(self.norm2(x))    # Pre-norm + residual
        return x
```

**Why Pre-Norm (LayerNorm before attention)?** The original Transformer uses post-norm. ViT uses pre-norm because it stabilizes training for deeper models, avoiding the need for careful learning rate warmup.

---

## Step 4: Complete ViT Model

```python
class VisionTransformer(nn.Module):
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
        dropout=0.0,
    ):
        super().__init__()
        self.embedding = ViTEmbedding(img_size, patch_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)       # (B, 197, 768)
        x = self.dropout(x)
        x = self.encoder(x)         # (B, 197, 768)
        x = self.norm(x)
        cls_token = x[:, 0]         # (B, 768) — only [CLS]
        logits = self.head(cls_token)  # (B, num_classes)
        return logits

# Create ViT-Base/16
model = VisionTransformer(
    img_size=224, patch_size=16,
    embed_dim=768, depth=12, num_heads=12,
    num_classes=1000
)

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"ViT-Base/16: {total_params / 1e6:.1f}M parameters")
# Output: ViT-Base/16: 86.6M parameters
```

---

## Step 5: Model Variants

```
┌────────────┬────────┬────────────┬──────────┬───────┬────────┐
│   Model    │ Layers │ Hidden (D) │ MLP Size │ Heads │ Params │
├────────────┼────────┼────────────┼──────────┼───────┼────────┤
│ ViT-Small  │   12   │    384     │   1536   │   6   │  22M   │
│ ViT-Base   │   12   │    768     │   3072   │  12   │  86M   │
│ ViT-Large  │   24   │   1024     │   4096   │  16   │  307M  │
│ ViT-Huge   │   32   │   1280     │   5120   │  16   │  632M  │
└────────────┴────────┴────────────┴──────────┴───────┴────────┘

Patch sizes: /16 (default), /32 (fewer tokens, faster), /14 (more tokens, better)
Example: ViT-B/16 = Base model with 16×16 patches = 196 tokens
         ViT-L/14 = Large model with 14×14 patches = 256 tokens
```

---

## Key Insights and Training Details

### 1. Data Scale Matters Enormously

```
Dataset Size vs. ViT Performance:

ImageNet-1K (1.3M)    │████░░░░░░░░░░░░│  ViT < ResNet (not enough data)
ImageNet-21K (14M)    │█████████░░░░░░░│  ViT ≈ ResNet  
JFT-300M (300M)       │████████████████│  ViT >> ResNet 🏆

Key insight: Without large-scale pre-training, ViT lacks the
inductive biases (locality, translation equivariance) that
CNNs have built-in. Data compensates for missing inductive bias.
```

### 2. Attention Distance Analysis

The paper shows that even in early layers, some attention heads attend to **distant patches** — something impossible in early CNN layers:

```
Layer 1:   Some heads → local attention (like convolution)
           Some heads → GLOBAL attention (unique to ViT!)

Layer 6:   Most heads → medium-range attention

Layer 12:  Most heads → global attention (full image context)

This is fundamentally different from CNNs where receptive
field grows slowly through stacking layers.
```

### 3. Position Embedding Similarity

The learned position embeddings reveal that ViT discovers 2D structure on its own:

```
Position embedding similarity pattern (ViT-L/16):

Patch (7,7) is most similar to:
  ██████████████     Row 7 (horizontal neighbors)
  █             █    Column 7 (vertical neighbors)
  █      ●      █    And the patch itself
  █             █
  ██████████████

The model learns a 2D grid structure from 1D position indices!
```

### 4. Training Recipe

```
Pre-training:
  - Dataset: JFT-300M or ImageNet-21K
  - Optimizer: Adam (β₁=0.9, β₂=0.999)
  - Learning rate: 3e-4 with warmup + cosine decay
  - Batch size: 4096
  - Augmentation: RandAugment, Mixup
  - Resolution: 224 × 224

Fine-tuning:
  - Higher resolution (384 × 384) with interpolated position embeddings
  - SGD with momentum, lower learning rate
  - Short training (< 20 epochs)
```

---

## What Came After ViT

ViT opened the floodgates for vision transformers:

```
2020  ViT          — Proved Transformers work for vision
2021  DeiT         — Data-efficient training via distillation
2021  Swin         — Hierarchical ViT with shifted windows
2021  CLIP         — ViT + text encoder for zero-shot vision
2022  MAE          — Masked autoencoder pre-training for ViT
2023  DINOv2       — Self-supervised ViT features
2023  SAM          — Segment Anything using ViT backbone
2024  ViT in LLMs  — GPT-4V, Gemini use ViT-based encoders
```

---

## Citation

```bibtex
@inproceedings{dosovitskiy2021an,
  title={An Image is Worth 16x16 Words: Transformers 
         for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and 
          Kolesnikov, Alexander and Weissenborn, Dirk and 
          Zhai, Xiaohua and Unterthiner, Thomas and 
          Dehghani, Mostafa and Minderer, Matthias and 
          Heigold, Georg and Gelly, Sylvain and 
          Uszkoreit, Jakob and Houlsby, Neil},
  booktitle={ICLR},
  year={2021}
}
```

---

*Tutorial by Koyilbek Valiev — AI / ML Engineer | Research Engineer*
