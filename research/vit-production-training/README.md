# Training ViT Right: From Unstable Loss to Production-Grade Pipeline

**Tutorial by:** Koyilbek Valiev  
**Topics:** Vision Transformers · Training Engineering · PyTorch · CIFAR-10

---

## The Problem: Why Vanilla ViT Training Fails

This is the loss curve of a ViT trained naively on CIFAR-10:

```
Epoch 1:   Loss 2.36   ← okay
Epoch 2:   Loss 29.77  ← exploding
Epoch 3:   Loss 5.72
Epoch 4:   Loss 4.32
Epoch 5:   Loss 19.44  ← exploding again
Epoch 6:   Loss 10.33
...
```

No convergence. No validation metric. No idea if anything is working.

This tutorial diagnoses every mistake, fixes them one by one, and builds a pipeline that trains stably to **~85% CIFAR-10 accuracy** in under 35 seconds per epoch on a T4 GPU.

---

## Diagnosis: 5 Root Causes

### Root Cause 1 — The 224×224 Trap

The most expensive mistake: resizing 32×32 CIFAR-10 images to 224×224.

```python
# What beginners write (WRONG)
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # ← WHY?
    transforms.ToTensor(),
])
```

ViT-Base was designed for ImageNet at 224×224. But CIFAR-10 images are **32×32 natively**. Resizing them is not just wasteful — it actively hurts:

```
Native 32×32 input:
  Pixels per image:  32 × 32 × 3 = 3,072
  Patches (patch=4): 8 × 8       = 64 patches
  Epoch time (T4):   ~30 seconds

After resize to 224×224:
  Pixels per image:  224 × 224 × 3 = 150,528  ← 49× more pixels
  Patches (patch=16): 14 × 14      = 196 patches
  Epoch time (T4):   ~9 minutes    ← 18× slower
  Information gain:  ZERO          ← you printed a stamp on a billboard
```

**Fix:** use native 32×32 with `patch_size=4`.

---

### Root Cause 2 — Model Oversized by 15×

ViT-Base has 86M parameters. It was designed for 1.28M ImageNet images. CIFAR-10 has 50k training examples.

```
Model-to-data ratio:

  ViT-Base on CIFAR-10:
    86M params / 50k images = 1,720 parameters per training example
    
  A well-fitted model (rule of thumb):
    ~10-100 parameters per training example
    
  ViT-Tiny on CIFAR-10:
    5.5M params / 50k images = 110 parameters per training example  ✓
```

An oversized model memorizes noise instead of learning structure. This is why validation accuracy never improves even as training loss decreases.

**Fix:** ViT-Tiny config — `embed_dim=192, num_heads=3, num_layers=12`.

```
┌─────────────────────────────────────────────────────────────┐
│                  Model Sizing Guide                         │
├───────────────┬──────────┬──────────┬──────────┬───────────┤
│  Dataset      │ Images   │ Model    │  Params  │ patch_size│
├───────────────┼──────────┼──────────┼──────────┼───────────┤
│ CIFAR-10/100  │   50K    │ ViT-Tiny │   5.5M   │     4     │
│ ImageNet-1K   │  1.28M   │ ViT-Small│   22M    │    16     │
│ ImageNet-21K  │   14M    │ ViT-Base │   86M    │    16     │
│ JFT-300M      │  300M    │ ViT-Large│   307M   │    16     │
└───────────────┴──────────┴──────────┴──────────┴───────────┘
```

---

### Root Cause 3 — Learning Rate Too High, No Warmup

`Adam(lr=1e-3)` is the default in most tutorials. For transformers, it is almost always wrong.

Transformers are sensitive to learning rate at initialization. The attention weights haven't formed meaningful patterns yet. Starting at `1e-3` fires massive gradient updates into randomly-initialized attention weights — creating the chaotic loss spikes you saw.

```
Learning rate without warmup:

  Step 1:  lr = 1e-3  ← full force on random weights → chaos
  Step 2:  lr = 1e-3  ← still full force → gradient explosion
  ...
  
Learning rate with warmup:

  Step 1:  lr = 1e-9  ← gentle nudges
  Step 2:  lr = 2e-7
  ...
  Step 1500 (epoch 5 end): lr = 3e-4  ← full rate, weights now stable
  ...
  Step 9000 (epoch 30 end): lr = 1e-6  ← cosine decay to near-zero
```

The **linear warmup + cosine decay** schedule is standard across all modern transformer training:

```
LR

3e-4 │          ╭───╮
     │         /     ╲
     │        /       ╲
     │       /         ╲___
     │      /               ────────────────────╲
1e-6 │─────/                                     ╲────
     └─────────────────────────────────────────────────── Epoch
           │←warmup→│←────────── cosine decay ──────────→│
           0        5                                     30
```

---

### Root Cause 4 — Adam vs AdamW

`Adam` absorbs weight decay into the gradient update, which makes it theoretically incorrect and practically worse for generalization. `AdamW` (Loshchilov & Hutter, 2019) decouples weight decay into a direct parameter shrinkage:

```
Adam update:
  θ ← θ - lr · (gradient + λ · θ)   ← weight decay mixed with gradient
  
AdamW update:
  θ ← θ - lr · gradient             ← gradient update only
  θ ← θ - lr · λ · θ                ← weight decay applied separately
```

More importantly: **not all parameters should have weight decay**. Bias terms, normalization scales, and embedding vectors are 1D — decaying them toward zero actively hurts training.

```
Parameters that need weight decay:
  nn.Linear.weight    (2D)  ✓  decay it
  nn.Conv2d.weight    (4D)  ✓  decay it

Parameters that must NOT have weight decay:
  nn.Linear.bias      (1D)  ✗  exclude
  nn.LayerNorm.weight (1D)  ✗  exclude
  nn.LayerNorm.bias   (1D)  ✗  exclude
  cls_token           (3D)  ✗  exclude (embedding, not a weight)
  pos_embed           (3D)  ✗  exclude
```

---

### Root Cause 5 — No Validation, No Checkpointing

The original training loop evaluates nothing. You have no idea whether the model is learning, overfitting, or oscillating. You also lose all progress if the kernel crashes.

---

## The Fix: Production Training Pipeline

### Step 1: Right-Sized Model for CIFAR-10

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,         # native CIFAR-10 size — no resize
        patch_size=4,        # 4×4 patches → 64 tokens
        in_channels=3,
        num_classes=10,
        embed_dim=192,       # ViT-Tiny: 4× smaller than ViT-Base
        num_heads=3,         # head_dim = 192/3 = 64
        mlp_dim=768,         # 4× embed_dim (standard ratio)
        num_layers=12,
        dropout=0.1,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2   # 64

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, 64, 192)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))
```

Parameter count comparison:

```
ViT-Base (original config):   86,567,656 parameters
ViT-Tiny (our config):         5,491,210 parameters
                               ──────────
Reduction:                        15.8×  smaller
```

---

### Step 2: Correct Data Loading

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-10 channel statistics — not generic (0.5, 0.5, 0.5)
# Computed from the actual training set
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # shift by up to 4 pixels
    transforms.RandomHorizontalFlip(),       # 50% horizontal flip
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# device-aware DataLoader settings
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps'  if torch.backends.mps.is_available() else 'cpu')
_pin     = (device.type == 'cuda')
_workers = 0 if device.type == 'mps' else 4

train_loader = DataLoader(
    datasets.CIFAR10('./data', train=True,  download=True, transform=train_transform),
    batch_size=128, shuffle=True,
    num_workers=_workers, pin_memory=_pin,
    persistent_workers=(_workers > 0),
    prefetch_factor=(2 if _workers > 0 else None),
)
val_loader = DataLoader(
    datasets.CIFAR10('./data', train=False, download=True, transform=val_transform),
    batch_size=256, shuffle=False,
    num_workers=_workers, pin_memory=_pin,
    persistent_workers=(_workers > 0),
    prefetch_factor=(2 if _workers > 0 else None),
)
```

**`prefetch_factor=2`**: each worker pre-loads 2 batches while the GPU processes the current one — eliminates the GPU stall waiting for data.

**`pin_memory=True`**: keeps CPU tensors in page-locked memory, enabling faster DMA transfers to GPU.

**`non_blocking=True`** on `.to(device)`: GPU transfer overlaps with the next CPU operation:

```python
# In the training loop:
imgs   = imgs.to(device, non_blocking=True)    # async transfer
labels = labels.to(device, non_blocking=True)  # async transfer
# GPU continues previous batch while these transfers happen
```

---

### Step 3: AdamW with Parameter Groups

```python
import torch.optim as optim

# Separate 2D weights (need decay) from 1D params (no decay)
decay_params    = [p for n, p in model.named_parameters()
                   if p.ndim >= 2 and p.requires_grad]
no_decay_params = [p for n, p in model.named_parameters()
                   if p.ndim <  2 and p.requires_grad]

optimizer = optim.AdamW(
    [
        {'params': decay_params,    'weight_decay': 1e-4},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ],
    lr=3e-4,
    fused=(device.type == 'cuda'),   # single fused kernel, ~15% faster
)
```

The `fused=True` flag uses a Triton-based kernel that updates all parameters in one GPU pass instead of a Python loop over parameter tensors. Available in PyTorch 2.0+ on CUDA.

---

### Step 4: Warmup + Cosine Decay Schedule

```python
NUM_EPOCHS    = 30
WARMUP_EPOCHS = 5

warmup = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1e-6,    # start at lr * 1e-6 ≈ 0
    end_factor=1.0,       # reach full lr=3e-4 at epoch 5
    total_iters=WARMUP_EPOCHS,
)
cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS - WARMUP_EPOCHS,   # 25 epochs of cosine
    eta_min=1e-6,                        # decay to near-zero
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[WARMUP_EPOCHS],
)
```

---

### Step 5: Label Smoothing

Standard cross-entropy pushes the model toward probability 1.0 for the correct class:

```
Target (hard):   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]   ← "be 100% sure"
Target (smooth): [.01, .01, .01, .91, .01, .01, .01, .01, .01, .01]
```

With only 50k training images, hard targets cause overconfidence and poor generalization. Label smoothing with `ε=0.1` acts as built-in regularization:

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

### Step 6: Mixed Precision (AMP) — Device-Aware

AMP computes the forward pass in `float16` (2× faster, 2× less memory) while keeping a `float32` master copy for gradient accumulation. The `GradScaler` prevents float16 underflow during the backward pass.

**Critical:** MPS (Apple Silicon) does not support `GradScaler`. The implementation must be device-aware:

```python
USE_AMP = (device.type == 'cuda')
scaler  = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# In training loop:
with torch.autocast(device_type=device.type, enabled=USE_AMP):
    logits = model(imgs)
    loss   = criterion(logits, labels) / ACCUM_STEPS

scaler.scale(loss).backward()
scaler.unscale_(optimizer)                           # unscale before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

---

### Step 7: Gradient Accumulation

Larger effective batch sizes improve gradient quality and training stability. But doubling batch size doubles GPU memory. Gradient accumulation simulates a large batch at no memory cost:

```
Batch size 128, accumulation steps 4:
  Effective batch = 128 × 4 = 512

Step 1: forward + backward (batch 1) ─┐
Step 2: forward + backward (batch 2)  ├── accumulate gradients
Step 3: forward + backward (batch 3)  │   (do NOT step optimizer)
Step 4: forward + backward (batch 4) ─┘
  → clip gradients → optimizer.step() → optimizer.zero_grad()
```

```python
ACCUM_STEPS = 4
optimizer.zero_grad()

for step, (imgs, labels) in enumerate(train_loader):
    imgs   = imgs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, enabled=USE_AMP):
        loss = criterion(model(imgs), labels) / ACCUM_STEPS  # scale loss

    scaler.scale(loss).backward()   # accumulate scaled gradients

    if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

---

### Step 8: GPU Memory Format & torch.compile

```python
# channels_last: stores (B, H, W, C) instead of (B, C, H, W)
# Better cache locality for Conv2d operations
if device.type == 'cuda':
    model = model.to(memory_format=torch.channels_last)

# cuDNN auto-selects fastest algorithm for fixed input shapes
torch.backends.cudnn.benchmark = True

# TF32: faster matmul on Ampere+ GPUs, negligible accuracy loss
torch.backends.cuda.matmul.allow_tf32 = True

# torch.compile: fused CUDA kernels (PyTorch 2.0+)
if device.type == 'cuda' and hasattr(torch, 'compile'):
    try:
        model = torch.compile(model)
    except Exception:
        pass
```

---

### Step 9: Full Training Loop

```python
import csv, time, os

best_val_acc = 0.0
LOG_PATH     = 'training_log.csv'

with open(LOG_PATH, 'w', newline='') as f:
    csv.writer(f).writerow(
        ['epoch','train_loss','train_acc','val_loss','val_acc','lr','time_s']
    )

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    # ── Train ──────────────────────────────────────────────
    model.train()
    t_loss = t_correct = t_total = 0
    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(train_loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == 'cuda':
            imgs = imgs.to(memory_format=torch.channels_last)

        with torch.autocast(device_type=device.type, enabled=USE_AMP):
            logits = model(imgs)
            loss   = criterion(logits, labels) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        t_loss    += loss.item() * ACCUM_STEPS
        t_correct += (logits.argmax(1) == labels).sum().item()
        t_total   += labels.size(0)

    scheduler.step()

    # ── Validate ───────────────────────────────────────────
    model.eval()
    v_loss = v_correct = v_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=USE_AMP):
                logits = model(imgs)
            v_loss    += criterion(logits, labels).item()
            v_correct += (logits.argmax(1) == labels).sum().item()
            v_total   += labels.size(0)

    # ── Log & Checkpoint ───────────────────────────────────
    ta = 100 * t_correct / t_total
    va = 100 * v_correct / v_total
    tl = t_loss / len(train_loader)
    vl = v_loss / len(val_loader)
    lr = scheduler.get_last_lr()[0]
    dt = time.time() - t0

    if va > best_val_acc:
        best_val_acc = va
        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save({'epoch': epoch, 'model_state_dict': raw.state_dict(),
                    'val_acc': va}, 'best_model.pth')

    with open(LOG_PATH, 'a', newline='') as f:
        csv.writer(f).writerow(
            [epoch, f'{tl:.4f}', f'{ta:.2f}', f'{vl:.4f}',
             f'{va:.2f}', f'{lr:.2e}', f'{dt:.1f}']
        )

    print(f'Epoch {epoch:02d}/{NUM_EPOCHS} | '
          f'Train {tl:.4f}/{ta:.2f}% | '
          f'Val {vl:.4f}/{va:.2f}% | '
          f'LR {lr:.2e} | {dt:.1f}s')
```

---

## Results: Before vs After

```
BEFORE (vanilla ViT-Base on 224×224 CIFAR-10)
─────────────────────────────────────────────
Epoch time:    ~9 minutes
Loss curve:    2.36 → 29.77 → 5.72 → 4.32 → 19.44  (unstable)
Val accuracy:  not measured
Parameters:    86M

AFTER (ViT-Tiny on native 32×32 CIFAR-10)
──────────────────────────────────────────
Epoch time:    ~30 seconds
Loss curve:    2.31 → 2.05 → 1.84 → 1.68 → 1.55  (smooth)
Val accuracy:  ~83–85% at epoch 30
Parameters:    5.5M
```

---

## Summary: The Checklist

```
Training pipeline checklist for ViT on CIFAR-10:

Image size
  [ ] Use native resolution (32×32), NOT resize to 224×224
  [ ] patch_size=4 for 32×32 images → 64 tokens

Model sizing
  [ ] ViT-Tiny: embed_dim=192, heads=3, layers=12, mlp_dim=768
  [ ] ~5.5M params for 50k training images

Optimizer
  [ ] AdamW, NOT Adam
  [ ] lr=3e-4 with fused=True on CUDA
  [ ] Separate param groups: decay 2D weights, skip 1D params

LR Schedule
  [ ] Linear warmup for first 5 epochs
  [ ] Cosine decay from epoch 5 to 30
  [ ] eta_min=1e-6

Regularization
  [ ] Dropout=0.1 in attention and MLP
  [ ] Label smoothing=0.1
  [ ] RandomCrop(32, padding=4) + RandomHorizontalFlip

Training loop
  [ ] Gradient clipping: max_norm=1.0
  [ ] Gradient accumulation: 4 steps → effective batch 512
  [ ] AMP: autocast + GradScaler (CUDA only)
  [ ] non_blocking=True transfers
  [ ] channels_last memory format (CUDA)
  [ ] torch.compile (PyTorch 2.0+, CUDA)
  [ ] cudnn.benchmark=True + allow_tf32=True

Monitoring
  [ ] Validation accuracy every epoch
  [ ] Save best model checkpoint
  [ ] CSV metrics log
```

---

## Citation

```bibtex
@inproceedings{dosovitskiy2021an,
  title={An Image is Worth 16x16 Words},
  author={Dosovitskiy et al.},
  booktitle={ICLR},
  year={2021}
}
@article{loshchilov2019decoupled,
  title={Decoupled Weight Decay Regularization},
  author={Loshchilov and Hutter},
  year={2019}
}
```

---

*Tutorial by Koyilbek Valiev — AI / ML Research Engineer*
