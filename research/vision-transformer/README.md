# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

**Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby

**Venue:** ICLR 2021  
**Paper:** [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

---

## Key Idea

While the Transformer architecture has become the standard for NLP tasks, its application to computer vision remained limited. This paper shows that a **pure Transformer** applied directly to sequences of image patches can perform very well on image classification tasks, without relying on CNNs at all.

## How ViT Works

1. **Patch Embedding:** An image is split into fixed-size patches (e.g., 16x16). Each patch is flattened and linearly projected to a embedding vector.
2. **Position Embeddings:** Learnable 1D position embeddings are added to retain spatial information.
3. **[CLS] Token:** A special classification token is prepended to the sequence, similar to BERT.
4. **Transformer Encoder:** The sequence of patch embeddings is passed through a standard Transformer encoder (multi-head self-attention + MLP blocks).
5. **Classification Head:** The output of the [CLS] token is passed through an MLP head for final classification.

## Key Results

- When pre-trained on large datasets (JFT-300M, ImageNet-21k), ViT attains **excellent results** on multiple image recognition benchmarks.
- **ViT-Large/16** achieves **87.76%** top-1 accuracy on ImageNet.
- ViT requires substantially **less computational resources** to train compared to state-of-the-art CNNs when pre-trained at scale.
- Without large-scale pre-training, ViT underperforms compared to CNNs of similar size — the inductive biases of CNNs (locality, translation equivariance) are beneficial with limited data.

## Architecture Variants

| Model | Layers | Hidden Size | MLP Size | Heads | Params |
|-------|--------|-------------|----------|-------|--------|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

## Why It Matters

ViT demonstrated that Transformers can match or exceed CNN performance in vision when given enough data. This opened the door to a unified Transformer architecture across modalities (text, image, video, audio), leading to the explosion of vision-language models (CLIP, DALL-E, etc.) and multimodal AI.

## Citation

```bibtex
@inproceedings{dosovitskiy2021an,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
