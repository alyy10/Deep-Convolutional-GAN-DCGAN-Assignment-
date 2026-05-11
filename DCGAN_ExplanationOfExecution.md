# Handwritten Digit Generation with DCGAN

## Overview

We built a Deep Convolutional GAN (DCGAN) that learns to generate realistic handwritten digits by training on the MNIST dataset. The core idea is two networks competing against each other — one generates fake images, the other tries to catch them — and through that competition, the generator gradually learns to produce convincing digits.

---

## Dataset

- **MNIST** — 60,000 grayscale images of handwritten digits (0–9), each 28×28 pixels
- Resized to **64×64** to give the convolutional layers more spatial resolution to work with
- Pixel values normalized to **[-1, 1]** to match the generator's Tanh output
- Loaded in batches of 128 using PyTorch's DataLoader

---

## Architecture

### Generator
Takes a random noise vector of size **100** (sampled from a standard normal distribution) and progressively upsamples it into a 64×64 image using 5 transposed convolution layers:

```
Noise (100×1×1) → 512×4×4 → 256×8×8 → 128×16×16 → 64×32×32 → 1×64×64
```

Each layer uses Batch Normalization + ReLU. The final layer uses **Tanh** to keep outputs in [-1, 1]. Total parameters: ~3.57M.

### Discriminator
Takes a 64×64 image (real or fake) and outputs a single probability — real or fake. It mirrors the generator using 5 strided convolution layers:

```
1×64×64 → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → scalar
```

Each layer uses Batch Normalization + LeakyReLU (slope 0.2). Final layer uses **Sigmoid**. Total parameters: ~2.76M.

---

## Training Setup

| Setting | Value |
|---|---|
| Epochs | 50 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Adam β1 | 0.5 |
| Loss function | Binary Cross-Entropy |
| Latent vector size | 100 |

Weight initialization follows the DCGAN paper — Normal(0, 0.02) for conv layers, Normal(1, 0.02) for batch norm.

Each training iteration has two steps:
1. **Update Discriminator** — train it on real images (label=1) and generated fakes (label=0)
2. **Update Generator** — generate fakes, pass through discriminator, compute loss against real labels (trying to fool it)

---

## Results

Training was healthy for the first ~25 epochs. The discriminator and generator losses stayed in a reasonable range, and generated digit quality improved steadily.

After epoch 30, the discriminator loss collapsed to ~0.0000 and the generator loss spiked to 40–50. This is **mode collapse** — the discriminator became too strong and the generator could no longer fool it, causing the gradient signal to vanish. The best visual results came from the **epoch 10–25** range.

**Loss progression (sample):**

| Epoch | Loss_D | Loss_G |
|---|---|---|
| 1 | 1.5863 | 0.9283 |
| 10 | 0.4298 | 1.5957 |
| 20 | 0.0482 | 5.5948 |
| 30 | 0.0002 | 8.9835 |
| 50 | 0.0000 | 43.7164 |

Generated sample grids were saved at epochs 10, 20, 30, 40, and 50. A final set of 16 samples and an inference demo of 10 digits were also produced.

---

## What Could Be Improved

- **Label smoothing** on the discriminator (real labels → 0.9 instead of 1.0) to slow it down
- **WGAN or WGAN-GP** to fix the vanishing gradient problem that caused collapse here
- **Fewer epochs or early stopping** — pulling the best checkpoint (around epoch 20) would give better results than training to epoch 50
