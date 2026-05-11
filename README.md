# Deep Convolutional GAN (DCGAN) — Handwritten Digit Generation

A PyTorch implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on the MNIST dataset to generate realistic handwritten digits. This project demonstrates the core principles of adversarial training, convolutional architectures, and deep learning generative models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
  - [Generator](#generator)
  - [Discriminator](#discriminator)
- [Training Setup](#training-setup)
- [Results](#results)
- [Key Findings & Mode Collapse Analysis](#key-findings--mode-collapse-analysis)
- [Installation & Usage](#installation--usage)
- [File Structure](#file-structure)
- [Improvements & Future Work](#improvements--future-work)
- [References](#references)

---

## Overview

This project implements a **DCGAN** — a class of generative models that combine the power of Convolutional Neural Networks (CNNs) with Generative Adversarial Networks (GANs). The model consists of two competing neural networks:

- **Generator (G):** Transforms random noise into synthetic images
- **Discriminator (D):** Classifies images as real or fake

The generator learns to fool the discriminator by generating increasingly realistic digits, while the discriminator learns to distinguish real MNIST digits from fake generated ones. This adversarial process drives both networks to improve.

### Why DCGAN?

- **Stable training:** Uses convolutional layers instead of fully connected layers, reducing mode collapse
- **Principled architecture:** Follows guidelines from the seminal DCGAN paper (Radford et al., 2016)
- **Lightweight:** ~3.6M parameters — trainable on a single GPU
- **High-quality output:** Capable of generating 64×64 grayscale images with sharp detail

---

## Dataset

| Property | Value |
|---|---|
| **Name** | MNIST (Modified National Institute of Standards and Technology) |
| **Size** | 60,000 training images |
| **Image Size (Original)** | 28×28 pixels |
| **Image Size (Resized)** | 64×64 pixels |
| **Channels** | 1 (grayscale) |
| **Pixel Range** | [-1, 1] (normalized) |
| **Batch Size** | 128 |
| **Data Loader Workers** | 2 |
| **Pin Memory** | True |
| **Batches per Epoch** | 469 |

### Preprocessing

```python
transforms.Compose([
    transforms.Resize(64),              # Upscale 28×28 → 64×64
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
])
```

The images are resized to 64×64 to provide the convolutional layers with more spatial information to work with. Normalization to [-1, 1] matches the generator's Tanh activation output.

---

## Architecture

### Generator

The generator transforms a random noise vector **z** (100-dimensional, sampled from standard normal distribution) into a 64×64 grayscale image through progressive upsampling.

#### Architecture Diagram

```
Input: z ~ N(0, 1) [100×1×1]
    ↓
ConvTranspose2d(100 → 512, 4×4, stride=1, padding=0) + BatchNorm2d + ReLU
    ↓ [512×4×4]
ConvTranspose2d(512 → 256, 4×4, stride=2, padding=1) + BatchNorm2d + ReLU
    ↓ [256×8×8]
ConvTranspose2d(256 → 128, 4×4, stride=2, padding=1) + BatchNorm2d + ReLU
    ↓ [128×16×16]
ConvTranspose2d(128 → 64, 4×4, stride=2, padding=1) + BatchNorm2d + ReLU
    ↓ [64×32×32]
ConvTranspose2d(64 → 1, 4×4, stride=2, padding=1) + Tanh
    ↓
Output: Image [-1, 1] [1×64×64]
```

#### Key Features

| Component | Details |
|---|---|
| **Total Parameters** | 3,574,656 (~3.57M) |
| **Activation (Hidden)** | ReLU |
| **Activation (Output)** | Tanh (output range: [-1, 1]) |
| **Normalization** | Batch Normalization after every layer |
| **Bias** | Disabled (follows DCGAN guidelines) |
| **Weight Initialization** | Normal(0, 0.02) for conv layers |

#### Architectural Rationale

- **Transposed Convolutions:** Replace pooling layers for upsampling; learnable and gradient-friendly
- **Batch Normalization:** Stabilizes training by normalizing layer inputs; critical for GAN stability
- **No Bias in Conv:** Batch norm makes bias redundant; reduces parameters and improves training
- **Tanh Activation:** Produces output in [-1, 1], matching the normalized data range

---

### Discriminator

The discriminator takes a 64×64 grayscale image (real or generated) and outputs a single probability (0–1) indicating whether it's real (1) or fake (0).

#### Architecture Diagram

```
Input: Image [1×64×64]
    ↓
Conv2d(1 → 64, 4×4, stride=2, padding=1) + LeakyReLU(0.2)
    ↓ [64×32×32]
Conv2d(64 → 128, 4×4, stride=2, padding=1) + BatchNorm2d + LeakyReLU(0.2)
    ↓ [128×16×16]
Conv2d(128 → 256, 4×4, stride=2, padding=1) + BatchNorm2d + LeakyReLU(0.2)
    ↓ [256×8×8]
Conv2d(256 → 512, 4×4, stride=2, padding=1) + BatchNorm2d + LeakyReLU(0.2)
    ↓ [512×4×4]
Conv2d(512 → 1, 4×4, stride=1, padding=0) + Sigmoid
    ↓
Output: Probability (scalar, 0–1)
```

#### Key Features

| Component | Details |
|---|---|
| **Total Parameters** | 2,763,520 (~2.76M) |
| **Activation (Hidden)** | LeakyReLU(slope=0.2) |
| **Activation (Output)** | Sigmoid (output range: [0, 1]) |
| **Normalization** | Batch Normalization (except first layer) |
| **Bias** | Disabled |
| **Weight Initialization** | Normal(0, 0.02) for conv layers |

#### Architectural Rationale

- **Strided Convolutions:** Replace pooling for downsampling; learnable and gradient-friendly
- **LeakyReLU(0.2):** Prevents dying ReLU problem; allows small negative gradients
- **No BatchNorm in First Layer:** Prevents the discriminator from seeing batch statistics; improves stability
- **Sigmoid Output:** Produces probability in [0, 1]; compatible with Binary Cross-Entropy loss

---

## Training Setup

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| **Epochs** | 50 | Later epochs exhibit mode collapse; best checkpoints around epoch 20 |
| **Batch Size** | 128 | Balances memory and gradient estimation |
| **Learning Rate** | 0.0002 | Standard DCGAN learning rate; low to prevent instability |
| **Adam β₁** | 0.5 | From original DCGAN paper; controls momentum |
| **Adam β₂** | 0.999 | Default value for exponential decay of squared gradients |
| **Loss Function** | Binary Cross-Entropy (BCE) | Standard GAN loss |
| **Latent Vector Size** | 100 | Generator input noise dimension |
| **Weight Initialization** | Normal(0, 0.02) | DCGAN paper guideline |
| **Batch Norm Init** | Normal(1, 0.02) | Ensures stable initialization |

### Training Loop

Each epoch consists of 469 batches (60,000 images ÷ 128). For each batch:

#### Step 1: Update Discriminator
```
1. Forward pass on real images → output ≈ 1 (ideally)
2. Compute loss: L_D_real = BCE(D(real), 1)
3. Generate fake images from noise
4. Forward pass on fake images → output ≈ 0 (ideally)
5. Compute loss: L_D_fake = BCE(D(fake), 0)
6. Total loss: L_D = L_D_real + L_D_fake
7. Backpropagate and update discriminator weights
```

#### Step 2: Update Generator
```
1. Generate fake images from noise
2. Forward pass through discriminator
3. Compute loss: L_G = BCE(D(fake), 1)  ← Try to fool discriminator
4. Backpropagate and update generator weights
```

### Loss Functions

```python
criterion = nn.BCELoss()  # Binary Cross-Entropy

# Discriminator loss
errD_real = criterion(netD(real_imgs), torch.ones(batch_size))
errD_fake = criterion(netD(fake_imgs.detach()), torch.zeros(batch_size))
errD = errD_real + errD_fake

# Generator loss
errG = criterion(netD(fake_imgs), torch.ones(batch_size))
```

---

## Results

### Training Progression

Training was **stable and healthy** for the first ~25 epochs, with the model learning to generate increasingly realistic digit-like patterns. However, **mode collapse** occurred from epoch 30 onwards.

### Loss Curves

| Epoch | Loss_D | Loss_G | D(x) | D(G(z)) |
|---|---|---|---|---|
| 1 | 1.5863 | 0.9283 | 0.2867 | 0.0023 / 0.4612 |
| 5 | 0.3811 | 2.8830 | 0.8397 | 0.1568 / 0.0803 |
| 10 | 0.4298 | 1.5957 | 0.6887 | 0.0071 / 0.2709 |
| 15 | 0.0254 | 5.3933 | 0.9911 | 0.0161 / 0.0067 |
| 20 | 0.0482 | 5.5948 | 0.9955 | 0.0420 / 0.0049 |
| 25 | 0.0114 | 6.0340 | 0.9963 | 0.0076 / 0.0038 |
| 30 | 0.0002 | 8.9835 | 1.0000 | 0.0002 / 0.0002 |
| 40 | 0.0000 | 45.7614 | 1.0000 | 0.0000 / 0.0000 |
| 50 | 0.0000 | 43.7164 | 1.0000 | 0.0000 / 0.0000 |

**Notation:**
- **Loss_D:** Total discriminator loss
- **Loss_G:** Generator loss
- **D(x):** Discriminator output on real images (should ≈ 1)
- **D(G(z)):** Discriminator output on fake images before/after generator update

### Visual Quality

Generated sample grids were saved at epochs **10, 20, 30, 40, and 50**:

- **Epochs 1–20:** Progressive improvement; clear digit-like shapes emerge
- **Epochs 20–25:** Best visual quality; diverse, recognizable digits
- **Epochs 30–50:** Mode collapse; discriminator too strong; generator outputs degenerate patterns

### Generated Outputs

The project produces:
- ✅ 64-sample training progression grids (epochs 10, 20, 30, 40, 50)
- ✅ Final 16-sample image grid (2×8 layout)
- ✅ 10-sample inference demonstration
- ✅ Loss curve visualization

---

## Key Findings & Mode Collapse Analysis

### What is Mode Collapse?

Mode collapse occurs when the generator converges to producing a limited variety of outputs (a "mode"), rather than exploring the full diversity of the target distribution. In this project:

- Generator discovers a few digit patterns that fool the discriminator
- Discriminator becomes too strong (D(fake) → 0, D(real) → 1)
- Generator cannot generate gradients (loss plateau at high values)
- Diversity in outputs dramatically decreases

### Why Did It Happen?

1. **Discriminator Learning Rate:** At lr=0.0002, the discriminator learns quickly and becomes overly powerful after ~25 epochs
2. **Binary Cross-Entropy Limitation:** BCE suffers from vanishing gradients when the discriminator is too confident
3. **No Regularization:** No label smoothing, spectral normalization, or other stabilization techniques were used
4. **No Early Stopping:** Training continued to epoch 50, long past the optimal checkpoint (~epoch 20)

### Evidence in Loss Curves

```
Epoch 20-25: Loss_D ≈ 0.01-0.05    (low but healthy)
             Loss_G ≈ 5.0-6.0       (moderate, training progressing)
             
Epoch 30-50: Loss_D ≈ 0.0000        (collapsed to zero)
             Loss_G ≈ 40-50          (severe degradation)
             D(fake) ≈ 0 / 0         (no gradients)
```

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- GPU recommended (T4 on Google Colab used for development)

### Setup

```bash
# Clone the repository
git clone https://github.com/alyy10/Deep-Convolutional-GAN-DCGAN-Assignment-.git
cd Deep-Convolutional-GAN-DCGAN-Assignment-

# Install dependencies
pip install torch torchvision numpy matplotlib

# (Optional) For GPU support on Google Colab
# Simply open in Colab and runtime will use GPU automatically
```

### Running the Notebook

```bash
jupyter notebook Handwritten_Digit_Generation_with_DCGAN.ipynb
```

Or run directly in **Google Colab** for GPU acceleration:
1. Upload the notebook to Colab
2. Set runtime type to **GPU (T4)**
3. Execute cells sequentially

### Training Time

- **Full 50 epochs:** ~3-4 minutes on T4 GPU
- **Per epoch:** ~4-5 seconds

### Loading a Pre-trained Generator

```python
import torch
from YourGeneratorClass import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netG.load_state_dict(torch.load("generator.pth", map_location=device))
netG.eval()

# Generate new samples
with torch.no_grad():
    noise = torch.randn(16, 100, 1, 1, device=device)
    fake_images = netG(noise)
```

---

## File Structure

```
Deep-Convolutional-GAN-DCGAN-Assignment-/
├── README.md                                    # This file
├── Handwritten_Digit_Generation_with_DCGAN.ipynb # Main notebook
├── DCGAN_ExplanationOfExecution.md              # Execution notes
├── generator.pth                                # Pre-trained generator weights
├── discriminator.pth                            # Pre-trained discriminator weights
└── generated_images/                            # Output directory
    ├── epoch_010.png                            # 8×8 sample grid at epoch 10
    ├── epoch_020.png                            # 8×8 sample grid at epoch 20
    ├── epoch_030.png                            # 8×8 sample grid at epoch 30
    ├── epoch_040.png                            # 8×8 sample grid at epoch 40
    ├── epoch_050.png                            # 8×8 sample grid at epoch 50
    ├── final_16_samples.png                     # 2×8 final sample grid
    ├── inference_10_digits.png                  # Inference demo (10 samples)
    ├── loss_curve.png                           # Generator/Discriminator loss plot
    └── training_progression.png                 # 5-stage training progression
```

---

## Improvements & Future Work

### Short-term Improvements (Achievable)

1. **Label Smoothing:**
   ```python
   # Replace hard labels
   real_label = 0.9  # Instead of 1.0
   fake_label = 0.1  # Instead of 0.0
   ```
   This reduces discriminator confidence and slows its learning.

2. **Early Stopping:**
   Load the best checkpoint from epoch 20 instead of epoch 50 for inference.

3. **Spectral Normalization:**
   Apply to discriminator to stabilize gradients:
   ```python
   from torch.nn.utils.spectral_norm import spectral_norm
   self.conv1 = spectral_norm(nn.Conv2d(1, 64, 4, 2, 1))
   ```

4. **Learning Rate Scheduling:**
   Reduce learning rate over time to stabilize late training.

### Medium-term Improvements (Recommended)

5. **WGAN or WGAN-GP:**
   Replace BCE with Wasserstein distance; solves vanishing gradient problem.
   ```python
   # WGAN loss: E[D(real)] - E[D(fake)]
   errD = -torch.mean(netD(real)) + torch.mean(netD(fake))
   errG = -torch.mean(netD(fake))
   ```

6. **Progressive Growing:**
   Start training at 8×8, progressively expand to larger resolutions.

7. **Conditional GAN (cGAN):**
   Add digit labels to enable class-conditional generation.

### Long-term Enhancements

8. **Diffusion Models:** Modern alternative to GANs; more stable training
9. **StyleGAN:** State-of-the-art architecture; produces higher-quality images
10. **Evaluation Metrics:**
    - Inception Score (IS)
    - Fréchet Inception Distance (FID)
    - Classification accuracy on generated digits

---

## Technical Implementation Details

### Weight Initialization

Proper initialization is critical for GAN training stability:

```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)
```

### Device Management

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)
```

### Data Pipeline

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True  # Accelerate GPU data transfer
)
```

---

## References

### Seminal Papers

1. **GAN (2014):** Goodfellow et al., "Generative Adversarial Networks"  
   https://arxiv.org/abs/1406.2661

2. **DCGAN (2016):** Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"  
   https://arxiv.org/abs/1511.06434

3. **WGAN (2017):** Arjovsky et al., "Wasserstein GAN"  
   https://arxiv.org/abs/1701.07875

### Key Concepts

- **Mode Collapse:** https://arxiv.org/abs/1606.03498
- **Batch Normalization:** https://arxiv.org/abs/1502.03167
- **LeakyReLU:** Maas et al., "Rectifier Nonlinearities Improve Neural Network Acoustic Models"

### Libraries & Tools

- **PyTorch:** https://pytorch.org/
- **Torchvision:** https://pytorch.org/vision/
- **MNIST Dataset:** http://yann.lecun.com/exdb/mnist/

---

## Learning Outcomes

By completing this project, you understand:

✅ **GAN Fundamentals:** How generator and discriminator compete  
✅ **DCGAN Architecture:** Convolutional designs for stable training  
✅ **Training Dynamics:** Adversarial loss functions and optimization  
✅ **Hyperparameter Tuning:** Impact of LR, batch size, weight init  
✅ **Mode Collapse:** Why it happens and how to mitigate it  
✅ **Generative Models:** From noise to realistic image synthesis  
✅ **PyTorch:** Building, training, and deploying neural networks  

---

## Author

**alyy10** — Deep Learning & Generative AI Enthusiast

---

## License

This project is provided as-is for educational purposes. Feel free to use, modify, and distribute for non-commercial applications.

---

## Questions & Troubleshooting

**Q: Why do my generated images look blurry after epoch 30?**  
A: This is mode collapse. The discriminator has become too strong. Try loading the checkpoint from epoch 20 instead, or apply label smoothing.

**Q: Can I train on a CPU?**  
A: Yes, but it will be ~50× slower. GPU is highly recommended. Use Google Colab for free GPU access.

**Q: How do I generate high-resolution (256×256) images?**  
A: You'd need to modify the architecture and train with progressive growing or use a larger latent vector. Start with this DCGAN as a foundation.

**Q: What if the discriminator loss stays at ~0 from the beginning?**  
A: The discriminator is too strong. Reduce its learning rate (try 0.0001) or add label smoothing.

---

## 🎉 Thank You

Thank you for exploring this DCGAN implementation! Questions, feedback, or contributions are always welcome.

**GitHub:** https://github.com/alyy10/Deep-Convolutional-GAN-DCGAN-Assignment-
