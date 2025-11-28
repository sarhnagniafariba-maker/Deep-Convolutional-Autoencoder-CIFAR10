# Convolutional Autoencoder and Transfer Learning on CIFAR-10

This project implements a **Convolutional Autoencoder (CAE)** using PyTorch to learn meaningful latent representations of CIFAR-10 images in a self-supervised manner.  
After unsupervised training, the **encoder** is used as a fixed feature extractor and a **classifier head** is trained on top of it to perform supervised image classification.

---

## 1. Project Overview

### Phase 1: Autoencoder Training (Self-Supervised)
The autoencoder learns to:
- Compress a 32×32 RGB image into a compact latent space  
- Reconstruct the image with minimum distortion  
- Extract features that capture semantic visual patterns

### Phase 2: Transfer Learning (Supervised)
The encoder is frozen and used as a feature extractor:
- A fully connected classification head is added  
- The classifier is trained on CIFAR-10 labels  
- Latent space improves accuracy due to pretrained representations  

### Included Features
- Convolutional Autoencoder architecture  
- Training loop with step-wise loss tracking  
- Loss curve visualization with Matplotlib  
- Reconstruction visualization (original vs reconstructed)  
- Transfer learning classifier with accuracy calculation  

---

## 2. Architecture

### Encoder
- 3 convolutional layers  
- Batch Normalization  
- ReLU activations  
- Latent feature size: **128 × 4 × 4**

### Decoder
- 3 ConvTranspose layers  
- ReLU + Tanh  
- Final output: **3 × 32 × 32**

### Classifier Head
Used after freezing the encoder:
- Flatten  
- Linear(2048 → 256)  
- Dropout(0.3)  
- Linear(256 → 10)

---

## 3. How to Run

### Install Requirements
```bash
pip install torch torchvision matplotlib numpy

