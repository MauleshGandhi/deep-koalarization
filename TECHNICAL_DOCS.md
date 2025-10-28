# Deep Koalarization: Technical Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Architecture Details](#architecture-details)
4. [Implementation](#implementation)
5. [Experimental Results](#experimental-results)
6. [Usage Guide](#usage-guide)

## Introduction

Image colorization is a challenging computer vision task that involves adding realistic colors to grayscale images. This project focuses specifically on manga colorization, which presents unique challenges due to the intensive use of patterns, hatching, and screening techniques.

### Problem Statement

Traditional colorization methods rely on intensity continuity, which works well for photographs but fails on manga images that contain:
- Hand-drawn hatching patterns
- Printed screening patterns  
- Complex textures and fine details
- Abrupt intensity changes

### Our Approach

We implement the "Deep Koalarization" methodology that combines:
- A custom CNN encoder-decoder architecture
- Pre-trained Inception-ResNet-v2 for semantic understanding
- LAB color space processing for perceptually uniform colorization

## Methodology

### Color Space Representation

**LAB Color Space Advantages:**
- **L channel**: Represents lightness/luminance
- **a channel**: Green-red color component
- **b channel**: Blue-yellow color component
- Perceptually uniform color differences
- Separates color information from luminance

**Processing Pipeline:**
1. Input: Grayscale image (L channel only)
2. Process: Predict a and b channels using neural networks
3. Output: Full color LAB image converted to RGB

### Architecture Overview

```
Input Image (Grayscale)
    ↓
LAB Conversion (L channel)
    ↓
┌─────────────────┐    ┌──────────────────────┐
│   CNN Encoder   │    │  Inception-ResNet-v2 │
│   (224×224×1)   │    │    (299×299×3)       │
└─────────────────┘    └──────────────────────┘
    ↓                             ↓
    └─────────── Fusion Layer ────────┘
                      ↓
               CNN Decoder
                      ↓
              ab Channels (224×224×2)
                      ↓
          Combine with L Channel
                      ↓
              LAB → RGB Conversion
                      ↓
          Colorized Output Image
```

## Architecture Details

### Encoder Network

**Purpose**: Extract spatial features from grayscale input

**Architecture**:
```python
Layer 1: Conv2D(64, 3×3, stride=2, ReLU) + He initialization
Layer 2: Conv2D(128, 3×3, stride=1, ReLU)
Layer 3: Conv2D(128, 3×3, stride=2, ReLU)
Layer 4: Conv2D(256, 3×3, stride=1, ReLU)
Layer 5: Conv2D(256, 3×3, stride=2, ReLU)
Layer 6: Conv2D(512, 3×3, stride=1, ReLU)
Layer 7: Conv2D(512, 3×3, stride=1, ReLU)
Layer 8: Conv2D(256, 3×3, stride=1, ReLU)
```

**Output**: 28×28×256 feature maps

### Pre-trained Feature Extractor

**Model**: Inception-ResNet-v2 (ImageNet pre-trained)
**Input Size**: 299×299×3 (L channel repeated 3 times)
**Output**: 1536-dimensional feature vector
**Training**: Frozen weights (no fine-tuning)

### Fusion Layer

**Purpose**: Combine spatial features with semantic understanding

**Process**:
1. Take 1536-D feature vector from Inception-ResNet-v2
2. Expand dimensions to match encoder output (28×28)
3. Replicate across spatial dimensions
4. Concatenate with encoder output: 28×28×(256+1536)
5. Apply 1×1 convolution: Conv2D(256, 1×1)

### Decoder Network

**Purpose**: Upsample features to predict ab channels

**Architecture**:
```python
Layer 1: Conv2D(128, 3×3, ReLU)
Layer 2: UpSampling2D(2×2) → 56×56
Layer 3: Conv2D(64, 3×3, ReLU)
Layer 4: Conv2D(64, 3×3, ReLU)
Layer 5: UpSampling2D(2×2) → 112×112
Layer 6: Conv2D(32, 3×3, ReLU)
Layer 7: Conv2D(2, 3×3, Tanh) → ab channels
Layer 8: UpSampling2D(2×2) → 224×224×2
```

**Output**: 224×224×2 (a and b channels)

## Implementation

### Data Preparation

**Datasets Used**:
1. **ImageNet Subset**: 50,000 general images for base training
2. **Danbooru2020small**: Anime/manga-specific images for domain adaptation

**Preprocessing Steps**:
```python
def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Prepare Inception input (299×299×3)
    img_299 = cv2.resize(lab, (299, 299))
    img_299 = inception_resnet_v2.preprocess_input(img_299)
    L_channel = img_299[:,:,0]
    inception_input = np.repeat(L_channel[:,:,np.newaxis], 3, axis=2)
    
    # Prepare encoder input (224×224×1)
    img_224 = cv2.resize(lab, (224, 224))
    encoder_input = img_224[:,:,0]
    
    # Target ab channels
    target_ab = img_224[:,:,1:]
    
    return encoder_input, inception_input, target_ab
```

### Training Configuration

**Hyperparameters**:
- **Batch Size**: 32
- **Learning Rates**: [0.01, 0.0005, 0.0001, 0.00001]
- **Epochs**: 3-100 (model dependent)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error
- **Validation Split**: 10%

**Hardware Requirements**:
- GPU: NVIDIA GPU with CUDA support
- Memory: 8GB+ GPU memory recommended
- Training Time: ~10 minutes per epoch (GPU dependent)

### Model Variants

| Model Name | Learning Rate | Epochs | Purpose |
|------------|---------------|--------|---------|
| Koala_01_20 | 0.01 | 20 | Fast convergence test |
| Koala_0005_40 | 0.0005 | 40 | Balanced training |
| Koala_0005_100 | 0.0005 | 100 | Extended training |
| Koala_0001_50 | 0.0001 | 50 | Fine-tuned learning |
| Koala_00001_20 | 0.00001 | 20 | Conservative learning |

## Experimental Results

### Quantitative Results

| Model | Test Loss | Test Accuracy | Training Time |
|-------|-----------|---------------|---------------|
| Koala_00001_20 | 0.0116 | 0.659 | 3.5 hours |
| Koala_0001_50 | 0.0128 | 0.649 | 8.5 hours |
| Koala_0005_100 | 0.0138 | 0.649 | 17 hours |
| Koala_01_20 | 0.965 | 0.633 | 3.5 hours |

### Key Findings

1. **Learning Rate Impact**: 
   - Very high LR (0.01) leads to instability
   - Very low LR (0.00001) provides best accuracy
   - Moderate LR (0.0005) offers good balance

2. **Training Duration**:
   - More epochs don't always improve results
   - Sweet spot around 20-50 epochs
   - Diminishing returns after 100 epochs

3. **Model Performance**:
   - Best model: Koala_00001_20
   - Stable convergence with conservative learning
   - Good generalization to unseen manga styles

### Qualitative Results

**Strengths**:
- Preserves fine details and textures
- Handles complex hatching patterns
- Maintains artistic consistency
- Works across different manga styles

**Limitations**:
- May struggle with very unusual art styles
- Color choices sometimes conservative
- Processing time dependent on image size

## Usage Guide

### Installation

```bash
git clone https://github.com/your-username/manga-colorization.git
cd manga-colorization
pip install -r requirements.txt
```

### Training New Models

```python
from train import train_model

# Train with custom parameters
model, history = train_model(
    train_folder='./data/train/',
    val_folder='./data/val/',
    test_folder='./data/test/',
    learning_rate=0.0005,
    epochs=50,
    batch_size=32,
    model_name='custom_model'
)
```

### Inference

```python
from inference import process_single_image

# Colorize single image
colorized = process_single_image(
    model_path='models/Koala_00001_20.h5',
    image_path='input/manga_page.jpg',
    output_path='output/colorized_page.jpg'
)
```

### Streamlit Demo

```bash
streamlit run streamlit_demo.py
```

### Batch Processing

```python
from inference import batch_colorize

# Process entire folder
batch_colorize(
    model_path='models/Koala_00001_20.h5',
    input_folder='input_images/',
    output_folder='colorized_output/'
)
```

## Future Improvements

### Technical Enhancements
1. **Attention Mechanisms**: Add attention layers for better feature focusing
2. **Progressive Training**: Implement multi-scale training approach  
3. **Loss Functions**: Experiment with perceptual and adversarial losses
4. **Data Augmentation**: Advanced augmentation for manga-specific patterns

### Model Architecture
1. **Skip Connections**: Add U-Net style skip connections
2. **Multi-Scale Processing**: Process multiple resolutions simultaneously
3. **Color Palette Guidance**: Incorporate color palette constraints
4. **Style Transfer**: Add style-aware colorization capabilities

### Dataset Improvements
1. **Larger Datasets**: Expand training data with more manga styles
2. **Quality Filtering**: Implement automatic quality assessment
3. **Paired Data**: Create more ground-truth color-grayscale pairs
4. **Style Categorization**: Organize data by artistic styles

This technical documentation provides comprehensive insights into the Deep Koalarization approach for manga colorization, covering both theoretical foundations and practical implementation details.