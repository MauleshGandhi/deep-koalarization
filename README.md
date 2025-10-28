# Deep Koalarization: Manga Colorization

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)

An implementation of automatic manga colorization using Deep Convolutional Neural Networks and Inception-ResNet-v2 architecture, based on the "Deep Koalarization" approach.

## 🎨 Live Demo

**[Try the live demo here!](https://your-streamlit-app-url.streamlit.app)**

Upload your grayscale manga images and see them colorized in real-time using our trained models.

## 📖 About

This project implements an advanced image colorization system specifically designed for manga and anime-style artwork. Unlike traditional colorization methods that rely on intensity continuity, our approach handles the complex patterns, hatching, and screening commonly found in manga.

### Key Features

- **Intelligent Colorization**: Preserves fine details like hatching and screening patterns
- **Multiple Models**: 7 different trained models with various hyperparameters
- **LAB Color Space**: Utilizes perceptually uniform color representation
- **Streamlit Interface**: User-friendly web application for testing
- **Batch Processing**: Colorize multiple images efficiently

## 🏗️ Architecture

Our model combines:
- **Encoder-Decoder CNN**: Custom architecture for feature learning
- **Inception-ResNet-v2**: Pre-trained feature extractor for semantic understanding
- **Fusion Layer**: Combines low-level and high-level features
- **LAB Color Space**: Separates luminance from chrominance for better colorization

```
Input (L channel) → Encoder → Fusion → Decoder → Output (ab channels)
                      ↑
            Inception-ResNet-v2 Features
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/manga-colorization.git
cd manga-colorization
pip install -r requirements.txt
```

### Run the Demo

```bash
streamlit run streamlit_demo.py
```

### Train Your Own Model

```bash
python train.py
```

### Colorize Images

```bash
python inference.py
```

## 📊 Model Performance

| Model | Learning Rate | Epochs | Test Accuracy | Test Loss |
|-------|---------------|--------|---------------|-----------|
| Koala_00001_20 | 0.00001 | 20 | 0.659 | 0.0116 |
| Koala_0001_50 | 0.0001 | 50 | 0.649 | 0.0128 |
| Koala_0005_100 | 0.0005 | 100 | 0.649 | 0.0138 |
| Koala_01_20 | 0.01 | 20 | 0.633 | 0.965 |

## 📁 Project Structure

```
manga-colorization/
├── streamlit_demo.py       # Interactive web demo
├── train.py               # Training script
├── inference.py           # Inference and evaluation
├── models/               # Trained model files (.h5)
│   ├── Koala_01_20.h5
│   ├── Koala_0005_40.h5
│   └── ...
├── datasets/             # Training data
├── results/              # Output images
└── docs/                # Documentation and paper
```

## 🔬 Technical Details

### Dataset
- **ImageNet**: 50,000 general images for base training
- **Danbooru2020small**: Anime/manga-specific dataset for fine-tuning

### Architecture Components
1. **Encoder Network**: 8-layer CNN for feature extraction from L channel
2. **Pre-trained Branch**: Inception-ResNet-v2 for semantic features
3. **Fusion Layer**: Combines spatial and semantic information
4. **Decoder Network**: Upsampling layers to reconstruct ab channels

### Color Space
- **Input**: LAB color space L channel (luminance)
- **Output**: LAB color space ab channels (chrominance)
- **Advantage**: Perceptually uniform color representation

## 🎯 Results

Our model achieves superior colorization results on manga images:
- Preserves fine details and textures
- Handles complex hatching patterns
- Maintains artistic style consistency
- Works on various manga art styles

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Web Interface**: Streamlit
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib

## 📚 Research Background

This project is based on the paper:
> "Deep Koalarization: Image Colorization using CNNs and Inception-Resnet-v2"

Key innovations:
- Fusion of CNN encoder and pre-trained feature extractor
- LAB color space optimization
- Manga-specific training approach

## 🔧 Training Details

### Hyperparameters
- **Batch Size**: 32
- **Learning Rates**: 0.01, 0.0005, 0.0001, 0.00001
- **Epochs**: 3-100 (varies by model)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error

### Data Preprocessing
1. Convert RGB to LAB color space
2. Resize images to 224×224 (encoder) and 299×299 (Inception)
3. Normalize using Inception-ResNet-v2 preprocessing
4. Extract L channel for input, ab channels for targets

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{koalarization2024,
  title={Deep Koalarization: Manga Colorization using CNNs and Inception-ResNet-v2},
  author={Mathur, Anish and Gandhi, Maulesh and Kondooru, Pavan},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Anish Mathur** - IIIT Hyderabad
- **Maulesh Gandhi** - IIIT Hyderabad  
- **Pavan Kondooru** - IIIT Hyderabad

## 🙏 Acknowledgments

- Original Deep Koalarization paper authors
- ImageNet and Danbooru2020 dataset contributors
- TensorFlow and Keras communities
- IIIT Hyderabad for computational resources

---

⭐ **Star this repository if you found it helpful!**