# Greyscale Image Classification with CNN

This project demonstrates a complete deep learning pipeline to classify grayscale images into one of four categories: **Food**, **Landscape**, **Building**, and **People**. It uses a Convolutional Neural Network (CNN) trained on preprocessed image data with data augmentation techniques to improve model generalization.

---

## 📌 Project Features

- Image preprocessing:
  - Grayscale conversion
  - Histogram equalization
  - Resizing to 224x224
- Data augmentation using TensorFlow's `ImageDataGenerator`
- Multi-class classification using CNN (with Dropout for regularization)
- Visualization of classification results
- Handles training, validation, and testing datasets

---

## 🗂️ Dataset Structure

The dataset follows this directory structure:


Dataset2/
├── training/
│ ├── food/
│ ├── landscape/
│ ├── building/
│ └── people/
├── validation/
│ ├── food/
│ ├── landscape/
│ ├── building/
│ └── people/
└── testing/
├── food/
├── landscape/
├── building/
└── people/
