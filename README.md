
---

# Fashion-MNIST Classification with PyTorch


This project implements a deep learning model using **PyTorch** to classify images from the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).
It was developed and trained in **Google Colab**, achieving strong accuracy on both training and test datasets.

---

## 📌 Project Overview

The goal of this project is to build and evaluate a neural network that classifies grayscale images of clothing items into one of **10 categories** (e.g., T-shirt, trousers, shoes).
The dataset is more challenging than the original MNIST digits dataset, making it a great benchmark for image classification.

---

## 🚀 Technologies Used

* **Python 3.10+**
* **PyTorch** for deep learning
* **Torchvision** for dataset handling and transformations
* **Google Colab** for training and experimentation
* **NumPy / Pandas** for numerical and data handling
* **Matplotlib** for visualization

---

## 📊 Dataset

* **Fashion-MNIST** (60,000 training images, 10,000 test images)
* Each image: `28x28` grayscale
* Classes (10 categories):

  1. T-shirt/top
  2. Trouser
  3. Pullover
  4. Dress
  5. Coat
  6. Sandal
  7. Shirt
  8. Sneaker
  9. Bag
  10. Ankle boot

---

## 🏗️ Model Architecture

The neural network consists of:

* Input Layer: 784 neurons (flattened 28×28 pixels)
* Hidden Layers: Fully connected layers with ReLU activation
* Dropout for regularization
* Output Layer: 10 neurons (one per class) with Softmax

Optimizer & Loss:

* **Optimizer**: Adam
* **Loss Function**: CrossEntropyLoss

---

## ⚡ Training & Results

* **Train Accuracy**: `93.90%`
* **Test Accuracy**: `88.62%`
* Training was done on GPU (Colab) for faster convergence.
* Accuracy and loss curves show good generalization with minimal overfitting.

---

## 🔧 Installation & Usage

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/fashion-mnist-pytorch.git
cd fashion-mnist-pytorch
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook Fashion-MNIST.ipynb
```

Or open directly in **Google Colab**:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YDVmsVD8zkdDh5lqumA_HtIh_WqH10FC?usp=sharing)

---

## 📈 Example Predictions

The model correctly classifies most samples, though challenging categories (e.g., Shirt vs. T-shirt) sometimes cause confusion.

---


## 📜 License

This project is licensed under the **MIT License**.

---


