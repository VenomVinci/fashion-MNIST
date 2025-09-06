
---

# Fashion MNIST Classification with PyTorch

This repository contains a deep learning project using PyTorch to classify images from the **Fashion MNIST dataset**. The model is trained on grayscale images of fashion items and predicts the correct category.

## Dataset

The dataset is a CSV format where:

* **Label**: Class of the fashion item (0–9).
* **Pixels**: 784 columns representing the 28×28 grayscale image.

Example:

| label | pixel1 | pixel2 | ... | pixel784 |
| ----- | ------ | ------ | --- | -------- |
| 0     | 9      | 0      | ... | 0        |
| 1     | 7      | 0      | ... | 0        |

**Classes**:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Model Architecture

The current model is a **Convolutional Neural Network (CNN)**:

* **Conv Layers**:

  * `Conv2d(1, 32, 3)` → `ReLU` → `MaxPool2d(2,2)`
  * `Conv2d(32, 64, 3)` → `ReLU` → `MaxPool2d(2,2)`
* **Fully Connected Layers**:

  * Flatten → `Linear(64*7*7, 128)` → `ReLU` → `Linear(128, 10)`

The CNN takes input images reshaped to `[batch_size, 1, 28, 28]`.

## Training

* **Loss Function**: `CrossEntropyLoss`
* **Optimizer**: `Adam` with learning rate `0.001`
* **Epochs**: 200–500 (configurable)

Example training loop:

```python
for batch_features, batch_labels in train_loader:
    batch_features = batch_features.view(-1, 1, 28, 28)
    outputs = model(batch_features)
    loss = criterion(outputs, batch_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Evaluation

The model is evaluated on the test dataset:

```python
total = 0
correct = 0

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.view(-1, 1, 28, 28)
        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)
        total += batch_labels.shape[0]
        correct += (predicted == batch_labels).sum().item()

print(correct / total)
```

**Test Accuracy**: \~0.8267 (82.67%)

## Usage

1. Clone the repository:

```bash
git clone <repository-url>
cd fashion-mnist-pytorch
```

2. Install dependencies:

```bash
pip install torch torchvision pandas
```

3. Train the model:

```bash
python train.py
```

4. Evaluate the model:

```bash
python evaluate.py
```

## License

This project is open-source under the MIT License.

---

