## MNIST Digit Classification
This project implements a high-accuracy digit recognition pipeline using an ensemble of three custom-designed CNN models trained on the MNIST dataset. The final predictions are generated using soft voting, achieving 99.60% validation accuracy and 99.550% on [Kaggle](https://www.kaggle.com/competitions/digit-recognizer).
---

## 🧠 Model Architectures
 - 📦 Model 1 - Basic CNN with Conv → ReLU → BN → Dropout → FC

 - 🧱 Model 2 - Deep CNN with stacked Conv-BN-ReLU blocks and adaptive pooling

 - 🌀 Model 3 - Mix of dilated and standard convolutions

## 📊 Performance
Model	Validation Accuracy
 - Model 1	- 99.31%
 - Model 2	- 99.48%
 - Model 3	- 99.40%
 - Ensemble (Soft Voting)	- 99.60% ✅
 - Kaggle Score	- 99.550% ✅

## 🛠️ How to Use
### 1. Clone the Repository

```bash
git clone https://github.com/AbhijatSahu/MNIST-Digit-Classification

```
### 2. Install Dependencies
```
pip install -r requirements.txt
```

## 🖼️ Data
Training data: train.csv (images + labels)

Test data: test.csv (images only)

Source: Kaggle Digit Recognizer

## 🧪 Key Techniques
Softmax outputs for soft voting

Early stopping to prevent overfitting

Learning rate decay via StepLR

Efficient inference using torch.no_grad() and .to(device)
