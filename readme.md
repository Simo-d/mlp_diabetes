# Binary Classification of Diabetes Using Multi-Layer Perceptron

A from-scratch implementation of a neural network for diabetes classification using the Pima Indians Diabetes Database.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Mathematical Foundations](#mathematical-foundations)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## 🔍 Overview

This project implements a Multi-Layer Perceptron (MLP) neural network from scratch using only NumPy for binary classification of diabetes. The implementation includes manual coding of forward propagation, backpropagation, and stochastic gradient descent optimization, providing educational insight into neural network fundamentals.

**Key Objectives:**
- Implement configurable MLP without deep learning frameworks
- Apply mathematical formulations for all neural network operations
- Evaluate performance on medical diagnostic classification
- Provide comprehensive educational documentation

## ✨ Features

- **From-scratch implementation** using only NumPy
- **Configurable architecture** with multiple hidden layers
- **Mathematical transparency** with detailed formulations
- **Comprehensive preprocessing** including missing value handling
- **Multiple evaluation metrics** (Accuracy, Precision, Recall, F1-Score)
- **Visualization tools** for training curves and confusion matrices
- **Educational focus** with extensive documentation and assertions

## 📊 Dataset

**Pima Indians Diabetes Database**
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)
- **Samples**: 768 instances
- **Features**: 8 physiological measurements
- **Target**: Binary diabetes classification (0 = non-diabetic, 1 = diabetic)
- **Challenge**: Class imbalance (~35% positive cases) and missing values

### Features Description
1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (weight in kg/(height in m)^2)
7. **DiabetesPedigreeFunction**: Diabetes pedigree function
8. **Age**: Age (years)

## 🚀 Installation

### Prerequisites
- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Simo-d/mlp_diabetes.git
cd mlp_diabetes
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
   - Download `diabetes.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)
   - Place it in the project root directory

## 💻 Usage

### Basic Training and Evaluation

```python
from neural_network_diabetes import NeuralNetwork, load_and_preprocess_data
from sklearn.model_selection import train_test_split

# Load and preprocess data
X, y, scaler = load_and_preprocess_data()

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# Create and train model
layer_sizes = [8, 16, 8, 1]  # Input, Hidden1, Hidden2, Output
nn = NeuralNetwork(layer_sizes, learning_rate=0.01)

# Train the model
train_losses, val_losses, train_acc, val_acc = nn.train(
    X_train, y_train, X_val, y_val, epochs=100, batch_size=32
)

# Make predictions
predictions = nn.predict(X_test)
```

### Running the Complete Pipeline

```bash
python neural_network_diabetes.py
```

This will:
1. Load and preprocess the dataset
2. Train two different architectures
3. Generate evaluation metrics
4. Display confusion matrices and learning curves

### Custom Architecture

```python
# Example with 3 hidden layers
layer_sizes = [8, 32, 16, 8, 1]
nn = NeuralNetwork(layer_sizes, learning_rate=0.01)
```

## 🏗️ Architecture

### Network Structure

The MLP consists of:
- **Input Layer**: 8 neurons (features)
- **Hidden Layers**: Configurable (default: 16, 8 neurons)
- **Output Layer**: 1 neuron (binary classification)

### Activation Functions
- **Hidden Layers**: ReLU activation
- **Output Layer**: Sigmoid activation

### Key Components

```python
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01)
    def forward(self, X)                    # Forward propagation
    def backward(self, X, y, outputs)       # Backpropagation
    def train(self, X, y, X_val, y_val, epochs, batch_size)
    def predict(self, X)                    # Predictions
    def compute_loss(self, y_true, y_pred)  # Binary cross-entropy
```

## 📐 Mathematical Foundations

### Forward Propagation

For layer *l*:
```
Z^[l] = A^[l-1] * W^[l] + b^[l]
A^[l] = g(Z^[l])
```

Where:
- `g(x) = max(0, x)` for hidden layers (ReLU)
- `g(x) = 1/(1 + e^(-x))` for output layer (Sigmoid)

### Loss Function

Binary Cross-Entropy:
```
J = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

### Backpropagation

Output layer:
```
dZ^[L] = A^[L] - y
dW^[L] = 1/m * (A^[L-1])^T * dZ^[L]
db^[L] = 1/m * Σ dZ^[L]
```

Hidden layers:
```
dZ^[l] = (dZ^[l+1] * (W^[l+1])^T) ⊙ ReLU'(Z^[l])
dW^[l] = 1/m * (A^[l-1])^T * dZ^[l]
db^[l] = 1/m * Σ dZ^[l]
```

### Parameter Updates

```
W^[l] ← W^[l] - η * dW^[l]
b^[l] ← b^[l] - η * db^[l]
```

## 📈 Results

### Performance Metrics

| Architecture | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| 2 Hidden Layers (16, 8) | 72% | 65% | 68% | 66% |
| 3 Hidden Layers (32, 16, 8) | 74% | 67% | 70% | 68% |

### Key Findings

- ✅ Successful convergence without overfitting
- ✅ Competitive performance for from-scratch implementation
- ✅ Good generalization with validation accuracy tracking training
- ⚠️ Moderate F1-score due to class imbalance
- 📊 Glucose, BMI, and Age identified as most predictive features

### Visualizations

The implementation generates:
- Training and validation loss curves
- Training and validation accuracy curves
- Confusion matrices
- Classification reports

## 📁 Project Structure

```
mlp_diabetes/
│
├── neural_network_diabetes.py    # Main implementation
├── requirements.txt              # Dependencies
├── README.md                    # This file
├── diabetes.csv                 # Dataset (download separately)
├── paper/                       # Scientific paper
│   ├── diabetes_classification.tex
│   └── diabetes_classification.pdf
├── figures/                     # Generated plots
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── architecture_comparison.png
└── docs/                       # Additional documentation
    ├── mathematical_derivations.md
    └── implementation_details.md
```

## 🔧 Configuration

### Hyperparameters

```python
# Default configuration
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 32
L2_REGULARIZATION = 0.01
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
```

### Customization

You can modify:
- Layer sizes and number of hidden layers
- Learning rate and training epochs
- Batch size for mini-batch SGD
- Regularization strength
- Data split ratios

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- [ ] Implementation of Adam optimizer
- [ ] Dropout regularization
- [ ] Cross-validation framework
- [ ] Additional activation functions
- [ ] Batch normalization
- [ ] Early stopping mechanism
- [ ] Hyperparameter optimization

## 📚 Educational Resources

This project is designed for educational purposes. Refer to:

- **Scientific Paper**: Complete IMRAD-structured analysis in `paper/`
- **Mathematical Derivations**: Detailed formulations in `docs/`
- **Code Comments**: Extensive documentation throughout the implementation
- **Assertions**: Built-in checks for understanding tensor shapes and operations

## 🏥 Medical Disclaimer

This implementation is for educational and research purposes only. It should not be used for actual medical diagnosis. Always consult healthcare professionals for medical advice.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this code in your research or educational work, please cite:

```bibtex
@misc{ajguernoun_diabetes_mlp_2024,
    title={Binary Classification of Diabetes Using Multi-Layer Perceptron: A From-Scratch Implementation},
    author={Mohamed Ajguernoun},
    year={2024},
    url={https://github.com/Simo-d/mlp_diabetes}
}
```

## 📞 Contact

- **Author**: Mohamed Ajguernoun
- **Program**: Masters IAA (Intelligence Artificielle et Applications)

## 🙏 Acknowledgments

- National Institute of Diabetes and Digestive and Kidney Diseases for the dataset
- Kaggle for hosting the Pima Indians Diabetes Database
- The open-source community for inspiration and resources
- Course instructors for guidance and feedback

---

**⭐ Star this repository if you found it helpful!**
