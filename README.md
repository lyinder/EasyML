# EasyML: Automated Machine Learning System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenML](https://img.shields.io/badge/Data-OpenML-orange.svg)](https://www.openml.org/)

An efficient AutoML system that leverages meta-learning for intelligent model selection and Particle Swarm Optimization (PSO) for hyperparameter tuning.

## 📋 Overview

EasyML is an automated machine learning system designed to simplify the ML modeling process by automatically selecting the best algorithm and optimizing its hyperparameters for a given dataset. Unlike traditional AutoML systems that require extensive computational resources, EasyML achieves superior performance with significantly reduced space and time complexity.

### Key Features

- **Meta-Learning Based Model Selection**: Intelligently recommends the best ML algorithm by analyzing dataset characteristics against a knowledge base of 473+ datasets
- **PSO-Based Hyperparameter Optimization**: Uses Particle Swarm Optimization for efficient parameter tuning
- **Automated Metadata Extraction**: Extracts 64+ statistical and landmarking features from datasets
- **Intelligent Preprocessing**: Automatically selects appropriate scaling methods (StandardScaler/MinMaxScaler)
- **Cached OpenML Integration**: Fast data retrieval with built-in caching mechanism
- **Minimal Human Intervention**: End-to-end automation of the ML pipeline

## 🚀 Performance

EasyML outperforms existing AutoML solutions:
- **SmartML**: Superior accuracy on benchmark datasets
- **Auto-WEKA**: Better efficiency with lower computational overhead

## 🏗️ Architecture

### 1. **Meta-Database Construction**
- Fetches datasets from OpenML repository
- Extracts basic features (instances, features, classes, etc.)
- Computes complex statistical measures (kurtosis, skewness, entropy, etc.)
- Performs landmarking using decision trees, KNN, and Naive Bayes
- Maps best-performing models to dataset characteristics

### 2. **Metadata Extractor**
The `feature_extractor()` function computes 64+ meta-features including:
- **Basic Features**: Number of instances, features, classes, missing values
- **Statistical Features**: Kurtosis, skewness, standard deviation, means
- **Information-Theoretic**: Class entropy, dimensionality
- **Landmarking Features**: Performance metrics from simple classifiers

### 3. **Meta-Learner**
The `meta_learner()` function:
- Computes Euclidean distance between new dataset and knowledge base
- Identifies the most similar dataset
- Recommends the best-performing model for that dataset type

### 4. **PSO Optimizer**
The `PSO()` function optimizes:
- Model-specific hyperparameters (max_depth, cache_size, etc.)
- Random state for reproducibility
- Train/test split ratio

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/lyinder/EasyML.git
cd EasyML

# Install required packages
pip install openml pandas numpy scipy scikit-learn xgboost matplotlib seaborn
```

## 💻 Usage

### Quick Start

```python
from easyml import PSO, feature_extractor, meta_learner

# Run EasyML on your dataset
PSO(
    data='your_dataset.csv',
    target='target_column',
    population=10,
    dimension=3,
    position_min=1,
    position_max=20,
    generation=10,
    fitness_criterion=90
)
```

### Extract Metadata from Your Dataset

```python
# Extract meta-features
metadata = feature_extractor(
    data='your_dataset.csv',
    target='target_column'
)

# Get recommended model
recommended_model = meta_learner(metadata)
print(f"Recommended Model: {recommended_model}")
```

## 📊 Supported Models

- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **Decision Tree Classifier**
- **XGBoost Classifier**
- **Gradient Boosting Classifier**
- **K-Nearest Neighbors**
- **Naive Bayes**

## 🗂️ Dataset Structure

The meta-database includes:
- 473+ datasets from OpenML
- 64+ meta-features per dataset
- Model performance evaluations
- Best model recommendations

### Example Datasets Included
- `abalone2.csv`
- `madelon2.csv`
- `semeion2.csv`
- `yeast.csv`

## 🔧 Key Components

### Meta-Features Extracted

| Category | Features |
|----------|----------|
| **Basic** | Number of instances, features, classes, missing values |
| **Statistical** | Kurtosis (max, min, mean, quartiles), Skewness, Standard deviation |
| **Distribution** | Majority/minority class percentages, class entropy |
| **Complexity** | Dimensionality, number of binary/symbolic/numeric features |
| **Landmarking** | J48, REPTree, RandomTree, KNN, Naive Bayes error rates and Kappa scores |

### PSO Parameters

- **Population**: Number of particles in swarm
- **Dimension**: Number of hyperparameters to optimize
- **Position Range**: Min/max values for parameters
- **Generation**: Number of optimization iterations
- **Fitness Criterion**: Target accuracy threshold

## 📈 Workflow

```mermaid
graph LR
    A[Input Dataset] --> B[Extract Metadata]
    B --> C[Meta-Learner]
    C --> D[Select Best Model]
    D --> E[PSO Optimization]
    E --> F[Optimized Pipeline]
```

## 🧪 Research Methodology

EasyML was developed using:
- **Design Science Research (DSR)**: For system design and development
- **CRISP-DM**: For data mining process structure

## 📝 Citation

If you use EasyML in your research, please cite:

```bibtex
@mastersthesis{easyml2024,
  title={EasyML: An Efficient AutoML System Using Meta-Learning and Particle Swarm Optimization},
  author={[Your Name]},
  year={2024},
  school={[Your University]}
}
```

## 🔬 Future Work

- Expand model library with deep learning algorithms
- Implement multi-objective optimization
- Add support for regression tasks
- Develop web-based interface
- Integrate with cloud platforms

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenML for providing the dataset repository
- scikit-learn community for ML algorithms
- Research inspired by SmartML and Auto-WEKA

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact lyindernswale@gmail.com.

---

**Note**: This is a research prototype developed as part of a Master's thesis. Performance may vary depending on dataset characteristics and computational resources.
