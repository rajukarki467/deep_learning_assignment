#deep_learning_assignment
# Deep Learning Assignment: Variational Autoencoders (VAEs)

A PyTorch implementation of a **Variational Autoencoder (VAE) from scratch** for learning continuous probabilistic latent representations and generating images from learned latent distributions.

The project uses **MNIST** and **Fashion-MNIST** to study image reconstruction, latent-space representations, and generative sampling.

---

## 📌 Overview

A **Variational Autoencoder (VAE)** is a generative deep learning model that combines:

* Neural network-based encoding and decoding
* Probabilistic latent representations
* The reparameterization trick
* Reconstruction learning
* KL-divergence regularization
* Generative sampling

Unlike a standard Autoencoder, which maps an input to a fixed latent vector, a VAE learns a probability distribution:

$$
q(z|x) = \mathcal{N}(\mu, \sigma^2)
$$

This structured latent space allows the decoder to generate new samples by sampling from the learned distribution.

### Main Goals

* Implement a VAE using PyTorch.
* Understand the encoder-decoder architecture.
* Implement the reparameterization trick.
* Implement reconstruction and KL-divergence losses.
* Train the model on MNIST and Fashion-MNIST.
* Reconstruct input images.
* Generate new images from random latent vectors.
* Visualize the learned latent space.
* Investigate the effect of model and training hyperparameters.
* Maintain reproducible experiments using configuration files and fixed random seeds.

---

## 🧠 VAE Architecture

The model consists of an **Encoder**, **Latent Space**, and **Decoder**.

```text
                    Variational Autoencoder

                         Input Image
                              │
                              ▼
                         ┌─────────┐
                         │ Encoder │
                         └────┬────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                   μ                 log(σ²)
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    Reparameterization
                              │
                              ▼
                    z = μ + σ × ε
                    ε ~ N(0, I)
                              │
                              ▼
                         ┌─────────┐
                         │ Decoder │
                         └────┬────┘
                              │
                              ▼
                    Reconstructed Image
```

### Encoder

The encoder receives an input image `x` and estimates:

* `μ` — mean of the latent distribution
* `log_var` — logarithm of the variance

The standard deviation is calculated as:

```text
σ = exp(0.5 × log_var)
```

### Reparameterization Trick

To allow gradients to flow through the stochastic sampling operation:

```text
z = μ + σ × ε
```

where:

```text
ε ~ N(0, I)
```

### Decoder

The decoder receives the sampled latent vector `z` and reconstructs the input image:

```text
z → Decoder → x̂
```

---

## 📐 VAE Objective

The VAE minimizes two main components.

### Reconstruction Loss

Measures the difference between the original image `x` and reconstructed image `x̂`.

Depending on the experiment, the implementation can use:

* Binary Cross-Entropy (BCE)
* Mean Squared Error (MSE)

```text
L_reconstruction = Loss(x̂, x)
```

### KL Divergence

The KL term regularizes the learned latent distribution toward a standard normal distribution:

```text
p(z) = N(0, I)
```

For a diagonal Gaussian:

```text
L_KL = -1/2 Σ(1 + log_var - μ² - exp(log_var))
```

### Total Loss

```text
L_VAE = L_reconstruction + β × L_KL
```

For a standard VAE:

```text
β = 1
```

The `β` parameter can also be varied to investigate its effect on reconstruction and latent-space organization.

---

## 📊 Datasets

### MNIST

* Handwritten digits: `0–9`
* Grayscale images
* Image size: `28 × 28`
* Single image channel
* 10 classes

### Fashion-MNIST

* Clothing and fashion images
* Grayscale images
* Image size: `28 × 28`
* Single image channel
* 10 classes

The dataset should be selected through the experiment configuration rather than modifying the model source code.

---

## 📁 Project Structure

```text
deep-learning-vae/
│
├── configs/
│   └── *.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── experiments/
│   └── <experiment_name>/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── latent_space_visualization.ipynb
│   └── results_analysis.ipynb
│
├── report/
│   ├── figures/
│   ├── report.tex
│   └── final_report.pdf
│
├── results/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── vae/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── loss.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── generate.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       ├── metrics.py
│       ├── visualization.py
│       └── logger.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Directory Description

| Directory      | Purpose                                                                |
| -------------- | ---------------------------------------------------------------------- |
| `configs/`     | YAML configuration files for experiments                               |
| `data/`        | Raw and processed datasets                                             |
| `experiments/` | Experiment-specific configurations and outputs                         |
| `notebooks/`   | Exploration, visualization, and analysis                               |
| `report/`      | LaTeX report and report figures                                        |
| `results/`     | Checkpoints, logs, figures, and tables                                 |
| `src/vae/`     | VAE model, loss, training, evaluation, and generation                  |
| `src/utils/`   | Reusable utilities for reproducibility, metrics, plotting, and logging |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd deep-learning-vae
```

### 2. Create a Virtual Environment

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet:

```bash
pip install torch torchvision pyyaml matplotlib numpy tqdm
```

---

## 📦 Dependencies

| Package         | Purpose                                          |
| --------------- | ------------------------------------------------ |
| **PyTorch**     | Model implementation and training                |
| **Torchvision** | MNIST/Fashion-MNIST datasets and transformations |
| **NumPy**       | Numerical computation                            |
| **Matplotlib**  | Visualization                                    |
| **PyYAML**      | YAML configuration management                    |
| **tqdm**        | Training progress bars                           |

---

## 🚀 Training

The training pipeline follows:

```text
Dataset
   ↓
DataLoader
   ↓
Encoder
   ↓
μ, log_var
   ↓
Reparameterization
   ↓
Latent Vector z
   ↓
Decoder
   ↓
Reconstruction
   ↓
VAE Loss
   ↓
Backpropagation
   ↓
Model Update
```

A typical training command is:

```bash
python -m src.vae.train --config configs/mnist.yaml
```

For Fashion-MNIST:

```bash
python -m src.vae.train --config configs/fashion_mnist.yaml
```

> Update the commands if your final implementation uses a different entry point.

---

## 🔍 Evaluation

The trained model should be evaluated using both **quantitative metrics** and **visual analysis**.

### Image Reconstruction

Compare:

```text
Original Image → Encoder → Latent z → Decoder → Reconstruction
```

Save representative reconstruction grids in the experiment output directory.

### Image Generation

New images can be generated without an input image:

```text
z ~ N(0, I)
      ↓
   Decoder
      ↓
Generated Image
```

Example:

```bash
python -m src.vae.generate --config configs/mnist.yaml
```

### Latent-Space Analysis

For a 2-dimensional latent space:

```text
latent_dim = 2
```

encoded samples can be visualized in two dimensions to investigate:

* Class clustering
* Latent-space structure
* Continuity
* Separation between classes
* Smooth transitions between generated samples

---

## 🧪 Experiments

The project is designed to support controlled experiments through YAML configuration files.

### 1. Dataset Comparison

Compare VAE performance on:

* MNIST
* Fashion-MNIST

### 2. Latent Dimension

Example configurations:

```text
latent_dim = 2
latent_dim = 8
latent_dim = 16
latent_dim = 32
```

Investigate how latent dimensionality affects:

* Reconstruction quality
* Latent-space structure
* Generation quality

### 3. β-VAE Experiments

Compare different KL-divergence weights:

```text
β = 0.1
β = 0.5
β = 1.0
β = 2.0
```

### 4. Training Hyperparameters

Investigate the effect of:

* Learning rate
* Batch size
* Number of epochs
* Hidden-layer dimensions
* Optimizer settings

All experiment configurations should be saved with the corresponding results.

---

## 📈 Results

The project should generate and analyze:

* Training loss curves
* Reconstruction-loss curves
* KL-divergence curves
* Original vs reconstructed images
* Randomly generated samples
* Latent-space visualizations
* Experiment comparison tables

Example result organization:

```text
experiments/
└── mnist_latent16/
    ├── config.yaml
    ├── checkpoints/
    │   └── best_model.pt
    ├── logs/
    │   └── training.log
    ├── figures/
    │   ├── reconstructions.png
    │   ├── generated_samples.png
    │   ├── loss_curve.png
    │   └── latent_space.png
    └── metrics.json
```

---

## 🔁 Reproducibility

Experiments should use fixed random seeds to improve reproducibility.

The seed should be applied to:

* Python
* NumPy
* PyTorch
* CUDA, when available

Example:

```yaml
seed: 42
```

Important experiment information should also be recorded:

```text
Python version
PyTorch version
CUDA availability
GPU/CPU
Dataset
Latent dimension
Learning rate
Batch size
Epochs
β
Random seed
```

---

## 📋 Expected Outcomes

After training, the VAE should be able to:

* Reconstruct input images.
* Learn meaningful continuous latent representations.
* Generate new images from random latent vectors.
* Produce smooth transitions in latent space.
* Demonstrate the effect of KL regularization.
* Show how latent dimensionality affects model performance.
* Provide meaningful visual and quantitative results.

---

## 📝 Assignment Deliverables

The final project should contain:

### Source Code

* VAE architecture
* Encoder
* Decoder
* Reparameterization trick
* VAE loss
* Training pipeline
* Evaluation pipeline
* Image generation
* Visualization utilities

### Configuration

YAML configuration files containing experiment parameters.

### Results

* Reconstruction images
* Generated samples
* Training curves
* KL-divergence curves
* Latent-space plots
* Comparison tables

### Notebooks

Exploratory and visualization notebooks, where applicable.

### Report

The final report should cover:

1. Introduction
2. Background
3. VAE architecture
4. Mathematical formulation
5. Dataset
6. Implementation
7. Experimental setup
8. Results
9. Discussion
10. Limitations
11. Conclusion
12. References

---

## 🛡️ Git and Data Management

Large datasets, virtual environments, checkpoints, and generated outputs should not be committed to Git.

The `.gitignore` should include at least:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environments
.venv/
venv/
env/
ENV/

# Environment variables
.env
.env.*

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# Dataset
data/raw/
data/processed/

# Experiment outputs
experiments/*/checkpoints/
experiments/*/logs/
experiments/*/figures/

# Results
results/checkpoints/
results/logs/
results/figures/
results/tables/

# Models
checkpoints/
models/

# Generated outputs
outputs/

# Report outputs
report/figures/
report/final_report.pdf

# OS files
.DS_Store
Thumbs.db
```

---

## 📚 References

The final report should include references for the theoretical background and datasets, particularly:

* Kingma, D. P., & Welling, M. — *Auto-Encoding Variational Bayes*.
* MNIST dataset documentation.
* Fashion-MNIST dataset documentation.
* PyTorch documentation.
* Torchvision documentation.

---

## 👤 Author

**Raju Karki**

Deep Learning Assignment — Variational Autoencoders

---

## 📄 License

This project is developed for **educational and academic purposes**.


### PyTorch Installation Note

This project requires a PyTorch wheel compiled for CUDA ≤ your driver version.
Check your driver's max CUDA with `nvidia-smi` (top-right corner), then install:

    # Example for driver supporting CUDA 12.5 (or lower):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124