# Installation Guide

This project requires a specific Python environment with OpenMM, PyTorch, and AIMNet2 support. We recommend using **Miniconda** or **Anaconda** to manage dependencies.

## Prerequisites

- Linux (recommended) or macOS  
- NVIDIA GPU (optional but highly recommended for AIMNet2 performance)  
- Conda (Miniconda or Anaconda)  

---

## Option 1: Quick Install (from `environment.yml`)

If you have the `environment.yml` file provided in this repository:

```bash
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate openmm-cpu
```

---

## Option 2: Manual Installation

If you prefer to build the environment step-by-step or need to resolve platform-specific conflicts:

### 1. Create Base Environment

```bash
conda create -n openmm-dynamic python=3.11
conda activate openmm-dynamic
```

### 2. Install OpenMM and MDTraj

Install OpenMM from `conda-forge` to ensure you get recent, compatible builds:

```bash
conda install -c conda-forge openmm mdtraj
```

### 3. Install PyTorch

> **Note**  
> AIMNet2 requires PyTorch. Install the version compatible with your CUDA driver (for example, CUDA 12.1).

Example for CUDA 12.x:

```bash
pip install torch torchvision torchaudio
```

(See the official PyTorch website for exact install commands for your OS/CUDA combination.)

### 4. Install AIMNet2

AIMNet2 is the core calculator for dynamic charges.

First install ASE (Atomic Simulation Environment):

```bash
pip install ase
```

Then install AIMNet2 (check the official repository for the latest release/commit):

```bash
pip install git+https://github.com/isayev/AIMNet2.git
```

### 5. Install Analysis Tools

These libraries are required for the `analyze_dynamic_properties.py` and other analysis scripts:

```bash
conda install -c conda-forge matplotlib scipy pandas
```

---

## Verification

To verify that the installation was successful and AIMNet2 is accessible, start Python in the environment and run:

```python
import openmm
import torch
from aimnet.calculators import AIMNet2ASE

print(f"OpenMM Version: {openmm.__version__}")
print(f"PyTorch Version: {torch.__version__}")
print("AIMNet2 imported successfully.")
```

If no errors are raised and the versions are printed, your installation is complete.
