Installation Guide

This project requires a specific Python environment with OpenMM, PyTorch, and AIMNet2 support. We recommend using Miniconda or Anaconda to manage dependencies.

Prerequisites

Linux (Recommended) or macOS.

NVIDIA GPU (Optional but highly recommended for AIMNet2 performance).

Conda (Miniconda or Anaconda).

Option 1: Quick Install (from environment.yml)

If you have the environment.yml file provided in this repository:

# 1. Create the environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate openmm-cpu


Option 2: Manual Installation

If you prefer to build the environment step-by-step or need to resolve platform-specific conflicts:

1. Create Base Environment

conda create -n openmm-dynamic python=3.11
conda activate openmm-dynamic


2. Install OpenMM and MDTraj

We install OpenMM from conda-forge to ensure we get the latest CUDA-enabled builds.

conda install -c conda-forge openmm mdtraj


3. Install PyTorch

Note: AIMNet2 requires PyTorch. Install the version compatible with your CUDA driver (e.g., CUDA 12.1).

# Example for CUDA 12.x
pip3 install torch torchvision torchaudio


4. Install AIMNet2

AIMNet2 is the core calculator for dynamic charges.

# Install ASE (Atomic Simulation Environment) first
pip install ase

# Install AIMNet2 (Check official repo for latest release/commit)
pip install git+[https://github.com/isayev/AIMNet2.git](https://github.com/isayev/AIMNet2.git)


5. Install Analysis Tools

Required for the analyze_dynamic_properties.py script.

conda install -c conda-forge matplotlib scipy pandas


Verification

To verify that the installation was successful and AIMNet2 is accessible:

import openmm
import torch
from aimnet.calculators import AIMNet2ASE

print(f"OpenMM Version: {openmm.__version__}")
print(f"PyTorch Version: {torch.__version__}")
print("AIMNet2 imported successfully.")

