
# COMP3620 - Focal Loss vs Cross-Entropy

## Project Overview

This project investigates the performance of Focal Loss
compared with class-weighted Cross-Entropy on imbalanced
datasets with different levels of label noise.

## Dataset

We use two classes from CIFAR-10:

- Deer: Class 0
- Horse: Class 1

Training:
- 4000 Deer
- 400 Horse

Validation:
- 1000 Deer
- 1000 Horse

Testing:
- 1000 Deer
- 1000 Horse

## Setup

1. Download the CIFAR-10 Python version:
   https://www.cs.toronto.edu/~kriz/cifar.html

2. Extract cifar-10-batches-py into the project folder.

3. Install NumPy:

   pip install numpy

4. Prepare the dataset:

   python prepare_deer_horse.py

5. Generate noisy datasets:

   python create_noisy_datasets.py

6. Verify the datasets:

   python check_noise.py

## Label Noise

- 0%: Clean dataset
- 2%: 88 incorrect labels
- 5%: 220 incorrect labels

## Notebooks

- NB1: Cross-Entropy baseline
- NB2: Focal Loss experiments and label noise
- NB3: Educational notebook
