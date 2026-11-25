
# COMP3620 - Focal Loss vs Cross-Entropy

## 1. Project Overview

This project investigates the performance of Focal Loss compared with
class-weighted Cross-Entropy Loss on imbalanced datasets.

We aim to investigate how the focusing parameter (gamma) affects
minority-class classification and whether Focal Loss remains effective
when training data contains incorrect labels.

### Research Question

How does the focusing parameter in Focal Loss affect minority-class
classification performance compared with class-weighted Cross-Entropy,
and how robust are these effects to label noise?

### Research Objectives

- Understand the theoretical differences between Cross-Entropy and Focal Loss.
- Investigate the effect of different gamma values.
- Evaluate minority-class classification performance.
- Investigate the effect of incorrect training labels.
- Explain why Focal Loss performs better or worse under different conditions.

Gamma is our main experimental variable, while label noise is our
sensitivity check.

---

## 2. Dataset

We use two classes from the CIFAR-10 dataset:

- Deer (original class 4, remapped to 0)
- Horse (original class 7, remapped to 1)

These classes were selected because they share some visual similarities,
making the classification task potentially more challenging.

### Dataset Distribution

| Dataset | Deer | Horse | Total |
|---------|------|-------|-------|
| Training | 4000 | 400 | 4400 |
| Validation | 1000 | 1000 | 2000 |
| Testing | 1000 | 1000 | 2000 |

The training dataset has a 10:1 class imbalance.

- Deer is the majority class.
- Horse is the minority class.
- Validation and testing datasets remain balanced.
- The dataset selection seed is 2026.
- The official CIFAR-10 test set is kept separate from training and validation.

### Label Noise

Three versions of the training dataset have been created.

| Noise Level | Deer to Horse | Horse to Deer | Incorrect Labels |
|-------------|---------------|---------------|------------------|
| 0% | 0 | 0 | 0 |
| 2% | 44 | 44 | 88 |
| 5% | 110 | 110 | 220 |

The same images are used in all three datasets.
Only the training labels are modified.

Equal numbers of labels are swapped between the two classes to maintain
the original 10:1 class distribution.

However, this means that a larger proportion of minority-class labels
are corrupted. This should be considered when analysing the results.

Validation and testing labels remain unchanged.

---

## 3. Project Setup

### Step 1 - Clone the Repository

```bash
git clone https://github.com/MohanadQaid/Focal-loss-project.git
cd Focal-loss-project
```

### Step 2 - Download CIFAR-10

Download the CIFAR-10 Python version from:

https://www.cs.toronto.edu/~kriz/cifar.html

Extract the `cifar-10-batches-py` folder into the project directory.

### Step 3 - Install Dependencies

```bash
pip install numpy matplotlib torch torchvision scikit-learn jupyter
```

Use the university's dl2026 environment when working on the lab machines.

### Step 4 - Prepare the Dataset

```bash
python prepare_deer_horse.py
```

### Step 5 - Generate Noisy Datasets

```bash
python create_noisy_datasets.py
```

### Step 6 - Verify the Datasets

```bash
python check_noise.py
```

Expected output:

```text
Same images (2%): True
Same images (5%): True
Wrong labels (2%): 88
Wrong labels (5%): 220
```

The original and generated datasets are excluded from GitHub because
of their size. Each member can generate the same datasets locally
using the provided scripts.

---

## 4. Research Plan

### Stage 1 - Research Design

- Finalise the research question.
- Understand the theory behind Cross-Entropy and Focal Loss.
- Review relevant literature.
- Establish our hypotheses before running the main experiments.
- Complete the challenge statement.
- Agree on the CNN architecture and experimental settings.

### Stage 2 - NB1: Cross-Entropy Baseline

- Load the clean imbalanced training dataset.
- Implement a simple Convolutional Neural Network (CNN).
- Implement class-weighted Cross-Entropy Loss.
- Select suitable hyperparameters using the validation dataset.
- Train the model using at least three different random seeds.
- Evaluate the baseline model.
- Calculate the mean and standard deviation of the results.
- Save the results and any required model checkpoints.

### Stage 3 - NB2: Focal Loss Experiments

- Implement Focal Loss.
- Verify that Focal Loss with gamma = 0 is equivalent to
  class-weighted Cross-Entropy using the same weights and reduction.
- Investigate different gamma values.
- Use the same CNN architecture and dataset as NB1.
- Tune nuisance hyperparameters fairly for each experimental condition.
- Train each condition using at least three random seeds.
- Compare the results against the baseline.
- Investigate and explain the differences in performance.

### Stage 4 - NB2: Label Noise Sensitivity Check

- Load the prepared datasets containing 2% and 5% label noise.
- Train selected gamma conditions on each noisy dataset.
- Keep validation and testing datasets unchanged.
- Compare performance against the clean-data experiments.
- Investigate whether increasing gamma makes the model more
  sensitive to incorrect labels.
- Analyse the behaviour of correctly and incorrectly labelled examples.
- Discuss limitations and alternative explanations.

### Stage 5 - NB3: Educational Notebook

- Explain Cross-Entropy and Focal Loss.
- Explain the focusing parameter and class weights.
- Demonstrate the problem of class imbalance.
- Present and explain our experimental results.
- Explain the effect of label noise.
- Include at least one interactive exercise or prediction.
- Create a notebook suitable for approximately 30 minutes of learning.

### Stage 6 - Final Submission

- Finalise all three notebooks.
- Complete the challenge statement.
- Complete the meeting log and AI-use log.
- Test the notebooks on the university lab environment.
- Record the group presentation.
- Submit all required files.

---

## 5. Proposed CNN Architecture

The same CNN architecture will be used across all experimental
conditions.

| Layer | Output Shape |
|-------|--------------|
| Input | 3 x 32 x 32 |
| Convolution + ReLU | 32 x 32 x 32 |
| Max Pooling | 32 x 16 x 16 |
| Convolution + ReLU | 64 x 16 x 16 |
| Max Pooling | 64 x 8 x 8 |
| Flatten | 4096 |
| Fully Connected + ReLU | 128 |
| Output Layer | 2 |

The convolutional layers will initially use a kernel size of 3,
stride of 1 and padding of 1.

The pooling layers will use a kernel size and stride of 2.

The output layer produces two logits representing Deer and Horse.

The architecture is provisional and should be confirmed through
a pilot experiment before running the main experiments.

---

## 6. Proposed Experimental Settings

| Parameter | Proposed Value |
|-----------|----------------|
| Framework | PyTorch |
| Model | CNN |
| Optimiser | Adam |
| Batch Size | 64 |
| Epochs | 30 |
| Learning Rates | 0.0001, 0.001, 0.01 |
| Random Seeds | 42, 123, 456 |
| Deer Class Weight | 0.55 |
| Horse Class Weight | 5.50 |
| Gamma Values | 0, 1, 2, 4 |
| Noise Levels | 0%, 2%, 5% |
| Primary Metric | Minority-Class F1-score |

These settings are proposed starting points and are subject to
group discussion and validation-based hyperparameter tuning.

The same architecture, data splits, class weights and evaluation
procedure must be used throughout the experiments.

The test dataset must not be used for hyperparameter tuning.

---

## 7. Experimental Matrix

### Main Experiment and Sensitivity Check

| Loss Function | Gamma | 0% Noise | 2% Noise | 5% Noise |
|---------------|-------|----------|----------|----------|
| Weighted Cross-Entropy | 0 | 3 runs | 3 runs | 3 runs |
| Focal Loss | 1 | 3 runs | - | - |
| Focal Loss | 2 | 3 runs | 3 runs | 3 runs |
| Focal Loss | 4 | 3 runs | 3 runs | 3 runs |

### Total Training Runs

- Clean-data experiments: 12 runs.
- Label-noise experiments: 18 runs.
- Total: 30 final training runs.

Additional pilot and hyperparameter tuning runs are not included
in this total.

The clean weighted Cross-Entropy results from NB1 will be reused
in NB2.

---

## 8. Evaluation Metrics

We will evaluate the models using:

- Overall Accuracy
- Balanced Accuracy
- Minority-Class Precision
- Minority-Class Recall
- Minority-Class F1-score
- Confusion Matrix
- Training Loss
- Validation Loss
- Mean and Standard Deviation across random seeds

Our primary evaluation metric will be the Horse F1-score because
Horse is the minority class.

### Planned Visualisations

- Gamma vs Minority-Class F1-score.
- Gamma vs Minority-Class Recall.
- Gamma vs Minority-Class Precision.
- Label Noise vs Minority-Class F1-score.
- Training and validation curves.
- Confusion matrices.
- Loss contributions of correctly and incorrectly labelled examples.

The results should include variability across random seeds.

We should explain the results rather than simply report which
model achieved the highest accuracy.

---

## 9. Work Allocation

The project will be divided between five members.

### Member 1 - Dataset Preparation and Research Design

Responsibilities:

- Download and prepare the CIFAR-10 dataset.
- Extract Deer and Horse classes.
- Create the imbalanced training dataset.
- Create validation and testing datasets.
- Generate datasets containing different levels of label noise.
- Verify the generated datasets.
- Document the dataset preparation process.
- Coordinate the research design and challenge statement.
- Maintain the shared repository and support project integration.

Deliverables:

- prepare_deer_horse.py
- create_noisy_datasets.py
- check_noise.py
- Dataset documentation
- Challenge statement coordination

Status: Dataset preparation and verification completed.

### Member 2 - CNN and Cross-Entropy Baseline

Responsibilities:

- Implement the CNN architecture.
- Implement the dataset loading process for PyTorch.
- Implement the training and evaluation functions.
- Implement class-weighted Cross-Entropy.
- Perform hyperparameter tuning.
- Train the baseline using three random seeds.
- Evaluate and analyse baseline performance.
- Save the results and required checkpoints.

Deliverables:

- NB1_Baseline.ipynb
- Shared CNN and training functions
- Baseline experimental results
- Baseline visualisations

### Member 3 - Focal Loss and Gamma Experiments

Responsibilities:

- Research the mathematical formulation of Focal Loss.
- Implement Focal Loss.
- Verify the implementation against Cross-Entropy when gamma = 0.
- Investigate gamma values of 1, 2 and 4.
- Tune hyperparameters fairly for each condition.
- Train each condition using three random seeds.
- Compare the results against the baseline.
- Explain the effect of changing gamma.

Deliverables:

- NB2 main experimental section
- Focal Loss implementation
- Gamma experimental results
- Comparison graphs

### Member 4 - Label Noise and Results Analysis

Responsibilities:

- Use the prepared noisy datasets.
- Investigate the effect of 2% and 5% label noise.
- Train selected gamma conditions using three random seeds.
- Compare clean-data and noisy-data performance.
- Calculate mean and standard deviation.
- Investigate the effect of incorrectly labelled examples.
- Analyse and explain the experimental findings.
- Discuss limitations and alternative explanations.

Deliverables:

- NB2 sensitivity check section
- Label noise experimental results
- Statistical analysis
- Noise comparison graphs

### Member 5 - Educational Notebook and Theory

Responsibilities:

- Research the theoretical foundations of both loss functions.
- Review relevant research papers.
- Explain class imbalance and label noise.
- Develop the educational notebook.
- Create interactive examples and exercises.
- Explain the experimental findings.
- Coordinate the video presentation.
- Support final notebook integration and testing.

Deliverables:

- NB3_Educational.ipynb
- Educational examples and exercises
- Theoretical explanations
- Final reproducibility checks

### Shared Responsibilities

All group members must:

- Attend group meetings.
- Contribute to the project discussion.
- Review each other's work.
- Maintain their own AI-use records.
- Understand the experimental results.
- Contribute to the final report and conclusions.
- Participate in the video presentation.

---

## 10. GitHub Collaboration

Each member should work on a separate branch.

| Member | Suggested Branch |
|--------|------------------|
| Member 1 | feat/dataset |
| Member 2 | feat/baseline |
| Member 3 | feat/focal-loss |
| Member 4 | feat/label-noise |
| Member 5 | feat/educational |

### Create a Branch

```bash
git switch main
git pull origin main
git switch -c feat/your-branch-name
```

### Save and Push Your Work

```bash
git add .
git commit -m "Description of changes"
git push -u origin feat/your-branch-name
```

Create a pull request on GitHub when your work is ready.

Another group member should review the changes before merging
them into the main branch.

### Collaboration Rules

- Do not push unfinished experimental code directly to main.
- Pull the latest changes regularly.
- Use meaningful commit messages.
- Do not upload large datasets unnecessarily.
- Document important changes.
- Agree on shared function names and interfaces.
- Avoid editing the same notebook simultaneously.
- Keep experimental results reproducible.

Members 2, 3 and 4 should agree on shared model, training and
evaluation functions before starting their experiments.

---

## 11. Proposed Project Structure

```text
Focal-loss-project/
|
|-- cifar-10-batches-py/
|
|-- deer_horse_data/
|   |-- train.npz
|   |-- validation.npz
|   |-- test.npz
|   |-- split_indices.npz
|   |
|   |-- noisy_datasets/
|       |-- train_noise_0pct.npz
|       |-- train_noise_2pct.npz
|       |-- train_noise_5pct.npz
|
|-- utils/
|   |-- model.py
|   |-- data_loader.py
|   |-- training.py
|   |-- evaluation.py
|
|-- results/
|-- checkpoints/
|
|-- prepare_deer_horse.py
|-- create_noisy_datasets.py
|-- check_noise.py
|
|-- NB1_Baseline.ipynb
|-- NB2_Experiment.ipynb
|-- NB3_Educational.ipynb
|
|-- challenge_statement.md
|-- project_log.md
|-- README.md
|-- .gitignore
```

The main experimental code and explanations should remain in the
notebooks, as required by the coursework.

Utility files should contain reusable supporting functions.

---

## 12. Project Timeline

| Date | Task |
|------|------|
| 25-26 November | Finalise research design, dataset and responsibilities |
| 27-29 November | Baseline CNN and Focal Loss implementation |
| 30 November | Initial experiments and formative feedback preparation |
| 1 December | Formative feedback deadline |
| 2-3 December | Feedback sessions |
| 4-6 December | Complete gamma and label noise experiments |
| 7-8 December | Complete analysis and educational notebook |
| 9 December | Notebook testing and project logs |
| 10 December | Record presentation and final checks |
| 11 December | Final submission before 23:59 |

Internal deadlines can be adjusted following group discussions.

---

## 13. Project Progress

### Dataset Preparation

- [x] Download CIFAR-10
- [x] Extract Deer and Horse classes
- [x] Create 10:1 class imbalance
- [x] Create training dataset
- [x] Create validation dataset
- [x] Create testing dataset
- [x] Generate 0% label noise dataset
- [x] Generate 2% label noise dataset
- [x] Generate 5% label noise dataset
- [x] Verify datasets

### Project Management

- [x] Create GitHub repository
- [x] Push dataset preparation scripts
- [ ] Confirm all members have repository access
- [ ] Finalise work allocation
- [ ] Finalise challenge statement
- [ ] Establish research hypotheses

### NB1 - Baseline

- [ ] Implement CNN architecture
- [ ] Implement PyTorch data loader
- [ ] Implement training function
- [ ] Implement evaluation function
- [ ] Implement class-weighted Cross-Entropy
- [ ] Perform hyperparameter tuning
- [ ] Train using three random seeds
- [ ] Save baseline results
- [ ] Analyse baseline performance

### NB2 - Experiments

- [ ] Implement Focal Loss
- [ ] Verify gamma = 0 equivalence
- [ ] Perform gamma experiments
- [ ] Perform hyperparameter tuning
- [ ] Run 2% label noise experiments
- [ ] Run 5% label noise experiments
- [ ] Generate comparison graphs
- [ ] Analyse experimental results
- [ ] Discuss limitations
- [ ] Write conclusions

### NB3 - Educational Notebook

- [ ] Explain Cross-Entropy
- [ ] Explain Focal Loss
- [ ] Explain gamma and class weights
- [ ] Present experimental findings
- [ ] Include educational exercises
- [ ] Complete educational notebook

### Final Submission

- [ ] Complete meeting log
- [ ] Complete AI-use log
- [ ] Implement QUICK mode
- [ ] Test all notebooks
- [ ] Check submission size
- [ ] Record group video
- [ ] Submit coursework

---

## 14. Coursework Requirements

All three notebooks must run from top to bottom using the
university's dl2026 environment.

Each notebook must include:

```python
QUICK = True
```

When QUICK is True, the notebook should load saved results
or checkpoints and run in under five minutes.

When QUICK is False, the full experiment should run.

The original training code must be included and must correspond
to the saved experimental results.

### Required Submissions

- Final challenge statement.
- NB1 - Baseline notebook.
- NB2 - Experimental notebook.
- NB3 - Educational notebook.
- Required utility code and checkpoints.
- Project log containing meeting and AI-use logs.
- Group video lasting 5-8 minutes.

The project log should be approximately 2-3 pages and must not
exceed four pages.

Every group member must speak in the video.

The Gradescope submission limit is 100 MB.

The final video is submitted separately through Minerva.

AI tools should be used for learning and advice, in accordance
with the coursework policy. Every member must be able to explain
and reproduce their own work.
