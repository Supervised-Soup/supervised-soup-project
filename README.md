# Supervised Soup Project

## Overview & Motivation

The focus of this repository is a systematic experimental study of how **data quality** and **training strategies** influence the performance and generalization of deep learning models for image classification.

This work was conducted as part of the **xAI-Proj-B** course at Otto-Friedrich University of Bamberg and is documented in the accompanying project report *"The Influence of Data Quality and Training Strategies on the Performance of Image Classification Models"*.

In this project, we investigate how different factors (dataset cleaning, optimizer choice, model depth, freezing strategies, and data augmentation) affect performance, convergence, and robustness of **ResNet-based image classifiers** pretrained on ImageNet. 

---

## How It Works

The project follows a controlled experimental pipeline for supervised image classification:

1. **Dataset preparation and analysis**  
   An ImageNetSubset with 10 classes is analyzed and manually cleaned to remove mislabeled, ambiguous, or irrelevant samples. Experiments are conducted on both the original and cleaned versions of the dataset.

2. **Model training with transfer learning**  
   ResNet architectures (ResNet-18, 34, 50, and 101) pretrained on ImageNet are fine-tuned on the target dataset. Different freezing strategies are applied, ranging from a fully frozen backbone to selective unfreezing of higher layers.

3. **Training strategy variations**  
   We systematically vary:
   - Optimization algorithms (SGD, AdamW, RMSProp, AdaGrad)
   - Freezing strategies (frozen, partially unfrozen, fully unfrozen)
   - Model depth
   - Data augmentation strength and policies

4. **Evaluation and comparison**  
   Models are evaluated using accuracy, macro F1-score, macro ROC-AUC, and cross-entropy loss. Best checkpoints are selected based on validation loss and evaluated on a test set that consists of photos made in an outdoor context.

This design allows us to isolate the effect of each factor while keeping all other variables fixed, enabling meaningful comparisons across experiments.

---

## Repository Structure

The repository is organized to clearly separate data, experiments, reusable code, and documentation:

```
.
├── data/                      # Datasets (gitignored; stored externally)
├── docs/                      # Documentation, guides, and notes
├── experiments/               # Experiment configurations and experiment-specific artifacts
├── notebooks/                 # Jupyter & Colab notebooks for training and evaluation
├── scripts/                   # Helper scripts (setup, utilities, automation)
├── supervised_soup/           # Core reusable code (models, training, evaluation logic)
├── tests/                     # Basic tests and sanity checks
├── .env.example               # Template for environment variables
├── .gitignore                 # Git ignore rules
├── LICENSE                    # License information
├── README.md                  # Project overview and instructions
├── environment.yml            # Conda environment definition (local, reproducible)
├── pyproject.toml             # Project metadata and packaging configuration
└── requirements.txt           # Python dependencies (general use)
```

**Design rationale:**
- Experimental logic is kept out of notebooks and implemented in `supervised_soup/` for reuse
- Notebooks focus on orchestration, visualization, and analysis
- Environment definitions ensure reproducibility across local machines and Colab

---

## Installation & Setup

### Option A: Local Setup (Recommended for Development)

**Prerequisites**
- Conda installed
- pip installed
- VS Code (or another Python IDE)

**Steps**

1. Clone the repository
```bash
git clone https://github.com/NeuralSpiral/supervised-soup-project.git
cd supervised-soup-project
```

2. Create a `.env` file in the project root
```bash
touch .env
```

Edit the file and add the following lines:
```
DATA_PATH=/absolute/path/to/ImageNetSubset
RESULTS_PATH=results
```

Notes:

- Look at the .env.example file for reference

- DATA_PATH should point to your local copy of the dataset. I recommend to keep it inside the repo in data, but it can be anywhere you want.

- Because the data folder is in the gitignore in order to keep the repo size small, you need to download the dataset and put it in the specified path, if you want to use it locally.


3. Run the local setup script
```bash
bash scripts/local_setup.sh
```
This will:
- Create the `supervised-soup-env` Conda environment
- Install all dependencies
- Install CPU‑only PyTorch for lighter local usage

4. Activate the environment
```bash
conda activate supervised-soup-env
```

5. Install the project as a package
```bash
pip install -e .
```
Check if the package is installed by running:
```bash
pip show supervised_soup
```

6. Test the setup
```bash
python tests/setup_test.py
```
If everything is correct, you should see:
```
Looks good.
```

7. Open environment in the project root in VS Code:
```bash
code .
```
Select conda environment supervised-soup-env as your Python interpreter.


---

### Option B: Google Colab Setup

For training on GPU, we recommend Colab.

1. Open a notebook from `notebooks/`
2. Run the setup cell from `colab_setup.ipynb`
3. Ensure the dataset shortcut exists in your Drive:
```
/content/drive/MyDrive/SupervisedSoupData/ImageNetSubset
```

The Colab setup installs the same dependencies, but uses GPU‑enabled PyTorch automatically.

---

## Quick Tutorial

This short tutorial walks you through reproducing the core experiments.

### 1. Dataset Setup

- Download or access the ImageNetSubset dataset
- Set the dataset path via the `.env` file or Colab setup


### 2. Training Experiments

- Open `colab_training_notebook_updated.ipynb` or run locally
- Select:
  - Model depth (e.g. ResNet-18, ResNet-50, ResNet-101)
  - Freezing strategy
  - Optimizer
  - Augmentation preset
- Train for the specified number of epochs

Checkpoints and logs are stored automatically with wandb or locally in `results/`.

### 3. Evaluation

- Test evaluation can be done locally with the test_evaluation.py module 

### 4. Reproducing Paper Results

All reported results in the paper can be reproduced using the provided notebooks and default configurations. Hyperparameters and augmentation presets are documented directly in the code and appendices of the paper.

---

## Reproducibility Notes

- All dependencies are pinned via `environment.yml`
- Data paths are handled through `.env`
- Experiments are isolated from core logic
- The project can be installed as a package for clean imports

---

## Tooling

This project integrates the following tools:
- **GitHub** – version control
- **Jira** – task tracking
- **Confluence** – documentation
- **Google Colab** – GPU training
- **VS Code** – local development

---

## Contact Information

**Project Team – xAI-Proj-B**

**Jonas Reutter**  
Email: jonas-felix.reutter@stud.uni-bamberg.de  

**Tatsiana Shelepen**  
Email: tatsiana.shelepen@stud.uni-bamberg.de  

**Paul Franz Amschler**   
Email: paul-franz.amschler@stud.uni-bamberg.de  

---

## License

This project is intended for academic and educational use as part of the xAI-Proj-B course.



