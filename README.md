# VEGFR2 HeteroEncoder: De Novo Molecule Design


A Heterogeneous Conditional VAE (Hetero-CVAE) pipeline with Reinforcement Learning for generating novel VEGFR2 inhibitors.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green)
![License](https://img.shields.io/badge/License-MIT-grey)

## 📌 Overview

This project implements a deep learning framework designed to generate high-affinity drug candidates targeting VEGFR2. The core architecture is a Heterogeneous CVAE that encodes both molecular syntax (SMILES) and physicochemical descriptors (MW, LogP, TPSA, etc.) into a shared latent space.

To overcome the limitations of standard generative models, this pipeline includes a Reinforcement Learning (RL) stage using the REINFORCE algorithm to fine-tune the generator for:
1.  Validity: Penalizing generation of non valide SMILES strings.
2.  Novelty: Penalizing molecules already present in the training set.
3.  Diversity: Penalizing repetitive generation within the same batch.
4.  Scaffold Retention: Enforcing the presence of the required pharmacophore (`O=C(N)c1ccnc2ccccc12`).

---

## 📊 Key Results & Performance

My two-stage training approach (Supervised + RL) yields significant improvements in the generation of unique, valid, and scaffold-compliant molecules.

### 1. Training Convergence
The supervised training phase demonstrates stable minimization of the Evidence Lower Bound (ELBO).
## 📉 Objective Functions & Training Strategy

To ensure the generation of valid, high-affinity, and novel molecules, the model creates a balance between learning chemical syntax (Supervised) and exploring new chemical space (RL).

### 1. Supervised Pre-training (CVAE Loss)
During the first stage, the model minimizes the **Evidence Lower Bound (ELBO)** loss, which consists of two weighted components:

$$
\mathcal{L}_{total} = \mathcal{L}_{recon} + \beta \cdot \mathcal{L}_{KL}
$$

**Reconstruction Loss ($\mathcal{L}_{recon}$):** Standard **Cross-Entropy Loss** between the predicted token probabilities and the actual SMILES tokens. This forces the model to learn correct chemical syntax and grammar.

$$
\mathcal{L}_{recon} = \left( \sum_{t=1}^{T} \log P(x_t \mid x_{<t}, z, c) \right)
$$
$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$

   *(Where $z$ is the latent vector and $c$ is the energy condition)*

**KL Divergence ($\mathcal{L}_{KL}$):** Regularizes the latent space to approximate a standard Normal distribution $\mathcal{N}(0, I)$. This ensures the latent space is continuous and can be sampled.

   $$
   \mathcal{L}_{KL} = D_{KL}(q(z|x) \parallel p(z)) = -\frac{1}{2} \sum (1 + \log(\sigma^2) - \mu^2 - \sigma^2)
   $$

**Weighting ($\beta$):** We use a fixed weight ($\beta = 0.005$) to prevent posterior collapse, ensuring the decoder relies on the latent code.

---

### 2. Reinforcement Fine-tuning (Reward Policy)
After pre-training, we use the **REINFORCE** algorithm (Policy Gradient) to fine-tune the decoder. The goal is to maximize the expected reward $J(\theta)$.

We implemented a **Tiered Reward Function** with a specific focus on **Diversity** and **Scaffold Retention**. The agent is penalized for generating molecules that are invalid, lack the pharmacophore, or are mere duplicates of the training data.

| Outcome | Reward | Condition |
| :--- | :--- | :--- |
| **Invalid** | **-5.0** | RDKit fails to parse the SMILES string. |
| **Valid (No Scaffold)** | **+0.5** | Chemically valid, but lacks the VEGFR2 core structure (`O=C(N)c1ccnc2ccccc12`). |
| **Valid + Scaffold (Duplicate)** | **+2.0** | Contains the scaffold but is either: <br>1. Present in the **Training Set** (Known).<br>2. Already generated in the **current epoch** (Mode collapse). |
| **Valid + Scaffold (Novel)** | **+10.0** | **TARGET:** Contains scaffold, is **NOT** in the database, and is **UNIQUE** in the current batch. |

This "Diversity Penalty" forces the model to explore the chemical space rather than memorizing high-affinity seeds.
<img width="3600" height="1800" alt="training_loss" src="https://github.com/user-attachments/assets/cabdbeb1-d657-4d41-8159-b8c855ad4f26" />

*Figure 1: Training and Validation loss over 20 epochs. The model successfully learns the chemical syntax and property embeddings.*

### 2. RL Optimization (Novelty & Reward)
During the RL fine-tuning phase, the model adapts to maximize the reward function. The "Diversity Penalty" forces the model to explore new chemical spaces rather than memorizing high-affinity seeds.
<img width="4800" height="1800" alt="rl_results" src="https://github.com/user-attachments/assets/c3f38e78-7a41-4d08-bd70-7f4fe902227d" />

*Figure 2: Evolution of Average Reward during RL fine-tuning. Higher reward indicates a higher rate of novel, scaffold-containing molecules.*

### 3. Generation Statistics
In a sample generation run of 1,000 attempts targeting a binding energy of -10.0 kcal/mol, the pipeline achieved the following metrics:

| Metric             | Results **without** fine-tuning | Results **with** fine-tuning | Description                                                                                       |
|:-------------------|:--------------------------------|:-----------------------------|:--------------------------------------------------------------------------------------------------|
| Validity           | 83%                             | 98%                          | Percentage of chemically valid SMILES generated.                                                  |
| Novelty            | 68%                             | 96%                          | Percentage of molecules that are different from the training/validating dataset.                  |
| Uniqueness         | 65%                             | 94%                          | Percentage of unique molecules obtained (compared to other generated molecules)                   |
| With pharmacophore | 80%                             | 94%                          | Percentage of molecules that have the required pharmacophore verified using RDKit SubstructMatch) |

---

## 🧠 Model Architecture

The HeteroEncoderCVAE fuses multiple data modalities:

1.  Encoder: 
    *   Text Branch: GRU processing SMILES tokens.
    *   Descriptor Branch: Dense layers processing normalized physical properties (Molecular Weight, LogP, TPSA).
    *   Fusion: Concatenation of the GRU hidden state and descriptor features.
2.  Latent Space:
    *   Parameters $\mu$ and $\sigma$ for the Gaussian distribution ($z \in \mathbb{R}^{64}$).
4.  Decoder: 
    *   Conditioned on Latent Vector $z$ + Target Binding Energy.
    *   Autoregressive GRU reconstructs the SMILES string token by token.

---

## 🛠️ Installation

1.  Clone the repository: 
```bash
git clone https://github.com/your-username/vegfr2-heterogen.git
```

```bash
cd vegfr2-heterogen
````
    
2.  Install dependencies:
       
```bash
pip install torch pandas numpy rdkit matplotlib tqdm
```

---    

## 💻 Usage

The pipeline is unified under main.py. You can run specific stages using the --mode argument.

### 1. Data Preprocessing
Filters the raw dataset, removes duplicates, calculates descriptors, and creates the vocabulary.
```bash
python main.py --mode preprocess
```

### 2. Supervised Training
Trains the CVAE on the processed dataset to learn chemical syntax.
```bash
python main.py --mode train
```

### 3. Reinforcement Learning
Fine-tunes the decoder to maximize validity,novelty and diversity scores.
```bash
python main.py --mode rl
```

### 4. Generation
Generates new molecules based on target energy.
* `--samples`: Number of molecules to attempt.
* `--energy`: Target binding energy (e.g., -10.0).
* `--noise`: Variance factor for latent space sampling.
```bash
python main.py --mode generate --samples 1000 --energy -12.0 --noise 0.2
```
**Output:** A file named `novel_molecules.csv` containing only unique, novel molecules suitable for docking.

---

## 📂 Project Structure

```text
.
├── dataset.csv               # Input raw data containing SMILES and Energy
├── main.py                   # Main pipeline entry point (CLI)
├── model.py                  # PyTorch model definition (HeteroEncoderCVAE)
├── data_preprocessing.py     # Logic for cleaning, filtering, and tokenizing
├── train.py                  # Supervised learning (Stage 1)
├── rl_train.py               # Reinforcement learning (Stage 2)
├── generate.py               # Generation of novel molecules
└── utilities.py              # Helpers: SmilesTokenizer, plotting, reward functions
```
