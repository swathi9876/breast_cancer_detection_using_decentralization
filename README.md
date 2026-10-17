# Fedprox_ml_breast_cancer_detection
Decentralized machine learning is employed for breast cancer detection while taking into account the privacy considerations of healthcare institutionsIn centralized ML, raw patient data is gathered and stored in a central location, posing privacy risks. DML, however, ensures privacy by sharing model insights and updates derived from local datasets
# Breast Cancer Detection using Decentralized (Federated) Machine Learning

Privacy-preserving breast cancer classification using **federated learning**. Each simulated hospital (client) trains a model on its own data, and only model updates are sent to a central server for aggregation, so raw patient data is never pooled.

The project compares the **FedAvg** and **FedProx** algorithms with **SVM** and **logistic regression** models, under both **IID** and **non-IID** data partitions.

> Final-year B.E. Computer Engineering project, Pune Institute of Computer Technology (PICT), Savitribai Phule Pune University, 2023–24.

---

## Why federated learning?

In centralized ML, patient records are collected in one place, which creates privacy and compliance risks (HIPAA, GDPR). In federated learning:

1. The server sends a global model to every client.
2. Each client trains it on local data.
3. Clients send back only updated parameters.
4. The server aggregates them into a new global model, and the cycle repeats.

**FedAvg** simply averages client updates. **FedProx** adds a proximal term to each client's local loss, `(μ/2)·||w − wᵗ||²`, which keeps local models close to the global one and is meant to help when client data is non-IID.

## Repository structure

```
├── IID/
│   └── svm_iid_federated.ipynb              # SVM, FedAvg & FedProx, IID split
├── non-IID/
│   └── svm_logistic_non_iid_federated.ipynb # SVM & logistic regression, non-IID split
└── README.md
```

| Folder | What it contains |
|---|---|
| `IID/` | Clients receive randomly and evenly distributed samples, so every client sees a similar data distribution. |
| `non-IID/` | Clients receive skewed partitions (for example, different class proportions), simulating real hospitals with different patient populations. |

## Run in Google Colab

| Notebook | Open |
|---|---|
| IID, SVM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swathi9876/breast_cancer_detection_using_decentralization/blob/main/IID/svm_iid_federated.ipynb) |
| Non-IID, SVM and logistic regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swathi9876/breast_cancer_detection_using_decentralization/blob/main/non-IID/svm_logistic_non_iid_federated.ipynb) |

1. Download the dataset (see below) and upload `breastcancer.csv` to the Colab session (Files panel → Upload).
2. Run all cells (`Runtime → Run all`).
3. Change the number of clients (2 to 10), number of rounds and learning rate in the config cell to reproduce the experiments.

## Dataset

Breast cancer dataset from Kaggle, `breastcancer.csv`: 569 samples, binary label (benign or malignant). It is a small public dataset used only for proof-of-concept experiments, not real clinical data.

## Tech stack

Python 3.10 · PyTorch · NumPy · pandas · scikit-learn · Matplotlib · Google Colab

## Results

Accuracy measured across runs with 2 to 10 clients.

| Model | Algorithm | Data split | Max acc. | Avg acc. |
|---|---|---|---|---|
| SVM | FedAvg | IID | 0.63 | 0.63 |
| SVM | FedProx | IID | 0.63 | 0.63 |
| SVM | FedAvg | Non-IID | 0.79 | 0.69 |
| SVM | FedProx | Non-IID | 0.64 | 0.40 |
| Logistic regression | FedAvg | – | 1.00 | 0.79 |
| Logistic regression | FedProx | – | 0.90 | 0.64 |

### Observations
- Results varied strongly with the number of clients, especially on non-IID data.
- FedProx did not consistently beat FedAvg here, and it was less stable on non-IID splits.


## Team

Ankita More · Rajat Singh · Mehek Sirwani · Swathi Babaram

**Guide:** Prof. Ashwini Bundele, Department of Computer Engineering, PICT, Pune

## References

1. T. Li et al., *Federated Learning: Challenges, Methods, and Future Directions*, IEEE Signal Processing Magazine, 2020.
2. Q. Yang et al., *Federated Machine Learning: Concept and Applications*, ACM TIST, 2019.


