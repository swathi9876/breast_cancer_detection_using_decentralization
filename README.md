# breast_cancer_detection_using_decentralisation

**Version 2.0**

## Summary

Countless individuals around the world are impacted by breast cancer, making it a substantial global health challenge. Detecting this disease at an early stage is essential for effective treatment, and recent technological progress has provided fresh opportunities to enhance the precision of breast cancer diagnosis. Decentralized machine learning is employed for breast cancer detection while taking into account the privacy considerations of healthcare institutions. Decentralized machine learning (DML) surpasses centralized machine learning (ML) in preserving privacy for breast cancer detection. In centralized ML, raw patient data is gathered and stored in a central location, posing privacy risks. DML, however, ensures privacy by sharing model insights and updates derived from local datasets, avoiding the transfer of identifiable patient information.

# Breast Cancer Detection using Decentralized (Federated) Machine Learning

Each simulated hospital (client) trains a model on its own data, and only model updates are sent to a central server for aggregation, so raw patient data is never pooled.

The project compares the **FedAvg** and **FedProx** algorithms with **SVM**, **logistic regression** and **feed-forward neural network** models, under both **IID** and **non-IID** data partitions.

> Final-year B.E. Computer Engineering project, Pune Institute of Computer Technology (PICT), Savitribai Phule Pune University, 2023–24.

---

## What's new in v2.0

- Non-IID client partitions are generated using a **Dirichlet distribution**, giving controlled label skew across clients.
- **TensorFlow removed and replaced with PyTorch**, as TensorFlow made training slow.
-Combined graphs for fed prox and fed avg. 
---

## Why federated learning?

In centralized ML, patient records are collected in one place, which creates privacy and compliance risks. In federated learning:

1. The server sends a global model to every client.
2. Each client trains it on local data.
3. Clients send back only updated parameters.
4. The server aggregates them into a new global model, and the cycle repeats.

**FedAvg** simply averages client updates. **FedProx** adds a proximal term to each client's local loss, `(μ/2)·||w − wᵗ||²`, which keeps local models close to the global one and is meant to help when client data is non-IID.



## Repository structure

```
fed_project/
│
├── IID/
│   └── svm/                    # SVM, FedAvg & FedProx, IID split
│
├── NON_IID/
│   ├── svm/                    # SVM, FedAvg & FedProx, Dirichlet split
│   ├── logistic/               # Logistic regression, FedAvg & FedProx
│   └── feedforward/            # Feed-forward neural network, FedAvg & FedProx
│
├── ank.csv                     # Breast cancer dataset
├── common_modules.py           # Shared helper modules used across experiments
├── federated_methods.py        # FedAvg and FedProx implementations
├── preprocessing.py            # Data loading, cleaning and client partitioning

```

| Folder | Data split | Models |
|--------|-----------|--------|
| 🟦 `IID/` | Randomly and evenly distributed samples, so every client sees a similar data distribution | SVM |
| 🟦 `NON_IID/` | Skewed partitions generated with a Dirichlet distribution, simulating hospitals with different patient populations | SVM, Logistic Regression, Feed-Forward NN |

| File | Purpose |
|------|---------|
| `ank.csv` | Dataset used for all experiments |
| `common_modules.py` | Shared helper modules |
| `federated_methods.py` | FedAvg and FedProx implementations |
| `preprocessing.py` | Data preprocessing and client partitioning |


## How to run

1. Make sure `ank.csv` is available in the `fed_project` folder (or upload it to the Colab session via the Files panel).
2. Open the notebook or script for the experiment you want inside `IID/` or `NON_IID/`.
3. Run all cells (`Runtime → Run all`).
4. Change the number of clients (2 to 10), number of rounds, learning rate and Dirichlet α in the config cell to reproduce the experiments.

## Tech stack

Python 3.10 · PyTorch · NumPy · pandas · scikit-learn · Matplotlib · Google Colab

> TensorFlow was removed in v2.0 and replaced with PyTorch, as TensorFlow made training slower.


## Team

Ankita More · Rajat Singh · Mehek Sirwani · Swathi Babaram

**Guide:** Prof. Ashwini Bundele, Department of Computer Engineering, PICT, Pune

## References

1. T. Li et al., *Federated Learning: Challenges, Methods, and Future Directions*, IEEE Signal Processing Magazine, 2020.
2. Q. Yang et al., *Federated Machine Learning: Concept and Applications*, ACM TIST, 2019.
