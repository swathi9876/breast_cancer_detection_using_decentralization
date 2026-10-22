from common_modules import *
import torch.nn as nn
def create_federated_dataset(client_data):
    federated_data = []
    for X, y in client_data:
        federated_data.append({'X': torch.tensor(X, dtype=torch.float32),
                               'y': torch.tensor(y, dtype=torch.long)})
    return federated_data

# Defining the FedAvg algorithm
def federated_average(model, client_models):
    averaged_state_dict = {}
    for i, client_state_dict in enumerate(client_models):
        for key, value in client_state_dict.items():
            if i == 0:
                averaged_state_dict[key] = value.clone().detach()
            else:
                averaged_state_dict[key] += value.clone().detach()
    for key in averaged_state_dict:
        averaged_state_dict[key] /= len(client_models)
    return averaged_state_dict
    
# Defining FedProx Algorithm
def fedprox(model, client_models):
    averaged_state_dict = {}
    for i, client_state_dict in enumerate(client_models):
        for key, value in client_state_dict.items():
            if i == 0:
                averaged_state_dict[key] = value.clone().detach()
            else:
                averaged_state_dict[key] += value.clone().detach()
    for key in averaged_state_dict:
        averaged_state_dict[key] /= len(client_models)
    return averaged_state_dict
