import json
import logging
from collections import Counter
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import torch
from base import Graph
from torch.autograd import Variable
from sklearn.model_selection import KFold

# set log format
handlers_list = [logging.StreamHandler()]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)


class PlusHIVData(Dataset):
    def __init__(self, filename, batch_size=64):
        self.batch_size = batch_size
        raw_data = pd.read_csv(filename)
        self.smiles = raw_data['smiles'].values
        self.features = raw_data.iloc[:, 2:].values
        self.activity = raw_data['activity'].values
        self.graphs = np.array([Graph(smiles) for smiles in self.smiles])
        self.size = self.graphs.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        batch_graphs = self.graphs[idx]
        batch_features = torch.from_numpy(self.features[idx]).type(torch.float32)
        batch_activity = torch.tensor(self.activity[idx], dtype=torch.long)
        return (batch_graphs, batch_features), batch_activity

    @classmethod
    def train_iterator(cls, data, feature, target, batch_size):
        pointer = 0
        while True:
            if pointer >= data.shape[0]:
                break
            batch_data = data[pointer: pointer + batch_size]
            batch_features = torch.from_numpy(feature[pointer:pointer + batch_size]).type(torch.float32)
            batch_targets = torch.from_numpy(target[pointer: pointer + batch_size]).type(torch.long)
            yield (batch_data, batch_features), batch_targets
            pointer += batch_size

    def get_cv_sets(self, folds=5):
        folder = KFold(n_splits=folds, shuffle=True)
        for train_index, test_index in folder.split(self.graphs):
            graph_train, graph_val = self.graphs[train_index], self.graphs[test_index]
            ac_train, ac_val = self.activity[train_index], self.activity[test_index]
            feature_train, feature_val = self.features[train_index], self.features[test_index]
            yield PlusHIVData.train_iterator(graph_train, feature_train, ac_train, self.batch_size), (graph_val, torch.from_numpy(feature_val).type(torch.float32)), torch.from_numpy(ac_val).type(torch.long)

    @classmethod
    def make_variables(cls, graphs):
        """
        wrap every serial value (atom and bonds) in Variable, for back_propagation in training process.
        :param graphs: Iterable of Graph
        """
        for graph in graphs:
            for node in graph.nodes:
                node.msg = Variable(node.msg)
                node.bonds = [Variable(bond) for bond in node.bonds]

    def __iter__(self):
        self._pointer = 0
        return self

    def __next__(self):
        if self._pointer >= self.size:
            raise StopIteration
        sample_index = np.random.randint(0, self.size, size=(self.batch_size,))
        batch_graphs = self.graphs[sample_index]
        batch_features = torch.from_numpy(self.features[sample_index]).type(torch.float32)
        batch_activity = torch.from_numpy(self.activity[sample_index]).type(torch.long)
        self._pointer += self.batch_size
        return (batch_graphs, batch_features), batch_activity




"""
Create your data loader for training/testing local & global model.
Keep the value of the return variable for normal operation.
"""

def load_partition(dataset, validation_split, batch_size):
    """
    The variables train_loader, val_loader, and test_loader must be returned fixedly.
    """

    data_loader = PlusHIVData('/home/ccl/Desktop/isfolder/ADMET/ADMET_RNN/datasets/plus_hiv_train.txt', batch_size=batch_size)

    # Split the dataset into training and validation sets
    train_size = int((1 - validation_split) * len(data_loader))
    val_size = len(data_loader) - train_size
    train_dataset, val_dataset = random_split(data_loader, [train_size, val_size])

    # DataLoader for client training, validation, and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Note: test_dataset needs to be defined or passed as an argument if required
    # Here we assume that the test dataset is also split from the initial dataset
    test_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def gl_model_torch_validation(batch_size):
    """
    Setting up a dataset to evaluate a global model on the server
    """

    data_loader = PlusHIVData('./datasets/plus_hiv_train.txt', batch_size=batch_size)
    
    # DataLoader for global model validation
    gl_val_loader = DataLoader(data_loader, batch_size=batch_size)

    return gl_val_loader
