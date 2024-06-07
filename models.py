from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
from base import Graph
from torch.utils.data import DataLoader, Dataset, random_split
from torch.autograd import Variable
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


class FeatureRNN(nn.Module):
    """
    Extract features based on the recursive structure of molecules
    Input: size = (K,), Iterable objects of base.Graph
    Output: size = (K, output_size), torch.Tensor

    :argument
    atom_msg: int, length of Node.msg
    bond_msg: int, length of tensors in Node.bonds
    hidden_size: int, number of neures in recursive net
    output_size: int, number of output features
    """

    def __init__(self, atom_msg=125, bond_msg=12, hidden_size=125, output_size=40):
        super(FeatureRNN, self).__init__()
        self.bond_msg = bond_msg
        self.output_size = output_size
        self.net = nn.Sequential(nn.Linear(atom_msg + bond_msg + output_size, hidden_size),
                                 nn.LeakyReLU(0.01),
                                 nn.Linear(hidden_size, output_size),
                                 nn.LeakyReLU(0.01))

    def forward(self, graphs):
        outputs = []
        for graph in graphs:
            outputs.append(self.single_forward(graph))
        return torch.cat(outputs, dim=0)

    def single_forward(self, graph):
        """
        predict only one graph at a time
        :param graph: Graph
        :return: torch.Tensor
        """
        output = torch.cat([self._recurse_node(tree.root) for tree in graph.get_trees()], dim=0).sum(dim=0).view(1, -1)
        return output

    def _recurse_node(self, node):
        """
        core recursive function
        """
        if not node.has_child():
            return self.net(
                torch.cat([node.msg, torch.zeros(size=(self.bond_msg + self.output_size,))], dim=0).view(1, -1))
        upper_msgs = torch.cat([self._recurse_node(child) for child in node.children], dim=0)
        bonds = torch.cat([bond.view(1, -1) for bond in node.child_bonds], dim=0)
        msg = torch.mm(torch.ones(size=(bonds.shape[0], 1)), node.msg.view(1, -1))
        output = self.net(torch.cat([msg, bonds, upper_msgs], dim=1)).sum(dim=0).view(1, -1)
        return output

# Define your custom Model    
class PlusHIVNet(nn.Module):
    def __init__(self, atom_msg=125, bond_msg=12, inner_hidden_size=125, feature_size=40, hidden_size=100):
        super(PlusHIVNet, self).__init__()
        self.feature_net = FeatureRNN(atom_msg, bond_msg, inner_hidden_size, feature_size)
        self.net = nn.Sequential(nn.Linear(feature_size + 196, hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size, 2),
                                 nn.LogSoftmax(1))
        self.descriptors = [desc[0] for desc in Descriptors._descList if not (
                desc[0] in ['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge'])]
        self.desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptors)

    def forward(self, data_bunch):
        graphs, plus_features = data_bunch
        features = self.feature_net(graphs)
        features = torch.cat([features, plus_features], dim=1)
        return self.net(features)

    def predict(self, smiles_es):
        graphs = [Graph(smiles) for smiles in smiles_es]
        plus_features = [list(self.desc_calc.CalcDescriptors(Chem.MolFromSmiles(smiles))) for smiles in smiles_es]
        plus_features = torch.Tensor(plus_features).type(torch.float32)
        return torch.exp(self((graphs, plus_features)).detach())

    def predict_class(self, smile_es):
        return self.predict(smile_es).argmax(dim=1)

# Set the torch train & test
# torch train
def train_torch():
    def custom_train_torch(model, train_loader, epochs, cfg):
        """
        Train the network on the training set.
        Model must be the return value.
        """
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
        loss_fn = nn.NLLLoss(weight=torch.Tensor([0.02, 0.98]))
        
        for epoch in range(epochs):
            running_loss = 0.0
            for data in tqdm(train_loader):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
            
        return model
    
    return custom_train_torch

# torch test
def test_torch():
    
    def custom_test_torch(model, test_loader, cfg):
        """
        Validate the network on the entire test set.
        Loss, accuracy values, and dictionary-type metrics variables are fixed as return values.
        """
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        metrics = {}
        with torch.no_grad():
            for data in tqdm(test_loader):
                inputs, labels = data
                outputs = model(inputs)
                loss = F.nll_loss(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        average_loss = running_loss / len(test_loader)
        accuracy = 100 * correct / total
        metrics["f1_score"] = f1_score(labels.cpu(), predicted.cpu(), average='weighted')
        
        model.to("cpu")  # move model back to CPU
        return average_loss, accuracy, metrics
    
    return custom_test_torch