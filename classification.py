import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
import os
import io
import time

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS

from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

color = list(colors.TABLEAU_COLORS.keys())

def evaluateQNN(X:torch.tensor, y:torch.tensor, epochs: int):
    dim = 2
    feature_map = ZZFeatureMap(dim)
    ansatz = RealAmplitudes(dim)
    qc = QuantumCircuit(dim)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    qnn1 = EstimatorQNN(
    circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters
)
    print(qnn1.num_weights)
    initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn1.num_weights) - 1)
    model1 = TorchConnector(qnn1, initial_weights=initial_weights)
    optimizer = LBFGS(model1.parameters())
    f_loss = MSELoss(reduction="sum")
    model1.train()  
    def closure():
        optimizer.zero_grad() 
        loss = f_loss(model1(X), y)  
        loss.backward()
        return loss
    start = time.time()
    for i in range(epochs):
        optimizer.step(closure)
    end = time.time()
    y_predict = []
    for x, y_target in zip(X, y):
        output = model1(Tensor(x))
        y_predict += [np.sign(output.detach().numpy())[0]]
    y_predict = torch.tensor(y_predict)
    accuracy = sum(y_predict == y) / len(y)
    for x, y_target, y_p in zip(X, y, y_predict):
        plt.plot(x[0], x[1], color=color[targets.index(y_target)], marker='o')
        if y_target != y_p:
            plt.scatter(x[0], x[1], s=200, facecolors="none", edgecolors="r", linewidths=2)
    plt.plot([-1, 1], [1, -1], "--", color="black")
    plt.xlabel(xDF.columns[0])
    plt.ylabel(xDF.columns[1])
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    return max(accuracy), img_buf, round(end-start, 1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 6)
        self.relu = nn.ReLU()
        self.output = nn.Linear(6, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) 
    return acc

def evaluateNN(X:torch.tensor, y:torch.tensor, epochs: int):
    net = NeuralNetwork()
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
    if len(targets) == 2:
        loss_fn = nn.BCEWithLogitsLoss()
    else: 
        loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), 
                                lr=0.1)
    start = time.time()
    net.train()
    X = X.unsqueeze(0)
    y = y.squeeze(1)
    X = X.float()
    for epoch in range(epochs):
        y_logits = net(X).squeeze() 
        if len(targets) == 2:
            loss = loss_fn(torch.sigmoid(y_logits), y.float())
        else: 
            loss = loss_fn(y_logits, y.float()) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = time.time()
    print(end-start)
    net.eval()
    with torch.inference_mode():
        test_logits = net(X).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        #train_loss = loss_fn(test_logits, y)
        train_acc = accuracy_fn(y_true=y, y_pred=test_pred)
    return train_acc, round(end-start, 1) 

def classify(df: pd.DataFrame, epochs: int):
    global targetDF, targets, xDF

    targetDF = df.iloc[:, -1]
    xDF = df.iloc[:, :2]
    X = torch.tensor(xDF.values, dtype=torch.double)
    y = torch.tensor(targetDF.values, dtype=torch.double).unsqueeze(1)
    Xlong = torch.tensor(xDF.values, dtype=torch.long)
    ylong = torch.tensor(targetDF.values, dtype=torch.long).unsqueeze(1)
    targets = list(targetDF.unique())
    qnnAcc, img_buf, qnnTime = evaluateQNN(X, y, epochs)
    nnAcc, nnTime = evaluateNN(Xlong, ylong, epochs)
    return {
        'qnnAcc': qnnAcc,
        'img_buf': img_buf,
        'qnnTime': qnnTime,
        'nnAcc': nnAcc,
        'nnTime': nnTime
    }
