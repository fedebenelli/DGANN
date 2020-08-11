from NeuralNetwork import *
from pprint import pprint
import numpy as np
import pickle

with open('nn.obj','rb') as f:
    nn = pickle.load(f)
with open('maxes.obj','rb') as f:
    maxes = pickle.load(f)

faults = {
    0: "PD",
    1: "D1",
    2: "D2",
    3: "T1y2",
    4: "T3"
}

while True:
    data_input=[]
    for gas in maxes:
        data_input.append(float(input(f'Cantidad de gas {gas}: '))/maxes[gas])

    gases = np.array(data_input)
    nn.input = gases
    nn.feedforward()

    certeza = dict()
    for i in range(len(nn.output)):
        certeza[i] = round(nn.output[i]/sum(nn.output)*100,2)
        certeza[faults[i]] = certeza.pop(i)
    pprint(certeza,width=1)