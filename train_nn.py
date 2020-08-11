import matplotlib.pylab as plt
from NeuralNetwork import *
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

# Clasification variables
faults = {0:"PD",1: "D1",2: "D2",3:"T12",4:"T3"}
gases = ["H2","CH4","C2H4","C2H6","C2H2","CO","CO2"]
gasesf = ["H2","CH4","C2H4","C2H6","C2H2","CO","CO2",'fault']

df = pd.read_excel('data.xlsx')[gasesf]

# Normalize data
# Give the gases a log10 scale
maxes = dict()
for gas in gases:
    maxes[gas] = df[gas].max()
    df[gas] = df[gas]/df[gas].max()

# Define X and Y
X = np.array(df[gases])
Y = list(df['fault'])

# Make a new Y with a vector of outcomes
dic = {
    "0":[1,0,0,0,0],
    "1":[0,1,0,0,0],
    "2":[0,0,1,0,0],
    "3":[0,0,0,1,0],
    "4":[0,0,0,0,1]
}
new_Y = []

for i in Y:
    new_Y.append(dic[str(i)])

Y = np.array(new_Y)

# Train
iterations = 487300
nodes = 50
nn    = NeuralNetwork(X, Y, nodes, iterations, maxes)
nn.train()
nn.results(df, gasesf)

# Save the Neural Network for later use
with open('nn.obj','wb') as w:
    pickle.dump(nn,w)