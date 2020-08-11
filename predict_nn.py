from NeuralNetwork import *
from pprint import pprint
import numpy as np
import pickle

# Load the desired NN
with open('nn.obj','rb') as f:
    nn = pickle.load(f)

def main():
    while True:
        user_input = []
        for gas in nn.maxes:
            user_input.append(float(input(f'Cantidad de gas {gas}: ')))
        certainity = nn.estimate(user_input)
        pprint(certainity,width=1)

if __name__ == "__main__":
    main()