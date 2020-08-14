from modules.NeuralNetwork import NeuralNetwork,relu,relu_derivative
from pprint import pprint
import numpy as np
import pickle

# Load the desired NN
with open('nn.obj','rb') as f:
    nn = pickle.load(f)

def get_prediction(input):
    certainity = nn.estimate(input)
    return certainity

def main():
    while True:
        user_input = [
            float(input(f'Cantidad de gas {gas}: ')) for gas in nn.maxes
            ]
        pprint(get_prediction(user_input),width=1)

if __name__ == "__main__":
    main()