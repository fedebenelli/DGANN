import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np

def relu(X):
    return np.log(1+np.exp(X))

def relu_derivative(X):
    return 1/(1+np.exp(-X))

class NeuralNetwork:
    def __init__(self, x, y, nodes, iterations, maxes):
        """
        Neural Network class
        self.input      = x = np.array() with the X values
        self.y          = y = np.array() with the Y (real data) values
        self.nodes      = amount of nodes that the internal layers will have
        self.output     = estimated y values
        self.iterations = amount of iterations to train the NN
        self.loss       = sum of squares of the diff between output values and real data
        """
        np.random.seed(20)
        self.input      = x
        self.maxes      = maxes
        self.weights1   = np.random.rand(self.input.shape[1], nodes) 
        self.weights2   = np.random.rand(nodes,5)                 
        self.y          = y
        self.output     = np.zeros(y.shape)
        self.iterations = iterations
        self.loss       = ((self.output-self.y)**2).sum()
    
    def feedforward(self):
        self.layer1 = relu(np.dot(self.input, self.weights1))
        self.output = relu(np.dot(self.layer1, self.weights2))
    
    def backprop(self):
        """
        Recalculate the weights, based on the gradients of the relu function
        """
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * relu_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * relu_derivative(self.output), self.weights2.T) * relu_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1/100
        self.weights2 += d_weights2/1000
    
    def train(self):
        """
        Method that will feedforward and backrop in a series of `self.iterations` number of iterations.
        Also a `self.loss_list` list will be made, wich contains a list of the `self.loss` value for each iteration.
        """
        self.loss_list = []
        for _ in range(self.iterations):
            self.feedforward()
            self.backprop()
            self.loss =((self.y-self.output)**2).sum()
            self.loss_list.append(self.loss)

    def estimate(self, values):
        """
        Receives an input of gases; as a dictionary but must be sorted as:
        ["H2","CH4","C2H6","C2H2","CO","CO2"]

        {
            "PD"  : 23,
            "D1"  : 75,
            "D2"  : 50,
            "T1y2": 1,
            "T3"  : 2
        }
        Then it returns a dictionary with the estimated values, in the form of:
        """
        faults = {
            0: "PD",
            1: "D1",
            2: "D2",
            3: "T1y2",
            4: "T3"
        }

        for i,gas in enumerate(self.maxes.keys()):
            values[i] = values[i]/self.maxes[gas]

        gases = np.array(values)
        self.input = gases
        self.feedforward()

        estimates = dict()
        for i in range(len(self.output)):
            estimates[i] = round(self.output[i]*100,2)
            estimates[faults[i]] = estimates.pop(i)
        return estimates
        
    def results(self, df, keys):
        # Tags of keys that are analyzed
        keys_y = keys
        keys   = keys[:-1]

        # Make lists of the real values and the predictions for comparison
        real = []
        pred = []
        iterations = self.iterations
        for pos in range(df.shape[0]):
            self.input = df[keys].iloc[pos]
            self.feedforward()
            maxval = -10
            for i, val in enumerate(self.output):
                if val > maxval:
                    maxval = val
                    index = i
            real.append(int(df[keys_y].iloc[pos]['fault']))
            pred.append(index)

        step = int(np.log10(iterations)*10)
        l2 = []
        count = 0
        mean  = sum([i for i in self.loss_list[:step]])/step
        for i in range(len(self.loss_list)):
            l2.append(mean)
            count += self.loss_list[i]
            if i % step == 0:
                mean = count/(step)
                count = 0
        plt.plot(self.loss_list,lw='0.01',color='gray')
        plt.plot(l2)
        plt.ylim(min(l2),100)
        plt.savefig('convergence.svg')
        plt.show()
        
        plt.figure(figsize=(5,5))
        sns.regplot(real,pred, scatter_kws={'alpha':0.15})
        plt.savefig('precision.svg')
        count=0
        for cond in ((np.array(pred) - np.array(real)) == 0):
            if cond:
                count+=1
        print('% Match: ', count/len(np.array(pred))*100)
        plt.show()