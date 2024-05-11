import numpy as np

class layer:
    def __init__(self, n_inputs, n_neurons):
        self.shape = [n_inputs, n_neurons]
        self.weights = 0.1*self.getRandomArray(n_inputs, n_neurons)
        self.biases = np.zeros(shape=(1, n_neurons))
        self.velocityW = 0
        self.velocityB = 0
        
    def forward(self, inputs):
        self.inputs = inputs
        
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        self.dinputs = np.dot(dvalues, self.weights.T)
        
    @staticmethod
    def getRandomArray(rows, cols):
        rng = np.random.default_rng(seed=0)
        rng.standard_normal()
        
        return rng.standard_normal(size=(rows, cols))