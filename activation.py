import numpy as np
import loss

class activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        
        self.output = np.maximum(0, inputs)
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        
        self.dinputs[self.inputs <= 0] = 0
        
class activation_softMax:
    def forward(self, inputs):
        
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
        probabilities = expValues / np.sum(expValues, axis=1, keepdims=True)
        
        self.output = np.array(probabilities)
        
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            
    
class activation_softmax_loss_CCE:
    def __init__(self):
        self.activation = activation_softMax()
        self.loss = loss.CCE_Loss()
    
    # Returns loss after forward through softmax at output layer.
    # Input is output of final hidden layer.
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        
        return self.loss.calculate(self.output, y_true)
    
    # Calculates gradient based on output of the forward pass 
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        
        # Discrete true labels for the little trick afor calculating gradient up next
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        
        # Nice trick where since derivative = y_pred - y_true and y_true is always a value of 1
        # I can just access each dinput predicted value and subtract 1 to get gradient array.
        self.dinputs[range(samples), y_true] -= 1
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples
