import numpy as np

class Loss:
    def calculate(self, modelOutput, targets):
        sampleLosses = self.forward(modelOutput, targets)
        dataLoss = np.mean(sampleLosses)
        return dataLoss

# Categorical cross entropy loss calculator
class CCE_Loss(Loss):  
      
    # y_pred is a 2d array of probability distributions, one per batch entry
    # y_true is a 2d array where the shape is the same as the output of the neural network    
    # Multiply the negative log of each input by the corresponding element in the target array
    def forward(self, y_pred, y_true):
        numSamples = (len(y_pred))
        
        # Clip to prevent /0
        clippedPredictions = np.clip(y_pred, 1e-7, 1-1e-7)
        
        # Calculate predictions for one hot or discrete
        if len(y_true.shape) == 1:
            correctConfidences = clippedPredictions[range(numSamples), y_true]
        elif len(y_true.shape) == 2:
            correctConfidences = np.sum(clippedPredictions*y_true, axis=1)
        
        self.sampleLosses = -np.log(correctConfidences)
        
        return self.sampleLosses
    
    def backward(self, dvalues, y_true):
        # Number of samples and labels
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        # Create one-hot encoded vectors if the true data isn't already one-hot encoded
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate and normalize the derivative
        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs / samples
        