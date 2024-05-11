from typing import NewType
import numpy as np
from framework import layer as l
from framework import activation as a
from framework import optimizer as o

# User defined type for layershapes in type suggestion
LayerShape = NewType('LayerShape', tuple[int, int])

# Takes training data in the form of a pandas array. 
# Will update to rectify and use Numpy in the future. 
class modelTrainer:
    @staticmethod
    def trainModel(layerShapes: list[LayerShape], trainingData, targets, epochs=2001, learningRate=0.5):
        # Instantiate optimizer for later use
        optimizer = o.SGD(learningRate)
        
        # Instantiate layers and activation functions
        # This creates an extra activation func, so pop the last one off and replace it with softmax
        layers = []
        activaters = []
        for layer in layerShapes:
            layers.append(l.layer(layer[0], layer[1]))
            activaters.append(a.activation_ReLU())
        
        activaters.pop()
        activaters.append(a.activation_softmax_loss_CCE())
        
        numLayers = len(layers)
        
        for epoch in range(epochs):
            # Pass data to input layer then forward to first activation func
            layers[0].forward(trainingData.to_numpy())
            activaters[0].forward(layers[0].output)
            
            # Forward pass through subsequent layers and activation functions
            for layerIndex in range(1, numLayers):
                layers[layerIndex].forward(activaters[layerIndex - 1].output)
                
                # If the loop reaches the last activation function, treat as softmax loss
                if layerIndex < numLayers - 1:
                    activaters[layerIndex].forward(layers[layerIndex].output)
                else:
                    loss = activaters[layerIndex].forward(layers[layerIndex].output, targets)
                
            # Now backpropagate 
            # Include optimization to reduce iterations
            for layerIndex in reversed(range(numLayers)):
                # Check for last layer and treat as such
                if layerIndex == numLayers - 1:
                    activaters[layerIndex].backward(activaters[layerIndex].output, targets)
                else:
                    activaters[layerIndex].backward(layers[layerIndex + 1].dinputs)
                    
                layers[layerIndex].backward(activaters[layerIndex].dinputs)
                    
                optimizer.updateParams(layers[layerIndex])
                
            # Calculate accuracy and report every 100 epochs
            if not epoch % 100:
                predictions = np.argmax(activaters[-1].output, axis=1)
                if len(targets.shape) == 2:
                    flatTargets = np.argmax(targets, axis=1)
                    accuracy = np.mean(flatTargets==predictions)
                else:
                    accuracy = np.mean(targets==predictions)
                
                print(f"Epoch: {epoch}, " + 
                        f"Loss: {loss:.3f}, " +
                        f"Acc: {accuracy:.3f}")

        
        # Return a list of trained layers and associated activation functions in order
        finalOutput = []
        for layer, activation in zip(layers, activaters):
            finalOutput.append(layer)
            finalOutput.append(activation)
            
        return finalOutput