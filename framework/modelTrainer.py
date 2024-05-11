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
    def trainModel(layerShapes: list[LayerShape], trainingData, epochs=1001, learningRate=0.1, batchSize=128):
        # Instantiate layers and activation functions
        # This creates an extra activation func, so pop the last one off and replace it with softmax
        layers = []
        activaters = []
        for layer in layerShapes:
            layers.append(l.layer(layer[0], layer[1]))
            activaters.append(a.activation_ReLU())
        numLayers = len(layers)
        
        activaters.pop()
        activaters.append(a.activation_softmax_loss_CCE())
        
        # Put data in batches
        batches = modelTrainer.batchData(trainingData, batchSize)
        batchNum = 1
        for batch in batches:
            # Instantiate optimizer
            optimizer = o.SGD(learningRate)
            
            # Pop off targets column from the batch
            targets = batch.pop('targets')
            
            for epoch in range(epochs):
                # Pass data to input layer then forward to first activation func
                layers[0].forward(batch.to_numpy())
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
                    
                 # Calculate accuracy and report after every 1000 epochs
                if not epoch % 1000:
                    predictions = np.argmax(activaters[-1].output, axis=1)
                    if len(targets.shape) == 2:
                        flatTargets = np.argmax(targets, axis=1)
                        accuracy = np.mean(flatTargets==predictions)*100
                    else:
                        accuracy = np.mean(targets==predictions)*100
                    
                    print(f"Batch: {batchNum}, " +
                            f"Epoch: {epoch}, " + 
                            f"Loss: {loss:.3f}, " +
                            f"Acc: {accuracy:.3f}")
                    
                    if loss < 1:
                        optimizer.learningRate = .0001
                    if accuracy > 90:
                        optimizer.learningRate = .00001
            
            batchNum += 1

        # Return a list of trained layers and associated activation functions in order
        finalOutput = []
        for layer, activation in zip(layers, activaters):
            finalOutput.append(layer)
            finalOutput.append(activation)
            
        return finalOutput
    
    
    # Takes complete dataset and returns batched data as a set of numpy arrays
    # Uses divide and conquer pattern
    def batchData(toSplit, batchSize=128):
        df = toSplit.copy()
        batches = []
        
        def split(data):
            if data.shape[0] <= batchSize:
                batches.append(data)
            else:
                mid = data.shape[0] // 2
                left = data.iloc[:mid,:]
                right = data.iloc[mid:,:]
                split(left)
                split(right)
                
        split(df)
        
        return batches