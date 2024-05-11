from ucimlrepo import fetch_ucirepo, list_available_datasets
import pandas as pd
import numpy as np
import time
from framework import layer as l
from framework import optimizer, activation, modelTrainer


def initializeDataSet():
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
    
    # data (as pandas dataframes) 
    X = heart_disease.data.features 
    y = heart_disease.data.targets 
    
    # # metadata 
    # print(heart_disease.metadata) 
    
    # # variable information 
    # print(heart_disease.variables) 
    
    return X, y


def cleanData():
    X, y = initializeDataSet()
    
    # Make it so any indication of heart disease (y>0) is it's own category.
    # Everybody else does this apparently. Model is predicting poorly when there are too many categories.
    y.mask(y > 0, 1, inplace=True)
    
    
    joinedSets = X.join(y)
    
    cleanedValues = joinedSets.dropna(axis=0)
    
    cleanedValues.rename(columns={"num": "targets"}, inplace=True)
    
    return cleanedValues
            
            
def trainTestSplit():
    allData = cleanData()
    
    trainingData = allData.sample(frac=0.7, axis=0, random_state=5)
    testData = allData.drop(index=trainingData.index)
    
    trainingData.reset_index(drop=True, inplace=True)
    testData.reset_index(drop=True, inplace=True)
    
    return trainingData, testData

# Defunct
# def trainModel(trainingData):
    actual = trainingData.pop('num')
    
    layer1 = l.layer(trainingData.shape[1], 64)
    activation1 = activation.activation_ReLU()
    layer2 = l.layer(64, 128)
    activation2 = activation.activation_ReLU()
    layer3 = l.layer(128, 2)
    softmax = activation.activation_softmax_loss_CCE()
    opt = optimizer.SGD(.0001)
    
    for epoch in range(5001):
        # Forward Pass
        layer1.forward(trainingData.to_numpy())
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        layer3.forward(activation2.output)
        loss = softmax.forward(layer3.output, actual)
        
        # Calculate accuracy
        predictions = np.argmax(softmax.output, axis=1)
        if len(actual.shape) == 2:
            actual = np.argmax(actual, axis=1)
        acc = np.mean(actual==predictions)
        
        if not epoch % 100:
            print(f"Epoch: {epoch}, " + 
                    f"Loss: {loss:.3f}, " +
                    f"Acc: {acc:.3f}")
        
        # Backward pass
        softmax.backward(softmax.output, actual)
        layer3.backward(softmax.dinputs)
        activation2.backward(layer3.dinputs)
        layer2.backward(activation2.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)
        
        # Update layers using optimizer
        opt.updateParamsMomentum(layer1)
        opt.updateParamsMomentum(layer2)
        opt.updateParamsMomentum(layer3)
        
    packagedModel = [layer1, layer2, layer3]
    return packagedModel

def trainModel(shapes, trainingData, epochs, learningRate, batchSize):
    return 

# Defunct
# def applyModel(model, test):
    targets = test.pop('num')
    lastLayerIndex = len(model) - 1
    
    # Pass inputs to first layer
    model[0].forward(test.to_numpy())
    
    # Forward pass through subsequent layers
    activationFunc = activation.activation_ReLU()
    for layer in range(1, lastLayerIndex + 1):
        activationFunc.forward(model[layer - 1].output)
        model[layer].forward(activationFunc.output)
    
    # Extract loss and predictions
    lossFunc = activation.activation_softmax_loss_CCE()
    lossFunc.forward(model[lastLayerIndex].output, targets)
    
    # Calculate accuracy
    predictions = np.argmax(lossFunc.output, axis=1)
    if len(targets.shape) == 2:
        targets = np.argmax(targets, axis=1)
    acc = np.mean(targets==predictions)
    
    predictions = pd.DataFrame(predictions).join(targets)
    
    print(f"Overall Accuracy: {acc} \n",
          f"Predictions vs Actual: \n {predictions}")
    
def applyModel(model, testData):
    print("Running test ...")
    targets = testData.pop('targets')
    numLayers = len(model)
    
    model[0].forward(testData.to_numpy())
    for layerIndex in range(1, numLayers):
        if layerIndex == numLayers - 1:
            model[layerIndex].forward(model[layerIndex - 1].output, targets)
        else:
            model[layerIndex].forward(model[layerIndex - 1].output)
    
    predictions = np.argmax(model[-1].output, axis=1)
    if len(targets.shape) == 2:
        flatTargets = np.argmax(targets, axis=1)
    else:
        flatTargets = targets
        
    accuracy = np.mean(targets==predictions)*100
    
    # for target, predicted in zip(flatTargets, predictions):
    #     print(
    #         f"Target: {target}",
    #         f"Predicted: {predicted}"
    #     )
        
    print(f"Overall accuracy: {accuracy:.3f}")
        
def runModelTest():
    startTime = time.time()
    
    training, test = trainTestSplit()
    shapes = [[training.shape[1] - 1, 64], [64, 128], [128, 2]]
    epochs = 10001
    learningRate = .0001
    batchSize = 128
    heartDiseasePredictor = modelTrainer.modelTrainer.trainModel(shapes, training, epochs, learningRate=learningRate, batchSize=batchSize)
    applyModel(heartDiseasePredictor, test)
    
    print("--- %.2f seconds ---" % (time.time() - startTime))
