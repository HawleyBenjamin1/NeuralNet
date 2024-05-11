import unittest
from framework import layer as l
import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from framework import activation, loss, optimizer

class TestLayerMethods(unittest.TestCase):
    def test_getOutput(self):
        inputData = np.array([[1, 2, 3, 2.5], 
                     [2.0, 5.0, -1.0, 2.0],
                     [-1.5, 2.7, 3.3, -0.8]])
        
        layer1 = l.layer(4, 5)
        layer2 = l.layer(5, 2)
        
        layer1.forward(inputData)
        layer2.forward(layer1.output)
        
        # print(layer1.weights, "\n", layer1.biases)
        
        self.assertTrue(layer1.output.shape == (3, 5))
        self.assertTrue(layer2.output.shape == (3, 2))
    
    def test_nnfsSpiralData(self):
        X, y = spiral_data(100, 3)
        # plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
        # plt.show()
        self.assertTrue(X.shape == (300, 2))
        
    def test_rectifiedLinear(self):
        X, y = spiral_data(100, 3)
        
        activation1 = activation.activation_ReLU()
        layer1 = l.layer(2,5)
        
        layer1.forward(X)
        activation1.forward(layer1.output)
        
    def test_softMax(self):
        layerOutput = [[4.8, 1.21, 2.385],
                  [8.9, -1.81, 0.2],
                  [1.41, 1.051, 0.0266]]
        
        activation1 = activation.activation_softMax()
        activation1.forward(layerOutput)
        
        expValues = np.exp(layerOutput - np.max(layerOutput, axis=1, keepdims=True)) 
        normValues = expValues / np.sum(expValues, axis=1, keepdims=True)
        
        self.assertTrue(np.array_equal(normValues, activation1.output))
        
        # values = expValues
        
        # badNorm = values / np.sum(values, axis=1)
        # goodNorm = values / np.sum(values, axis=1, keepdims=True)
        
        # nonsense = np.sum(badNorm, axis=1)
        # expectedNormSum = np.sum(goodNorm, axis=1)
        
        
        # print("What happens if I sum up my normalized values without retaing dimension during division: \n", nonsense)
        # print("What I would expect to see after summing normalized values: \n", expectedNormSum)
        
    def test_forwardPass(self):
        X, y = spiral_data(samples=100, classes=3)
        
        layer1 = l.layer(2, 4)
        layer2 = l.layer(4, 4)
        layer3 = l.layer(4, 3)

        activation1 = activation.activation_ReLU()
        activation2 = activation.activation_ReLU()
        
        activationLast = activation.activation_softMax()
        
        layer1.forward(X)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        layer3.forward(activation2.output)
        activationLast.forward(layer3.output)
        
        self.assertEqual(activationLast.output.shape, (300, 3))
        # print("Layer1: ", layer1.output)
        # print("Activation1: ", activation1.output)
        # print("Layer2: ", layer2.output)
        # print("Activation2: ", activation2.output)
        # print("Layer3: ", layer3.output)
        # print("Final output: ", activationLast.output)
    
    def test_lossCCE(self):
        inputs = np.array([0.7, 0.1, 0.2])
        target_output = np.identity(3)
        target_output_1d = np.array([0,1,2])
        
        layer1 = l.layer(3, 4)
        layer2 = l.layer(4, 4)
        layer3 = l.layer(4, 3)

        activation1 = activation.activation_ReLU()
        activation2 = activation.activation_ReLU()
        
        activationLast = activation.activation_softMax()
        
        layer1.forward(inputs)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        layer3.forward(activation2.output)
        activationLast.forward(layer3.output)
        
        lossCalculator = loss.CCE_Loss()
        meanLosses = lossCalculator.calculate(activationLast.output, target_output)
        
        self.assertEqual(lossCalculator.sampleLosses.shape, (3,))
        
        meanLosses1d = lossCalculator.calculate(activationLast.output, target_output_1d)
        self.assertEqual(lossCalculator.sampleLosses.shape, (3,))
        
        # print(meanLosses1d, meanLosses)
        
    def test_accuracy(self):
        softmax_output = np.array([[0.7, 0.1, 0.2], [.05, 0.1, 0.4], [0.02, 0.9, 0.08]])
        target_output = np.array([0,1,1])
        
        predictions = np.argmax(softmax_output, axis=1)
        
        if len(target_output.shape) == 2:
            target_output = np.argmax(target_output, axis=1)
        
        accuracy = np.mean(target_output == predictions)
        # print(accuracy, predictions, target_output)
        
    def test_backprop(self):
        # Assuming forward pass complete
        softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                    [0.1, 0.5, 0.4],
                                    [0.02, 0.9, 0.08]])
        class_targets = np.array([0, 1, 1])
        
        softmax_loss = activation.activation_softmax_loss_CCE()
        softmax_loss.backward(softmax_outputs, class_targets)
        dvalues1 = softmax_loss.dinputs
        
        lossActivation = activation.activation_softMax()
        lossActivation.output = softmax_outputs
        lossOutput = loss.CCE_Loss()
        lossOutput.backward(softmax_outputs, class_targets)
        lossActivation.backward(lossOutput.dinputs)
        dvalues2 = lossActivation.dinputs
        
        # print('Gradients: combined loss and activation:')
        # print(dvalues1)
        # print('Gradients: separate loss and activation:')
        # print(dvalues2)
        
    
    def test_optimizer(self):
        X, y = spiral_data(100, 3)
        
        # Just one layer for now
        layer1 = l.layer(2, 64)
        activation1 = activation.activation_ReLU()
        layer2 = l.layer(64, 3)
        softmax = activation.activation_softmax_loss_CCE()
        opt = optimizer.SGD(.5)
        
        for epoch in range(10001):
            # Forward Pass
            layer1.forward(X)
            activation1.forward(layer1.output)
            layer2.forward(activation1.output)
            loss = softmax.forward(layer2.output, y)
            
            # Calculate accuracy
            predictions = np.argmax(softmax.output, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            acc = np.mean(y==predictions)
            
            if not epoch % 100:
                print(f"Epoch: {epoch}, " + 
                      f"Loss: {loss:.3f}, " +
                      f"Acc: {acc:.3f}")
            
            # Backward pass
            softmax.backward(softmax.output, y)
            layer2.backward(softmax.dinputs)
            activation1.backward(layer2.dinputs)
            layer1.backward(activation1.dinputs)
            
            # Update layers using optimizer
            opt.updateParamsMomentum(layer1)
            opt.updateParamsMomentum(layer2)
        
        
        fig, (real, pred) = plt.subplots(1,2)
        real.scatter(X[:,0], X[:,1], c=y, cmap="brg")
        real.set_title('Expected')
        
        if len(softmax.output.shape) == 2:
            predictions = np.argmax(softmax.output, axis=1)
            
        pred.scatter(X[:,0], X[:,1], c=predictions, cmap="brg")
        pred.set_title('Predicted')
        
        plt.show()
        
        

if __name__ == '__main__':
    unittest.main()