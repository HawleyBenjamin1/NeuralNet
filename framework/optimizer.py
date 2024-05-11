class optimizer():
    def __init__(self, learningRate=1.0):
        self.learningRate = learningRate
        

class SGD(optimizer):
    def updateParams(self, layer):
        # Subtract gradient times learning rate from the weights in layer
        layer.weights += -self.learningRate * layer.dweights
        layer.biases += -self.learningRate * layer.dbiases
    
    def updateParamsMomentum(self, layer, momentum=0.9):
        velocityW = momentum * layer.velocityW - self.learningRate * layer.dweights
        velocityB = momentum * layer.velocityB - self.learningRate * layer.dbiases
        
        layer.weights += velocityW
        layer.biases += velocityB
        
        layer.velocityW = velocityW
        layer.velocityB = velocityB
        
        