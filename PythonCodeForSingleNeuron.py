#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function (used for backpropagation)
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Single Neuron Model
class SingleNeuron:
    def __init__(self):
        # Initialize random weights and bias
        self.weight = np.random.rand()
        self.bias = np.random.rand()

    # Forward propagation: Calculate output based on input x
    def forward(self, x):
        z = self.weight * x + self.bias  # Linear combination
        return sigmoid(z)                # Activation function (Sigmoid)
    
    # Train the neuron with a simple learning rule (Gradient Descent)
    def train(self, X, y, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            for i in range(len(X)):
                # Forward pass
                prediction = self.forward(X[i])

                # Error computation
                error = y[i] - prediction

                # Backpropagation: Update weight and bias
                self.weight += learning_rate * error * X[i] * sigmoid_derivative(prediction)
                self.bias += learning_rate * error * sigmoid_derivative(prediction)
            
            if epoch % 100 == 0:
                loss = np.mean((y - self.forward(X))**2)
                print(f"Epoch {epoch}, Loss: {loss}")
    
    # Predict on new input
    def predict(self, x):
        output = self.forward(x)
        return 1 if output >= 0.5 else 0  # Binary classification (Threshold 0.5)

# Example usage
if __name__ == "__main__":
    # Simple dataset: AND logic gate
    X = np.array([0, 0, 1, 1])  # Inputs
    y = np.array([0, 0, 0, 1])  # Outputs (AND logic)

    neuron = SingleNeuron()
    neuron.train(X, y, learning_rate=0.1, epochs=1000)

    # Test predictions
    for i in range(len(X)):
        print(f"Input: {X[i]}, Predicted Output: {neuron.predict(X[i])}")


# In[ ]:




