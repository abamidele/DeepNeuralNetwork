#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# McCulloch-Pitts Neuron class
class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        # Initialize weights and threshold
        self.weights = np.array(weights)
        self.threshold = threshold

    # Activation function: step function (binary output)
    def activation(self, weighted_sum):
        return 1 if weighted_sum >= self.threshold else 0

    # Forward propagation: compute weighted sum and apply activation
    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)  # Weighted sum
        return self.activation(weighted_sum)

# Example usage: Implementing a logical AND gate using McCulloch-Pitts neuron
if __name__ == "__main__":
    # Weights for the inputs (AND gate: both inputs must be 1 to output 1)
    weights = [1, 1]
    
    # Threshold for the neuron to fire (AND gate needs sum to be >= 2)
    threshold = 2

    # Create a McCulloch-Pitts Neuron
    neuron = McCullochPittsNeuron(weights, threshold)

    # Input dataset (AND gate logic inputs)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Predict outputs
    print("AND Gate Output:")
    for i in inputs:
        output = neuron.forward(i)
        print(f"Input: {i}, Output: {output}")

    # Example: You can adjust weights and threshold to simulate other logic gates (OR, NOT)


# In[ ]:




