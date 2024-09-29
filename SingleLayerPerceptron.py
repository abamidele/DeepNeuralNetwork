#!/usr/bin/env python
# coding: utf-8


import numpy as np

# Perceptron class
class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(num_inputs + 1)  # Including bias (extra weight for bias)
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Step activation function (binary output)
    def activation(self, weighted_sum):
        return 1 if weighted_sum >= 0 else 0

    # Training the Perceptron using the Perceptron Learning Rule
    def train(self, inputs, labels):
        for epoch in range(self.epochs):
            for input_vec, label in zip(inputs, labels):
                # Compute the weighted sum (dot product of inputs and weights, including bias)
                weighted_sum = np.dot(input_vec, self.weights[1:]) + self.weights[0]
                prediction = self.activation(weighted_sum)

                # Perceptron weight update rule
                self.weights[1:] += self.learning_rate * (label - prediction) * input_vec
                self.weights[0] += self.learning_rate * (label - prediction)  # Update bias weight

    # Predict function for unseen data
    def predict(self, input_vec):
        weighted_sum = np.dot(input_vec, self.weights[1:]) + self.weights[0]
        return self.activation(weighted_sum)

# Example usage: AND gate with Perceptron
if __name__ == "__main__":
    # Training data for AND gate
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 0, 0, 1])  # AND gate outputs

    # Initialize perceptron with 2 inputs
    perceptron = Perceptron(num_inputs=2, learning_rate=0.1, epochs=10)

    # Train the perceptron
    perceptron.train(inputs, labels)

    # Test the trained perceptron
    print("Perceptron Output for AND gate:")
    for input_vec in inputs:
        output = perceptron.predict(input_vec)
        print(f"Input: {input_vec}, Output: {output}")






