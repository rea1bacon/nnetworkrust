# A functionnal neural network written in rust

This is a simple neural network written in rust. It is not optimized for speed and efficiency.
It was written as a learning experience.

Gets around 90% accuracy on the MNIST dataset with the following parameters:
- 5 layers
```
first layer : 784 inputs, 50 neurons, activation function : sigmoid
second layer : 50 inputs, 100 neurons, activation function : sigmoid
third layer : 100 inputs, 60 neurons, activation function : tanh
fourth layer : 60 inputs, 40 neurons, activation function : sigmoid
fifth layer : 40 inputs, 10 neurons, activation function : softmax
```
- batch size: 100
- learning rate: 0.2