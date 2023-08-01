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

 ## Backpropagation explanation

The backpropagation algorithm is used to calculate the gradient of the loss function with respect to the weights and biases of the network. This gradient is then used to update the weights and biases of the network in the direction of the minimum of the loss function.

After forwarding the input, we compute the the gradient to update the weights and biases. Here is the code :

```rust
pub fn backward(&mut self, output_error: Array2<f64>, learning_rate: f64) -> Array2<f64> {
    let output_error = self.activation.apply_der(&self.output).t().to_owned() * output_error;
    let inp_err = output_error.dot(&self.weights);
    let weight_err = self.input.dot(&output_error);
    self.weights = &self.weights - learning_rate * &weight_err.t();
    self.bias = &self.bias - learning_rate * &output_error.t();
    inp_err
}
```

We need the compute this values : ![latex](https://latex.codecogs.com/gif.image?%5Cdpi%7B110%7D%5Cbg%7Bwhite%7D%5Cfrac%7B%5Cpartial%20c%7D%7B%5Cpartial%20W%5Ei%7D,%5Cfrac%7B%5Cpartial%20c%7D%7B%5Cpartial%20B%5Ei%7D)
Where W is the matrix of weights and B the vector of biases for the i-th layer and c the cost function.

We can compute the first one with the chain rule :

![latex](https://latex.codecogs.com/svg.image?%5Cdpi%7B110%7D%5Cbg%7Bwhite%7D%5Cfrac%7B%5Cpartial%20c%7D%7B%5Cpartial%20W%5Ei%7D%20%3D%20%5Cfrac%7B%5Cpartial%20c%7D%7B%5Cpartial%20a%5Ei%7D%20%5Cfrac%7B%5Cpartial%20a%5Ei%7D%7B%5Cpartial%20z%5Ei%7D%20%5Cfrac%7B%5Cpartial%20z%5Ei%7D%7B%5Cpartial%20W%5Ei%7D)

Where a is the activation function and z the output of the layer.

To calculate it, it is easier to look at one weight at a time. Let's take the weight ![latex](https://latex.codecogs.com/svg.image?%5Cdpi%7B110%7D%5Cbg%7Bwhite%7DW_%7Bjk%7D%5Ei) for example. We have :