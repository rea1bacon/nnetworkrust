# A functionnal neural network written in rust

This is a simple neural network written in rust. It is not optimized for speed and efficiency.
It was written as a learning experience.

Gets around 94% accuracy on the MNIST dataset with the following parameters:
- 3 layers
```
first layer : 784 inputs, 100 neurons, activation function : sigmoid
second layer : 100 inputs, 512 neurons, activation function : Relu
third layer : 512 inputs, 10 neurons, activation function : softmax
```
- batch size: 100
- learning rate: 0.01
- loss function: cross entropy

 ## Backpropagation explanation

The backpropagation algorithm is used to calculate the gradient of the loss function with respect to the weights and biases of the network. This gradient is then used to update the weights and biases of the network in the direction of the minimum of the loss function.

After forwarding the input, we compute the gradient to update the weights and biases. Here is the code :

```rust
pub fn backward(&mut self, output_error: Array2<f64>, learning_rate: f64) -> Array2<f64> {
    let output_error = self.activation.apply_der(&self.output) * output_error;
    let inp_err = self.weights.t().dot(&output_error);
    let weight_err = output_error.dot(&self.input.t());
    self.weights = &self.weights - learning_rate * &weight_err;
    self.bias = &self.bias - learning_rate * &output_error;
    inp_err
}
```
Let $X^i$ be the input of the layer $i$ and $Z^i$ the output of the layer $i$ before the activation function and $A^i$ the output of the layer $i$ after the activation function.

We have $A^i = \sigma(Z^i)$ where $\sigma$ is the activation function and $Z^i = W^i X^i + B^i$. Where $W^i$ is the matrix of weights and $B^i$ the vector of biases for the $i$-th layer and $c$ the cost function.

$j$ represents the number of outputs and $k$ the number of inputs of the layer $i$. 

We need to compute this values : $\frac{\partial c}{\partial W^i}$ and $\frac{\partial c}{\partial B^i}$. We will assume that  $\frac{\partial c}{\partial A^i}$ is known (which is equal to $\frac{\partial c}{\partial X^{i+1}}$)



For the layer $i$, we represent the matrix of weights as  <!-- $W^i=\begin{bmatrix}w_{1,1}&.&.&w_{1,k}\\.&.&.&.\\.&.&.&.\\w_{j,1}&.&.&w_{j,k}\\\end{bmatrix}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://github.com/rea1bacon/nnetworkrust/blob/main/svg/YWSRPMJ6l7.png">
and the vector of biases as $B^i=\begin{bmatrix}b_1\\.\\.\\.\\b_j\\\end{bmatrix}$

So we know that $ \frac{\partial c}{\partial W^i} = \begin{bmatrix}\frac{\partial c}{\partial w_{1,1}^i}&.&.&\frac{\partial c}{\partial w_{1,k}^i}\\.&.&.&.\\.&.&.&.\\\frac{\partial c}{\partial w_{j,1}^i}&.&.&\frac{\partial c}{\partial w_{j,k}^i}\\\end{bmatrix}$ (partial derivative of a scalar with respect to a matrix)

Now let's calculate $\frac{\partial c}{\partial w_{1,1}^i}$ for example.

By the chain rule, we have $\frac{\partial c}{\partial w_{1,1}^i} = \frac{\partial c}{Z^i}\frac{\partial Z^i}{\partial w_{1,1}^i}$.

We know that $\frac{\partial Z^i}{\partial w_{1,1}^i} = \begin{bmatrix}\frac{\partial Z_1^i}{\partial w_{1,1}^i}\\.\\.\\\frac{\partial Z_j^i}{\partial w_{1,1}^i}\\\end{bmatrix}$ which is equal to $\begin{bmatrix}X_1^i\\0\\.\\.\\0\\\end{bmatrix}$ because $Z^i = \begin{bmatrix}w_{1,1}&.&.&w_{1,k}\\.&.&.&.\\.&.&.&.\\w_{j,1}&.&.&w_{j,k}\\\end{bmatrix} \begin{bmatrix}X_1^i\\.\\.\\.\\X_k^i\\\end{bmatrix} + \begin{bmatrix}b_1\\.\\.\\.\\b_j\\\end{bmatrix}$ so the partial derivative of the biases with respect to $w_{1,1}^i$ will be equal to 0.

Also $\begin{bmatrix}w_{1,1}^i&.&.&w_{1,k}^i\\.&.&.&.\\.&.&.&.\\w_{j,1}^i&.&.&w_{j,k}^i\\\end{bmatrix} \begin{bmatrix}X_1^i\\.\\.\\.\\X_k^i\\\end{bmatrix} = \begin{bmatrix}w_{1,1}^iX_1^i + ... + w_{1,k}^iX_k^i\\.\\.\\.\\w_{j,1}^iX_1^i + ... + w_{j,k}^iX_k^i\\\end{bmatrix}$ so the partial derivative with respect to $w_{1,1}^i$ will be equal to $\begin{bmatrix}X_1^i\\0\\.\\.\\0\\\end{bmatrix}$.

So now we have $\frac{\partial c}{\partial w_{1,1}^i} = \frac{\partial c}{\partial A^i}\frac{\partial A^i}{\partial Z^i}\begin{bmatrix}X_1^i\\0\\.\\.\\0\\\end{bmatrix} = \begin{bmatrix}\frac{\partial c }{\partial Z^i_1} \times X^i_1\end{bmatrix}$ We can generalize this to $\forall (a,b) \in [1,j]\times[1,k], \frac{\partial c}{\partial w_{a,b}^i} = \frac{\partial c }{\partial Z^i_a} \times X^i_b$. 

We finally have $\frac{\partial c}{\partial W^i} = \begin{bmatrix}\frac{\partial c }{\partial Z^i_1} \times X^i_1&\cdots&\frac{\partial c }{\partial Z^i_k} \times X^i_1\\\vdots&\ddots&\vdots\\\frac{\partial c }{\partial Z^i_1} \times X^i_j&\cdots&\frac{\partial c }{\partial Z^i_k} \times X^i_j\\\end{bmatrix}$ which can be simplified to $\frac{\partial c}{\partial Z^i} \cdot {X^i}^\intercal = \begin{bmatrix}\frac{\partial c}{\partial Z^i_1}\\\vdots\\\frac{\partial c}{\partial Z^i_1}\end{bmatrix}\cdot \begin{bmatrix}X^i_1&\cdots&X^i_j\end{bmatrix}$

We now need to compute $\frac{\partial c}{\partial B^i}$.

We know that $\frac{\partial c}{\partial B^i} = \begin{bmatrix}\frac{\partial c}{\partial b_1^i}\\.\\.\\.\\\frac{\partial c}{\partial b_j^i}\\\end{bmatrix}$

We have $\frac{\partial c}{\partial b_1^i} = \frac{\partial c}{\partial Z^i}\frac{\partial Z^i}{\partial b_1^i}$. We can easily notice that $\frac{\partial Z^i}{\partial b_1^i} = \begin{bmatrix}1\\0\\.\\.\\0\\\end{bmatrix}$. So $\frac{\partial c}{\partial b_1^i} = \frac{\partial c}{\partial Z^i_1}$. We can generalize this to $\forall a \in [1,j], \frac{\partial c}{\partial b_a^i} = \frac{\partial c}{\partial Z^i_a}$. So $\frac{\partial c}{\partial B^i} = \begin{bmatrix}\frac{\partial c}{\partial Z^i_1}\\.\\.\\.\\\frac{\partial c}{\partial Z^i_j}\\\end{bmatrix} = \frac{\partial c}{\partial Z^i}$

Finally we need to calculate $\frac{\partial c}{\partial X^i}$ to feed it to the $i-1$ nth layer. We have $\frac{\partial c}{\partial X^i} = \begin{bmatrix}\frac{\partial c}{\partial X^i_1}\\.\\.\\.\\\frac{\partial c}{\partial X^i_k}\\\end{bmatrix}$

Let's look at $\frac{\partial c}{\partial X^i_1}$. We have $\frac{\partial c}{\partial X^i_1} = \frac{\partial c}{\partial Z^i}\frac{\partial Z^i}{\partial X^i_1}$. 
$\frac{\partial Z^i}{\partial X^i_1}$ is equal to $ \begin{bmatrix}w_{1,1}^i\\.\\.\\.\\w_{j,1}^i\\\end{bmatrix}$. So $\frac{\partial c}{\partial X^i_1} = \begin{bmatrix}w_{1,1}^i\\.\\.\\.\\w_{j,1}^i\\\end{bmatrix} \cdot \begin{bmatrix}\frac{\partial c}{\partial Z^i_1}\\.\\.\\.\\\frac{\partial c}{\partial Z^i_j}\\\end{bmatrix} = \begin{bmatrix}w_{1,1}^i\frac{\partial c}{\partial Z^i_1} + ... + w_{j,1}^i\frac{\partial c}{\partial Z^i_j}\\\end{bmatrix}$. 

By generalizing this to $\forall a \in [1,k], \frac{\partial c}{\partial X^i_a} = \begin{bmatrix}w_{1,a}^i\frac{\partial c}{\partial Z^i_1} + ... + w_{j,a}^i\frac{\partial c}{\partial Z^i_j}\\\end{bmatrix}$ we get $\frac{\partial c}{\partial X^i} = \begin{bmatrix}w_{1,1}^i\frac{\partial c}{\partial Z^i_1} + ... + w_{j,1}^i\frac{\partial c}{\partial Z^i_j}\\.\\.\\.\\w_{1,k}^i\frac{\partial c}{\partial Z^i_1} + ... + w_{j,k}^i\frac{\partial c}{\partial Z^i_j}\\\end{bmatrix} = \begin{bmatrix}w_{1,1}^i&...&w_{j,1}^i\\\vdots&\ddots&\vdots\\w_{1,k}^i&...&w_{j,k}^i\\\end{bmatrix} \cdot \begin{bmatrix}\frac{\partial c}{\partial Z^i_1}\\.\\.\\.\\\frac{\partial c}{\partial Z^i_j}\\\end{bmatrix} = {W^i}^\intercal \cdot \frac{\partial c}{\partial Z^i}$

We now have all the gradients we need except for $\frac{\partial c}{\partial Z^i}$. We have $\frac{\partial c}{\partial Z^i} = \begin{bmatrix}\frac{\partial c}{\partial Z^i_1}\\.\\.\\.\\\frac{\partial c}{\partial Z^i_j}\\\end{bmatrix}$. 

Let's look at $\frac{\partial c}{\partial Z^i_1}$.

$\frac{\partial c}{\partial Z^i_1} = \frac{\partial c}{\partial A^i_1}\frac{\partial A^i_1}{\partial Z^i_1} + ... + \frac{\partial c}{\partial A^i_j}\frac{\partial A^i_j}{\partial Z^i_1} $. $\forall a \in [1,j] \setminus \{1\}, \frac{\partial A^i_a}{\partial Z^i_1} = 0$ and $\frac{\partial A^i_1}{\partial Z^i_1} = \sigma'(Z^i_i)$.

So $\frac{\partial c}{\partial Z^i_1} = \frac{\partial c}{\partial A^i_1}\sigma'(Z^i_i)$. By generalizing this to $\forall a \in [1,j], \frac{\partial c}{\partial Z^i_a} = \frac{\partial c}{\partial A^i_a}\sigma'(Z^i_a)$ we get $\frac{\partial c}{\partial Z^i} = \begin{bmatrix}\frac{\partial c}{\partial A^i_1}\sigma'(Z^i_1)\\.\\.\\.\\\frac{\partial c}{\partial A^i_j}\sigma'(Z^i_j)\\\end{bmatrix} = \begin{bmatrix}\frac{\partial c}{\partial A^i_1}\\.\\.\\.\\\frac{\partial c}{\partial A^i_j}\\\end{bmatrix} \odot \begin{bmatrix}\sigma'(Z^i_1)\\.\\.\\.\\\sigma'(Z^i_j)\\\end{bmatrix} = \frac{\partial c}{\partial A^i} \odot \sigma'(Z^i)$.

Now we can examine the code :

Suppose we have $l$ layer in our network. We first calculate $\frac{\partial c}{\partial A^l}$ :
```rust
let mut error = cross_entropy_prime(&label_array, &pred);
```

Then we iterate over each layer in reverse order:
```rust
for layer in model.layers.iter_mut().rev() {
    error = layer.backward(error, LEARNING_RATE);
}
```

The `backward` function returns $\frac{\partial c}{\partial X^i}$ and updates the weights and biases of the layer. Let's look at the code for the `backward` function of the `Layer` struct :

We first calculate $\frac{\partial c}{\partial Z^i} = \sigma'(Z^i) \odot \frac{\partial c}{\partial A^i}$ :
```rust
let output_error = self.activation.apply_der(&self.output) * output_error;
```

Then we calculate $\frac{\partial c}{\partial X^i} = {W^i}^\intercal \cdot \frac{\partial c}{\partial Z^i}$ :
```rust
let inp_err = self.weights.t().dot(&output_error);
```

Finally we calculate $\frac{\partial c}{\partial W^i} = \frac{\partial c}{\partial Z^i}{X^i}^\intercal$ and $\frac{\partial c}{\partial B^i} = \frac{\partial c}{\partial Z^i}$ ,update the weights and biases and return $\frac{\partial c}{\partial X^i}$ :
```rust
let weight_err = output_error.dot(&self.input.t());
self.weights = &self.weights - learning_rate * &weight_err;
self.bias = &self.bias - learning_rate * &output_error;
inp_err
```
