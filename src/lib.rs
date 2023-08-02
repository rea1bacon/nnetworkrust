use ndarray::prelude::*;
use ndarray::Dim;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
pub mod loss;

#[cfg(test)]
mod test;
/// A neural network model consisting of multiple layers.
pub struct Model {
    pub layers: Vec<Layer>,
}

impl Model {
    pub fn add_layer(&mut self, layer: Layer) {
        if let Some(l) = self.layers.last() {
            assert_eq!(
                l.shape().1,
                layer.shape().0,
                "Shapes between layers don't match : \n
                {}-nth layer : {}\n
                Pushed layer : {}
                ",
                self.layers.len(),
                l,
                layer
            )
        }
        self.layers.push(layer)
    }

    pub fn new() -> Self {
        Model { layers: Vec::new() }
    }

    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let h = input;
        let mut h = self.layers[0].forward(h);
        for layer in self.layers.iter_mut().skip(1) {
            h = layer.forward(&h);
        }
        h
    }
}

/// A single layer in the neural network.
#[derive(Debug)]
pub struct Layer {
    pub weights: Array2<f64>,
    bias: Array2<f64>,
    activation: ActivationFunc,
    has_bias: bool,
    grad_computed: bool,
    input: Array2<f64>,
    output: Array2<f64>,
    pub(crate) output_act: Array2<f64>,
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.shape();
        write!(
            f,
            "[Layer, inputs = {}, neurons = {}, has_bias = {}, activation_function = {:?}]",
            s.0, s.1, self.has_bias, self.activation
        )
    }
}

impl Layer {
    /// Initializes a new layer with random weights and biases.
    ///
    /// # Arguments
    ///
    /// * `inps` - Number of input.
    /// * `neur` - Number of neurons in the layer.
    ///
    /// # Returns
    ///
    /// A new `Layer` instance with random weights and biases.
    pub fn init(inps: usize, neur: usize) -> Self {
        let w = Array::random((neur, inps), Uniform::new(-0.5, 0.5));
        let b = Array::random((neur, 1), Uniform::new(-0.5, 0.5));
        Layer {
            weights: w,
            bias: b,
            activation: ActivationFunc::Sigmoid,
            has_bias: true,
            grad_computed: false,
            input: Array2::zeros((1, 1)),
            output: Array2::zeros((1, 1)),
            output_act: Array2::zeros((1, 1)),
        }
    }

    /// Removes the bias from the layer, setting it to zero.
    ///
    /// # Returns
    ///
    /// The modified `Layer` instance with zero biases.
    pub fn without_bias(mut self) -> Self {
        self.bias = Array2::zeros((self.shape().0, 1));
        self.has_bias = false;
        self
    }

    pub fn backward(&mut self, output_error: Array2<f64>, learning_rate: f64) -> Array2<f64> {
        let output_error = self.activation.apply_der(&self.output) * output_error;
        self.grad_computed = true;
        let inp_err = self.weights.t().dot(&output_error);
        let weight_err = output_error.dot(&self.input.t());
        self.weights = &self.weights - learning_rate * &weight_err;
        self.bias = &self.bias - learning_rate * &output_error;
        inp_err
    }

    /// Sets the activation function for the layer.
    ///
    /// # Arguments
    ///
    /// * `activation` - The activation function to use.
    ///
    /// # Returns
    ///
    /// The modified `Layer` instance with the specified activation function.
    pub fn with_activation(mut self, activation: ActivationFunc) -> Self {
        self.activation = activation;
        self
    }

    /// Forward propagates the input through the layer to produce an output.
    ///
    /// # Arguments
    ///
    /// * `inp` - The input array to be propagated through the layer.
    ///
    /// # Returns
    ///
    /// An array representing the output of the layer after applying the activation function.
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use noto::Layer;
    ///
    /// let layer = Layer::init(2, 3); // Create a layer with 2 input and 3 neurons
    /// let input_data = array![0.1, 0.2]; // Input data with shape (2,)
    /// let output_data = layer.forward(&input_data); // Compute the output of the layer
    /// println!("{:?}", output_data); // Print the output data
    /// ```
    pub fn forward(
        &mut self,
        inp: &Array2<f64>,
    ) -> ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> {
        self.grad_computed = false;
        // reshape input
        let inp = inp.clone().into_shape((inp.len(), 1)).unwrap();
        let a = self.weights.dot(&inp);

        let a = a + &self.bias;
        self.output = a.clone();
        self.input = inp;
        let a = self.activation.apply(&a);
        self.output_act = a.clone();
        a
    }

    /// Returns the shape of the input and the output of the layer.
    ///
    /// # Returns
    ///
    /// A tuple representing the shape of the input and the output like so (input, output).
    fn shape(&self) -> (usize, usize) {
        let s = self.weights.dim();
        (s.1, s.0)
    }

    /// Creates a new `Layer` instance from provided weights and biases.
    ///
    /// # Arguments
    ///
    /// * `weights` - 2D vector representing the weights of the layer.
    /// * `bias` - 1D vector representing the biases of the layer.
    ///
    /// # Returns
    ///
    /// A new `Layer` instance with the given weights and biases.
    ///
    /// # Example
    /// ```
    /// use noto::Layer;
    ///
    /// let weights = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
    /// let bias = vec![0.5, 0.6];
    /// let layer = Layer::from_vec(weights, bias);
    /// ```
    pub fn from_vec(weights: Vec<Vec<f64>>, bias: Vec<f64>) -> Self {
        let mut w = Vec::new();

        let wcols = weights.first().map_or(0, |row| row.len());
        let mut wrows = 0;
        for r in weights.iter() {
            w.extend_from_slice(r.as_slice());
            wrows += 1;
        }
        let weights = Array2::from_shape_vec((wrows, wcols), w).unwrap();
        let bias = Array2::from_shape_vec((wrows, 1), bias).unwrap();
        Layer {
            weights,
            bias,
            activation: ActivationFunc::Sigmoid,
            has_bias: true,
            grad_computed: false,
            input: Array2::zeros((1, 1)),
            output: Array2::zeros((1, 1)),
            output_act: Array2::zeros((1, 1)),
        }
    }
}

/// Enum representing different activation functions for the layer.
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunc {
    Sigmoid,
    Relu,
    Softmax,
    Tanh,
}

impl ActivationFunc {
    /// Applies the activation function to the given input value.
    ///
    /// # Arguments
    ///
    /// * `x` - The input value to apply the activation function to.
    ///
    /// # Returns
    ///
    /// The output value after applying the activation function.
    fn apply(self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Sigmoid => sigmoid(x),
            Self::Relu => relu(x),
            Self::Softmax => softmax(x),
            Self::Tanh => tanh(x),
        }
    }

    fn apply_der(self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Relu => relu_prime(x),
            Self::Sigmoid => sigmoid_prime(x),
            Self::Softmax => softmax_prime(x),
            Self::Tanh => tanh_prime(x),
        }
    }
}

/// Computes the sigmoid function for the given input value.
fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|e| 1. / (1. + (-e).exp()))
}

fn sigmoid_prime(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|e| {
        let t = 1. / (1. + (-e).exp());
        t * (1. - t)
    })
}
fn relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|e| e.max(0.))
}

fn softmax(z: &Array2<f64>) -> Array2<f64> {
    let sum = z.fold(0., |a, e| a + f64::exp(*e));
    z.mapv(|e| e.exp() / sum)
}

fn relu_prime(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|e| if e > 0. { 1. } else { 0. })
}

fn softmax_prime(x: &Array2<f64>) -> Array2<f64> {
    let pc = softmax(x);
    pc.map(|e| e * (1. - e))
}

fn tanh(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|e| e.tanh())
}

fn tanh_prime(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|e| 1. - e.tanh() * e.tanh())
}
