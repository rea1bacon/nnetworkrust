use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::*;
#[test]
fn init_layer() {
    let _l = Layer::init(123, 24);
}

#[test]
fn layer_from_vec_ok() {
    let weights = vec![vec![1., 3., 4.], vec![-3., 8., 10.], vec![9., -6., 8.]];
    let bias = vec![-3., 6., -1.];
    let _layer = Layer::from_vec(weights, bias);
}

#[test]
#[should_panic]
fn layer_from_vec_panic() {
    let weights = vec![vec![1., 3., 4.], vec![-3., 8., 10.], vec![9., -6.]];
    let bias = vec![-3., 6., -1.];
    let _layer = Layer::from_vec(weights, bias);
}

#[test]
#[should_panic]
fn layer_from_vec_panic_2() {
    let weights = vec![vec![1., 3., 4.], vec![-3., 8., 10.], vec![9., -6., 8.]];
    let bias = vec![-3., 6.];
    let _layer = Layer::from_vec(weights, bias);
}

#[test]
#[should_panic]
fn layer_from_vec_panic_3() {
    let weights = vec![vec![1., 3., 4.], vec![-3., 8., 10.]];
    let bias = vec![-3., 6., 5.];
    let _layer = Layer::from_vec(weights, bias);
}

#[test]
fn test_shape_dot() {
    let mut layer = Layer::init(123, 54);
    let sh = layer.shape();
    let inp = Array2::random((sh.0, 1), Uniform::new(-5., 5.));
    let out = layer.forward(&inp);
    assert!(sh.1 == out.len())
}
