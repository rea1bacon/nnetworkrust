use ndarray::Array2;
pub fn mse(ground_truth: &Array2<f64>, prediction: &Array2<f64>) -> f64 {
    assert_eq!(ground_truth.shape(), prediction.shape());
    let n = prediction.len() as f64;
    let sum_squared_diff: f64 = ground_truth
        .iter()
        .zip(prediction.iter())
        .map(|(y_i, y_pred_i)| (*y_i - *y_pred_i) * (*y_i - *y_pred_i))
        .sum();
    sum_squared_diff / n
}

pub fn mse_derivative(ground_truth: &Array2<f64>, prediction: &Array2<f64>) -> Array2<f64> {
    assert_eq!(ground_truth.shape(), prediction.shape());
    let n = prediction.len();
    let factor = 2.0 / n as f64;
    //println!("pred {:?}\n gt {:?}", prediction, ground_truth);
    let res = (prediction - ground_truth) * factor;
    res.to_shape((1, n)).unwrap().to_owned()
}
