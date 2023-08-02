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

pub fn mse_prime(ground_truth: &Array2<f64>, prediction: &Array2<f64>) -> Array2<f64> {
    assert_eq!(ground_truth.shape(), prediction.shape());
    let n = prediction.len();
    let factor = 2.0 / n as f64;
    //println!("pred {:?}\n gt {:?}", prediction, ground_truth);
    let res = (prediction - ground_truth) * factor;
    res
}

pub fn cross_entropy(ground_truth: &Array2<f64>, prediction: &Array2<f64>) -> f64 {
    assert_eq!(ground_truth.shape(), prediction.shape());
    let sum: f64 = -ground_truth
        .iter()
        .zip(prediction.iter())
        .map(|(y_i, y_pred_i)| {
            if *y_pred_i == 0. {
                1e-15
            } else {
                *y_i * y_pred_i.ln()
            }
        })
        .sum::<f64>();
    sum
}

pub fn cross_entropy_prime(ground_truth: &Array2<f64>, prediction: &Array2<f64>) -> Array2<f64> {
    assert_eq!(ground_truth.shape(), prediction.shape());
    let res: Vec<f64> = prediction
        .iter()
        .zip(ground_truth)
        .map(|(y_pred, y_true)| {
            let mut y_pred = *y_pred;
            if y_pred == 0. {
                y_pred = 1e-15;
            }
            if y_pred == 1. {
                y_pred = 0.9999999999999999;
            }
            -y_true / y_pred + (1. - y_true) / (1. - y_pred)
        })
        .collect();
    let res = Array2::from_shape_vec(prediction.dim(), res).unwrap();
    res
}
