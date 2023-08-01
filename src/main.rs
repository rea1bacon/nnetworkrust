use csv::ReaderBuilder;
use ndarray::s;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::Axis;
use noto::loss::mse;
use noto::loss::mse_derivative;
use noto::Layer;
use noto::Model;
fn main() {
    let mut model = Model::new();
    model.add_layer(Layer::init(784, 50).with_activation(noto::ActivationFunc::Sigmoid));
    model.add_layer(Layer::init(50, 100).with_activation(noto::ActivationFunc::Sigmoid));
    model.add_layer(Layer::init(100, 60).with_activation(noto::ActivationFunc::Tanh));
    model.add_layer(Layer::init(60, 40).with_activation(noto::ActivationFunc::Sigmoid));
    model.add_layer(Layer::init(40, 10).with_activation(noto::ActivationFunc::Softmax));
    let data = load_dataset();
    const LEARNING_RATE: f64 = 0.2;
    // iterate over batchs of 100 rows
    for batch in data.axis_chunks_iter(Axis(0), 100) {
        let mut e = 0.;
        // compute predictions and prepare label
        for row in batch.rows() {
            //println!("{:?}", row);
            let label = row[0] as usize;
            let mut label_array = Array2::zeros((1, 10));
            label_array[[0, label]] = 1.;
            let input = row.slice(s![1..]).insert_axis(Axis(0)).map(|e| *e / 255.);
            let pred = model.forward(&input);
            //println!("{:?} {:?}", label_array, pred);
            let cost = mse(&label_array, &pred);
            e += cost;
            let mut error = mse_derivative(&label_array, &pred);
            for layer in model.layers.iter_mut().rev() {
                error = layer.backward(error, LEARNING_RATE);
            }
        }
        println!("avg loss : {}", e / 100.);
    }

    // now test on the file "mnist_test.csv"
    let file_path = "datasets/mnist_test.csv";

    // Open the CSV file
    let file = std::fs::File::open(file_path).unwrap();

    // Create a CSV reader with a flexible reader that can handle varying CSV formats
    let mut rdr = ReaderBuilder::new().flexible(true).from_reader(file);
    let mut num_rows = 0;
    // Read the CSV records as Array2<f64>
    let records: Vec<f64> = rdr
        .records()
        .flat_map(|result| {
            num_rows += 1;
            result
                .unwrap()
                .iter()
                .map(|field| field.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    let records = Array2::from_shape_vec((num_rows, 785), records).unwrap();
    let mut num_correct = 0;
    for row in records.rows() {
        let label = row[0] as usize;
        let input = row.slice(s![1..]).insert_axis(Axis(0)).map(|e| *e / 255.);
        let pred = model.forward(&input);
        let pred = pred.into_shape((10,)).unwrap();
        let mut max = 0.;
        let mut max_index = 0;
        for (i, e) in pred.iter().enumerate() {
            if *e > max {
                max = *e;
                max_index = i;
            }
        }
        if max_index == label {
            num_correct += 1;
        }
    }
    println!("accuracy : {}", num_correct as f64 / num_rows as f64);
}

fn load_dataset() -> Array2<f64> {
    let file_path = "datasets/mnist_train.csv";

    // Open the CSV file
    let file = std::fs::File::open(file_path).unwrap();

    // Create a CSV reader with a flexible reader that can handle varying CSV formats
    let mut rdr = ReaderBuilder::new().flexible(true).from_reader(file);
    let mut num_rows = 0;
    // Read the CSV records as Vec<f64>
    let records: Vec<f64> = rdr
        .records()
        .flat_map(|result| {
            num_rows += 1;
            result
                .unwrap()
                .iter()
                .map(|field| field.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect();

    // Get the number of rows and columns

    let num_columns = rdr.headers().unwrap().len();

    // Create the 2D matrix (ndarray::Array2) from the records
    //println!("{} {num_rows}, {num_columns}", records.len());
    let matrix: Array2<f64> = ArrayBase::from_shape_vec((num_rows, num_columns), records).unwrap();
    matrix
    /*

    // create dataset : a matrice of 1000 rows, each has 3 columns : a || b, a, b with a and b are 0 or 1
    let rows = 1000000;
    let mut matrix = Array2::zeros((rows, 3));

    let mut rng = rand::thread_rng();
    for i in 0..rows {
        let a = rng.gen_range(0..=1);
        let b = rng.gen_range(0..=1);
        matrix[[i, 0]] = (a | b) as f64;
        matrix[[i, 1]] = a as f64;
        matrix[[i, 2]] = b as f64;
    }
    matrix*/
}
