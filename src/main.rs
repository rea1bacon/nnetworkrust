use csv::ReaderBuilder;
use ndarray::s;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::Axis;
use noto::loss::*;
use noto::Layer;
use noto::Model;
fn main() {
    let mut model = Model::new();
    model.add_layer(Layer::init(784, 100).with_activation(noto::ActivationFunc::Sigmoid));
    model.add_layer(Layer::init(100, 300).with_activation(noto::ActivationFunc::Relu));
    model.add_layer(Layer::init(300, 100).with_activation(noto::ActivationFunc::Sigmoid));
    model.add_layer(Layer::init(100, 10).with_activation(noto::ActivationFunc::Softmax));
    let data = load_dataset();
    const LEARNING_RATE: f64 = 0.05;
    // iterate over batchs of 100 rows
    for batch in data.axis_chunks_iter(Axis(0), 300) {
        let mut e = 0.;
        // compute predictions and prepare label
        for row in batch.rows() {
            //println!("{:?}", row);
            let label = row[0] as usize;
            let mut label_array = Array2::zeros((10, 1));
            label_array[[label, 0]] = 1.;
            let input = row.slice(s![1..]).insert_axis(Axis(0)).map(|e| *e / 255.);
            let pred = model.forward(&input);
            //println!("{:?} {:?}", label_array, pred);
            let cost = cross_entropy(&label_array, &pred);
            e += cost;
            let mut error = cross_entropy_prime(&label_array, &pred);
            for layer in model.layers.iter_mut().rev() {
                error = layer.backward(error, LEARNING_RATE);
            }
        }
        println!("avg loss : {}", e / 300.);
    }

    // now test the model on "mnist_test.csv"
    let file_test = std::fs::File::open("datasets/mnist_test.csv").unwrap();
    let mut rdr_test = ReaderBuilder::new().flexible(true).from_reader(file_test);
    let mut num_rows = 0;
    let mut num_correct = 0;
    for result in rdr_test.records() {
        num_rows += 1;
        let record = result.unwrap();
        let label = record[0].parse::<usize>().unwrap();
        let mut label_array = Array2::zeros((10, 1));
        label_array[[label, 0]] = 1.;
        let input = record
            .iter()
            .skip(1)
            .map(|e| e.parse::<f64>().unwrap() / 255.)
            .collect::<Vec<_>>();
        let input = Array2::from_shape_vec((784, 1), input).unwrap();
        let pred = model.forward(&input);
        let pred = pred
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        if pred == label {
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
