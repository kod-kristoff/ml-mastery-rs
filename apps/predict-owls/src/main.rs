use csv::{ReaderBuilder, WriterBuilder};
use ndarray::{Array1, Array2};
use ndarray_csv::{Array2Reader, Array2Writer};
use neural_network::Network;
use rand::RngCore;
use std::error::Error;
use std::fs::File;

fn main() {
    let mut rng = rand::thread_rng();
    owls_and_albatrosses(&mut rng);
    match owls_and_albatrosses_from_file(&mut rng) {
        Err(err) => {
            println!("Error: {:?}", err);
            ::std::process::exit(2);
        }
        _ => {}
    }
    condors_and_albatrosses(&mut rng);
    match condors_and_albatrosses_from_file(&mut rng) {
        Err(err) => {
            println!("Error: {:?}", err);
            ::std::process::exit(2);
        }
        _ => {}
    }
}

fn owls_and_albatrosses(rng: &mut dyn RngCore) {
    println!(">>> owls_and_albatrosses");

    let (x, y) = generate_albatrosses_owls(rng);

    let now = std::time::Instant::now();
    evaluate_perceptron(&x, &y, rng);
    println!("Perceptron, elapsed time: {:.6?}", now.elapsed());

    let now = std::time::Instant::now();
    evaluate_adaline(&x, &y, rng);
    println!("Adaline, elapsed time: {:.6?}", now.elapsed());

    println!("<<< owls_and_albatrosses");
}

fn owls_and_albatrosses_from_file(rng: &mut dyn RngCore) -> Result<(), Box<dyn Error>> {
    println!(">>> owls_and_albatrosses_from_file");

    let (x, y) = read_albatrosses_owls()?;

    let now = std::time::Instant::now();
    evaluate_perceptron(&x, &y, rng);
    println!("Perceptron, elapsed time: {:.6?}", now.elapsed());

    let now = std::time::Instant::now();
    evaluate_adaline(&x, &y, rng);
    println!("Adaline, elapsed time: {:.6?}", now.elapsed());

    println!("<<< owls_and_albatrosses_from_file");
    Ok(())
}

fn condors_and_albatrosses(rng: &mut dyn RngCore) {
    println!(">>> condors_and_albatrosses");

    let (x, y) = generate_albatrosses_condorss(rng);

    let now = std::time::Instant::now();
    evaluate_perceptron(&x, &y, rng);
    println!("Perceptron, elapsed time: {:.6?}", now.elapsed());

    let now = std::time::Instant::now();
    evaluate_adaline(&x, &y, rng);
    println!("Adaline, elapsed time: {:.6?}", now.elapsed());

    println!("<<< condors_and_albatrosses");
}

fn condors_and_albatrosses_from_file(rng: &mut dyn RngCore) -> Result<(), Box<dyn Error>> {
    println!(">>> condors_and_albatrosses_from_file");

    let (x, y) = read_albatrosses_condorss()?;

    let now = std::time::Instant::now();
    evaluate_perceptron(&x, &y, rng);
    println!("Perceptron, elapsed time: {:.6?}", now.elapsed());

    let now = std::time::Instant::now();
    evaluate_adaline(&x, &y, rng);
    println!("Adaline, elapsed time: {:.6?}", now.elapsed());

    println!("<<< condors_and_albatrosses_from_file");
    Ok(())
}

fn generate_albatrosses_owls(rng: &mut dyn RngCore) -> (Array2<f64>, Array1<i32>) {
    let mut x = Array2::<f64>::zeros((200, 2));
    let mut y = Array1::<i32>::zeros(200);
    let (a_x, a_y) = species_generator(9000.0, 800.0, 300.0, 20.0, 100, 1, rng);
    let (o_x, o_y) = species_generator(1000.0, 200.0, 100.0, 15.0, 100, -1, rng);
    x.slice_mut(ndarray::s![0..100, ..]).assign(&a_x);
    x.slice_mut(ndarray::s![100.., ..]).assign(&o_x);
    y.slice_mut(ndarray::s![0..100]).assign(&a_y);
    y.slice_mut(ndarray::s![100..]).assign(&o_y);
    // let x = ndarray::stack_new_axis![ndarray::Axis(0), a_x, o_x];

    let shuffle = rand::seq::index::sample(rng, 200, 200);

    let mut x_shuffled = Array2::<f64>::zeros((200, 2));
    let mut y_shuffled = Array1::<i32>::zeros(200);

    for (j, i) in shuffle.iter().zip(0..200) {
        x_shuffled.row_mut(i).assign(&x.row(j));
        y_shuffled[i] = y[j];
    }
    (x_shuffled, y_shuffled)
}

fn read_albatrosses_owls() -> Result<(Array2<f64>, Array1<i32>), Box<dyn Error>> {
    let file = File::open("albaowl_data.csv")?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let x: Array2<f64> = reader.deserialize_array2((200, 2))?;
    let file = File::open("albaowl_species.csv")?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let y: Array2<i32> = reader.deserialize_array2((200, 1))?;
    let y = y.column(0).to_owned();
    Ok((x, y))
}

fn generate_albatrosses_condorss(rng: &mut dyn RngCore) -> (Array2<f64>, Array1<i32>) {
    let mut x = Array2::<f64>::zeros((200, 2));
    let mut y = Array1::<i32>::zeros(200);
    let (a_x, a_y) = species_generator(9000.0, 800.0, 300.0, 20.0, 100, 1, rng);
    let (o_x, o_y) = species_generator(12000.0, 1000.0, 290.0, 15.0, 100, -1, rng);
    x.slice_mut(ndarray::s![0..100, ..]).assign(&a_x);
    x.slice_mut(ndarray::s![100.., ..]).assign(&o_x);
    y.slice_mut(ndarray::s![0..100]).assign(&a_y);
    y.slice_mut(ndarray::s![100..]).assign(&o_y);
    // let x = ndarray::stack_new_axis![ndarray::Axis(0), a_x, o_x];
    let shuffle = rand::seq::index::sample(rng, 200, 200);

    let mut x_shuffled = Array2::<f64>::zeros((200, 2));
    let mut y_shuffled = Array1::<i32>::zeros(200);

    for (j, i) in shuffle.iter().zip(0..200) {
        x_shuffled.row_mut(i).assign(&x.row(j));
        y_shuffled[i] = y[j];
    }
    (x_shuffled, y_shuffled)
}

fn read_albatrosses_condorss() -> Result<(Array2<f64>, Array1<i32>), Box<dyn Error>> {
    let file = File::open("albacondor_data.csv")?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let x: Array2<f64> = reader.deserialize_array2((200, 2))?;
    let file = File::open("albacondor_species.csv")?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let y: Array2<i32> = reader.deserialize_array2((200, 1))?;
    let y = y.column(0).to_owned();
    Ok((x, y))
}

fn species_generator(
    mu1: f64,
    sigma1: f64,
    mu2: f64,
    sigma2: f64,
    n_samples: usize,
    target: i32,
    rng: &mut dyn RngCore,
) -> (Array2<f64>, Array1<i32>) {
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;

    let normal1 = Normal::new(mu1, sigma1).unwrap();
    let normal2 = Normal::new(mu2, sigma2).unwrap();
    let values1 = Array1::<f64>::random_using(n_samples, normal1, rng);
    let values2 = Array1::<f64>::random_using(n_samples, normal2, rng);
    let x = ndarray::stack![ndarray::Axis(1), values1, values2];
    assert_eq!(x.shape(), &[n_samples, 2]);
    let y = Array1::<i32>::from_elem(n_samples, target);
    (x, y)
}

fn evaluate_perceptron(x: &Array2<f64>, y: &Array1<i32>, rng: &mut dyn RngCore) {
    let eta = 0.01;
    let n_iter = 200;
    let network = Network::perceptron_fit(x, y, eta, n_iter, rng);
    let y_pred = network.predict(x);
    let num_correct: i64 = y
        .iter()
        .zip(y_pred.iter())
        .map(|(y, yp)| if y == yp { 1 } else { 0 })
        .sum();
    let accuracy = (num_correct as f64 / y.shape()[0] as f64) * 100.0;
    println!("Perceptron accuracy: {:.4} %", accuracy);
}

fn evaluate_adaline(x: &Array2<f64>, y: &Array1<i32>, rng: &mut dyn RngCore) {
    let eta = 1e-10;
    let n_iter = 120;
    let network = Network::adaline_fit(x, y, eta, n_iter, rng);
    let y_pred = network.predict(x);
    let num_correct: i64 = y
        .iter()
        .zip(y_pred.iter())
        .map(|(y, yp)| if y == yp { 1 } else { 0 })
        .sum();
    let accuracy = (num_correct as f64 / y.shape()[0] as f64) * 100.0;
    println!("Adaline accuracy: {:.4} %", accuracy);
}
