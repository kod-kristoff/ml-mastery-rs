use csv::ReaderBuilder;
use ndarray::{s, Array2};
use ndarray_csv::Array2Reader;
use std::error::Error;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn Error>> {
    let file_name = "ionosphere.csv";
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(file_name)?;

    let data: Array2<String> = reader.deserialize_array2_dynamic()?;
    println!("Data loaded!");
    println!("  shape: {:?}", &data.shape());

    let features: Array2<f64> = data
        .slice(s![.., ..34])
        .mapv(|elem| f64::from_str(&elem).unwrap());
    println!("  features-shape: {:?}", &features.shape());
    Ok(())
}
