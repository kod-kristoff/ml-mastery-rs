use ndarray::{Array0, Array1, Array2, ArrayView1, ArrayView2};

pub struct Network {
    weights: Array1<f64>,
    bias: f64,
    errors: Array1<i64>,
}

impl Network {
    pub fn fit(x: &Array2<f64>, y: &Array1<i32>, eta: f64, n_iter: usize) -> Self {
        let (mut weights, mut bias) = random_weights(x);
        let mut errors = Array1::<i64>::zeros(n_iter);
        for j in 0..n_iter {
            for i in 0..y.len() {
                let xi = x.slice(ndarray::s![i..(i+1), ..]);
                // println!("predicy(xi) = {:?}", do_predict(&xi.t(), &weights));
                let target = y[i];
                
                let delta = eta * (target - do_predict(&xi, &weights, bias)[0]) as f64;
                // for k in 1..weights.len() {
                 
                // weights[k] += delta * xi[[0, k-1]];
                // }
                weights += &(delta * &xi.row(0));
                bias += delta;
                if !approx::abs_diff_eq!(delta, 0.0) {
                    errors[j] += 1;
                }
            }
        }
        Self { weights, bias, errors }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<i32> {
        do_predict(&x.view(), &self.weights, self.bias)
    }
}

fn do_predict(x: &ArrayView2<f64>, w: &Array1<f64>, bias: f64) -> Array1<i32> {
    net_input(x, &w.view(), bias).map(|x| {
        if *x < 0.0 {
            -1
        } else {
            1
        }
    })
}

fn net_input(x: &ArrayView2<f64>, w: &ArrayView1<f64>, bias: f64) -> Array1<f64> {
    x.dot(w) + bias
}


fn random_weights(x: &Array2<f64>) -> (Array1<f64>, f64) {
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;

    let normal = Normal::new(0., 0.01).unwrap();
    let weights = Array1::<f64>::random(x.shape()[1], normal);
    let bias = Array1::<f64>::random(1, normal)[0];
    (weights, bias)
    // Array1::<f64>::zeros(1 + x.shape()[1])
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
