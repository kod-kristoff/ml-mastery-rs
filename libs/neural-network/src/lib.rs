use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::RngCore;

pub struct Network {
    weights: Array1<f64>,
    bias: f64,
}

impl Network {
    pub fn perceptron_fit(x: &Array2<f64>, y: &Array1<i32>, eta: f64, n_iter: usize, rng: &mut dyn RngCore) -> Self {
        let (mut weights, mut bias) = random_weights(x, rng);
        for _ in 0..n_iter {
            for i in 0..y.len() {
                let xi = x.slice(ndarray::s![i..(i+1), ..]);
                // println!("predicy(xi) = {:?}", do_predict(&xi.t(), &weights));
                let target = y[i];
                
                let delta = eta * (target - do_predict(&xi, &weights.view(), bias)[0]) as f64;
                // for k in 1..weights.len() {
                 
                // weights[k] += delta * xi[[0, k-1]];
                // }
                weights += &(delta * &xi.row(0));
                bias += delta;
            }
        }
        Self { weights, bias }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<i32> {
        do_predict(&x.view(), &self.weights.view(), self.bias)
    }
}

fn do_predict(x: &ArrayView2<f64>, w: &ArrayView1<f64>, bias: f64) -> Array1<i32> {
    net_input(x, w, bias).map(|x| {
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


fn random_weights(x: &Array2<f64>, rng: &mut dyn RngCore) -> (Array1<f64>, f64) {
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;
    use rand::distributions::Distribution;

    let normal = Normal::new(0., 0.01).unwrap();
    let weights = Array1::<f64>::random_using(x.shape()[1], normal, rng);
    let bias = normal.sample(rng);
    (weights, bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    mod random_weights {
        use super::*;
        use rand_chacha::ChaCha8Rng;
        use rand::SeedableRng;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());

            let x = Array2::<f64>::zeros((4, 5));

            let (weights, bias) = random_weights(&x, &mut rng);

            assert_eq!(weights.shape(), &[x.shape()[1]]);
            assert_relative_eq!(bias, -0.010646159174430792);
            assert_relative_eq!(weights[0], 0.013776972067507914);
        }
    }
}
