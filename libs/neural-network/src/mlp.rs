use rand::RngCore;
use ndarray::{Array2};

pub struct MlpNetwork {
    layers: Vec<Layer>,
}

struct Layer {
    bias: Array2<f64>,
    weights: Array2<f64>,
}

impl MlpNetwork {
    pub fn random(layers: &[usize], rng: &mut dyn RngCore) -> Self {
        let layers = layers
            .windows(2)
            .map(|layers| {
                Layer::random(layers[0], layers[1], rng)
            })
            .collect();

        Self { layers }
    }
}

impl Layer {
    pub fn random(input_neurons: usize, output_neurons: usize, rng: &mut dyn RngCore) -> Self {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Uniform;

        let uniform = Uniform::new(-0.1, 0.1);
        let bias = Array2::random_using((1, output_neurons), uniform, rng);
        let weights = Array2::random_using((input_neurons, output_neurons), uniform, rng);
        Self { bias, weights }
    }

    fn propagate(&self, x: &Array2<f64>) -> Array2<f64> {
        (x.dot(&self.weights) + &self.bias).map(|v| 1.0/(1.0 + (-v).exp()))
    } 
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    mod random {
        use super::*;
        use rand_chacha::ChaCha8Rng;
        use rand::SeedableRng;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let network = MlpNetwork::random(&[2, 3, 2], &mut rng);

            assert_eq!(network.layers.len(), 2);
            assert_eq!(network.layers[0].bias.shape(), &[1, 3]);
            assert_eq!(network.layers[0].weights.shape(), &[2, 3]);
            assert_eq!(network.layers[1].bias.shape(), &[1, 2]);
            assert_eq!(network.layers[1].weights.shape(), &[3, 2]);

            assert_relative_eq!(network.layers[0].bias[[0, 0]], 0.06738395137652944);
        }
    }

    mod propagate {
        use super::*;
        use ndarray::arr2;

        #[test]
        fn layer_propagate() {
            let weights = arr2(
                &[[1., 2., 3.],
                  [3., 1., 2.],]
            );
            let bias = arr2(&[
                [1., 2., 3.]
            ]);
            let layer = Layer { weights, bias };

            let x = arr2(&[[1., 2.]]);
            let res = layer.propagate(&x);

            assert_eq!(res.shape(), &[1, 3]);
            assert_relative_eq!(res[[0, 0]], 1./(1. + (-8_f64).exp()));

            let x = arr2(
                &[[1., 2.],
                  [3., 1.]]);
            let res = layer.propagate(&x);

            assert_eq!(res.shape(), &[2, 3]);
        }
    }
}
