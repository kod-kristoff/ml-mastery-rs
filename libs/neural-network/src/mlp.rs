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

    pub fn fit(x: &Array2::<f64>, y: &Array2::<f64>, topology: &[usize], iterations: usize, eta: f64, rng: &mut dyn RngCore) -> Self {
        let mut network = Self::random(topology, rng);

        for _ in 0..iterations {
            // Forward-propagation
            let s1 = network.layers[0].propagate(x);
            let s2 = network.layers[1].propagate(&s1);

            // Back-propagation
            let delta2 = (&s2 - y)*&s2*(1. - &s2);
            let w2_gradients = s1.t().dot(&delta2);
            network.layers[1].weights -= &(w2_gradients * eta);
            network.layers[1].bias -= delta2.sum() * eta;

            let delta1 = delta2.dot(&network.layers[1].weights.t()) * &s1 * (1. - &s1);
            let w1_gradients = x.t().dot(&delta1);
            network.layers[0].weights -= &(w1_gradients * eta);
            network.layers[0].bias -= delta1.sum() * eta;
        }

        network
    }

    pub fn propagate(&self, x: &Array2<f64>) -> Array2<f64> {
        let x = x.clone();
        self.layers
            .iter()
            .fold(x, |x, layer| layer.propagate(&x)) 
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
    use ndarray::arr2;

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

        #[test]
        fn fit() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let x = arr2(
                &[[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]]
            );
            let y = arr2(
                &[[0.],
                  [1.],
                  [1.],
                  [0.]]
            );
            let iterations = 10;
            let eta = 0.01;
            let network = MlpNetwork::fit(&x, &y, &[2, 3, 1], iterations, eta, &mut rng);

            assert_eq!(network.layers.len(), 2);

            assert_relative_eq!(
                &network.layers[0].weights,
                &arr2(
                    &[[-0.07648684679919386, -0.048877017544423015, -0.08020791605296769],
 [-0.09868510742094942, -0.0476596245401859, -0.036130561247895]])
            );
            assert_relative_eq!(
                &network.layers[1].weights,
                &arr2(
                    &[[0.03266122626760181],
                      [-0.0165584307986786],
                      [0.018798716274266193]]
                )
            );
        }
    }

    mod propagate {
        use super::*;

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
            assert_relative_eq!(res[[0, 0]], 1./(1. + (-8_f64).exp()));
            assert_relative_eq!(res[[1, 0]], 1./(1. + (-7_f64).exp()));
        }

        #[test]
        fn network_propagate() {
            let weights1 = arr2(
                &[[1., 2., 3.],
                  [3., 1., 2.],]
            );
            let bias1 = arr2(&[
                [1., 2., 3.]
            ]);
            let weights2 = arr2(
                &[[1., 2.],
                  [3., 1.],
                  [3., 2.]]
            );
            let bias2 = arr2(&[
                [1., 2.]
            ]);
            let layers = vec![
                Layer { weights: weights1, bias: bias1 },
                Layer { weights: weights2, bias: bias2 },
            ];

            let network = MlpNetwork { layers };
            let x = arr2(
                &[[1., 2.],
                  [3., 1.]]);
            let res = network.propagate(&x);

            assert_eq!(res.shape(), &[2, 2]);
            assert_relative_eq!(
                res, 
                network.layers[1].propagate(
                    &network.layers[0].propagate(&x)
                )
            );
        }
    }
}
