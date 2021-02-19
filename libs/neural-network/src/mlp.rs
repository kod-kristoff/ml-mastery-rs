use rand::RngCore;

pub struct MlpNetwork {}

impl MlpNetwork {
    pub fn random(layers: &[usize], rng: &mut dyn RngCore) -> Self {
        Self {}
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    mod random {
        use super::*;
        use rand_chacha::ChaCha8Rng;
        use rand::SeedableRng;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let network = MlpNetwork::random(&[3, 2], &mut rng);

        }
    }
}
