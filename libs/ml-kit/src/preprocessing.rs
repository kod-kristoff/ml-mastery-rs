use std::collections::HashMap;

pub struct LabelEncoder {
    map: HashMap<usize, usize>,
}

impl LabelEncoder {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn fit(&mut self, data: &[usize]) {
        for d in data {
            if !self.map.contains_key(d) {
                self.map.insert(*d, self.map.len());
            }
        }
    }

    pub fn classes(&self) -> Vec<usize> {
        self.map.keys().copied().collect()
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    mod label_encoder {
        use super::*;

        #[test]
        fn fit_creates_classes() {
            let mut label_encoder = LabelEncoder::new();

            label_encoder.fit(&[1, 2, 2, 6]);

            assert_eq!(label_encoder.classes(), &[2, 6, 1]);
        }
    }
}
