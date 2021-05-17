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

    mod label_encoder {
        use super::*;

        #[test]
        fn fit_creates_classes() {
            let mut label_encoder = LabelEncoder::new();

            label_encoder.fit(&[1, 2, 2, 6]);

            let classes = label_encoder.classes();
            assert_eq!(classes.len(), 3);
            assert!(classes.contains(&1));
            assert!(classes.contains(&2));
            assert!(classes.contains(&6));
        }
    }
}
