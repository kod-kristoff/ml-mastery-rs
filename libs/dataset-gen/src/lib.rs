pub fn make_circles(factor: f64) -> ((), ()) {
    if factor < 0.0 || factor > 1.0 {
        panic!("'factor' has to be between 0 and 1, got {}", factor);
    }
    ((), ())
}
#[cfg(test)]
mod tests {
    use super::*;

    mod make_circles {
        use super::*;

        #[test]
        #[should_panic]
        fn factor_below_0_panics() {
            let factor = -1.0;
            let (x, y) = make_circles(factor);
        }

        #[test]
        #[should_panic]
        fn factor_above_1_panics() {
            let factor = 1.01;
            let (x, y) = make_circles(factor);
        }
    }
}
