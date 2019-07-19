extern crate ndarray;

use ndarray::*;

fn main() {
    let mut a = Array2::<f64>::zeros((3, 4));
    a += 0.5;
    a[[0, 0]] += 0.5;
    print!("{:?}", a);
}
