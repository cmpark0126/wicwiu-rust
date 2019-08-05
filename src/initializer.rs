use crate::tensor::Tensor;
use num::{Num, NumCast, Float, FromPrimitive};
use std::fmt::Debug;
use rand::Rng;

pub fn random_initializer<T: Num + NumCast + Float + Clone + FromPrimitive + Debug>(
    parameter: &mut Tensor<T>,
    in_features: usize,
){
    // let mut parameter = parameter.borrow_mut();
    let len = parameter.shape.capacity();
    let in_features : T = T::from_usize(in_features).unwrap();
    let two : T = T::from_usize(2).unwrap();
    let range = T::one() / T::sqrt(in_features + T::one());
    let mut rng = rand::thread_rng();

    for w in 0..len{
        let generated_val : T = T::from_f32(rng.next_f32()).unwrap();
        parameter.longarray[w] = generated_val * two * range - range;
    }

}
