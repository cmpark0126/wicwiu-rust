// use crate::modules::Module;
// use ndarray::*;
//
// pub struct Linear{
//     pub result: Array1<f64>,
//     pub weight: Array2<f64>,
//     pub bias: Option<Array1<f64>>,
//     inputs: Vec<Box<dyn Module>>,
//     in_features: usize,
//     out_features: usize,
//     use_bias: bool
//
// }
//
// impl Linear{
//     pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Linear{
//         let result = Array1::<f64>::zeros(out_features);
//         let weight = Array2::<f64>::zeros((out_features, in_features));
//         let mut bias = None;
//         if use_bias {
//             bias = Some(Array1::<f64>::zeros(out_features));
//         }
//         Linear{
//             result: result,
//             inputs: Vec::new(),
//             weight: weight,
//             bias: bias,
//             in_features: in_features,
//             out_features: out_features,
//             use_bias: use_bias}
//     }
// }
//
// impl Module for Linear {
//     fn forward(&self) -> &Module{
//         println!("forward for Linear : use_bias = {}", self.use_bias);
//         self
//     }
//
//     fn backward(&self) -> &Module{
//         println!("backward for Linear : use_bias = {}", self.use_bias);
//         self
//     }
// }
