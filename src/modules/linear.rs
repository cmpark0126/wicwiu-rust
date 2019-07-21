use crate::modules;

#[derive(Debug)]
pub struct Linear{
    use_bias: bool
}

impl Linear {
    pub fn new(use_bias: bool) -> Linear{
        Linear{use_bias: use_bias}
    }
}

impl modules::Module for Linear {
    fn forward(&self) {
        println!("forward for Linear : use_bias = {}", self.use_bias);
    }

    fn backward(&self) {
        println!("backward for Linear : use_bias = {}", self.use_bias);
    }
}
