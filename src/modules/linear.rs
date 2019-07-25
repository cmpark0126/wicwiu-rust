use crate::modules::Module;

#[derive(Debug)]
pub struct Linear{
    use_bias: bool
}

impl Linear {
    pub fn new(use_bias: bool) -> Linear{
        Linear{use_bias: use_bias}
    }
}

impl Module for Linear {
    fn forward(&self) -> &Module{
        println!("forward for Linear : use_bias = {}", self.use_bias);
        self
    }

    fn backward(&self) -> &Module{
        println!("backward for Linear : use_bias = {}", self.use_bias);
        self
    }
}
