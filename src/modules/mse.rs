use crate::modules::Module;

#[derive(Debug)]
pub struct MSE{}

impl MSE {
    pub fn new() -> MSE{
        MSE{}
    }
}

impl Module for MSE {
    fn forward(&self) -> &Module{
        println!("forward for MSE");
        self
    }

    fn backward(&self) -> &Module{
        println!("backward for MSE");
        self
    }
}
