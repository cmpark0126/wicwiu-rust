use crate::modules;

#[derive(Debug)]
pub struct MSE{}

impl MSE {
    pub fn new() -> MSE{
        MSE{}
    }
}

impl modules::Module for MSE {
    fn forward(&self) {
        println!("forward for MSE");
    }

    fn backward(&self) {
        println!("backward for MSE");
    }
}
