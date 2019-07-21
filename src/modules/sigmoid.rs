pub use crate::modules;

#[derive(Debug)]
pub struct Sigmoid{}

impl Sigmoid {
    pub fn new() -> Sigmoid{
        Sigmoid{}
    }
}

impl modules::Module for Sigmoid {
    fn forward(&self) {
        println!("forward for Sigmoid");
    }

    fn backward(&self) {
        println!("backward for Sigmoid");
    }
}
