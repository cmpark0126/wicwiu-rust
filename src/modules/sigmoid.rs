pub use crate::modules::Module;

#[derive(Debug)]
pub struct Sigmoid{}

impl Sigmoid {
    pub fn new() -> Sigmoid{
        Sigmoid{}
    }
}

impl Module for Sigmoid {
    fn forward(&self) -> &Module{
        println!("forward for Sigmoid");
        self
    }

    fn backward(&self) -> &Module{
        println!("backward for Sigmoid");
        self
    }
}
