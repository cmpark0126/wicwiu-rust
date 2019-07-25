pub trait Module {
    fn forward(&self) -> &Module;
    fn backward(&self) -> &Module;
}
