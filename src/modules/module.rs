pub trait Module {
    fn forward(&self);
    fn backward(&self);
}
