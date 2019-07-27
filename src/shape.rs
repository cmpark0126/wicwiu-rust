#[derive(Debug, Clone)]
pub struct Shape{
    pub rank: usize,
    // if dim == None, Tensor which has this shape represents scalar type
    pub dim: Vec<usize>,
}

impl Shape{
    pub fn new(dim: Vec<usize>) -> Shape{
        Shape {rank: dim.len(), dim: dim}
    }
}
