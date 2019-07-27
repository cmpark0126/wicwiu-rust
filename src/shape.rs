#[derive(Debug, Clone)]
pub struct Shape{
    pub rank: usize,
    // if dim == None, Tensor which has this shape represents scalar type
    pub dim: Vec<usize>,
}

impl Shape{
    pub fn new(dim: Vec<usize>) -> Shape{
        // match dim {
        //     None => Shape {rank: 0, dim: None},
        //     Some(v) => Shape {rank: v.len(), dim: Some(v)},
        // }
        Shape {rank: dim.len(), dim: dim}
    }
}
