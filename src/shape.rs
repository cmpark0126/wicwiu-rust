#[derive(Debug, Clone)]
pub struct Shape{
    pub rank: usize,
    // if dim == None, Tensor which has this shape represents scalar type
    pub dim: Option<Vec<usize>>,
}

impl Shape{
    pub fn new(dim: Option<Vec<usize>>) -> Shape{
        match dim {
            None => Shape {rank: 0, dim: None},
            Some(v) => Shape {rank: v.len(), dim: Some(v)},
        }
    }
}
