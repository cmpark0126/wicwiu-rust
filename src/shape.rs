use std::cmp::Eq;

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

    pub fn capacity(&self) -> usize{
        let mut capacity: usize = 1;

        for i in 0..self.rank{
            capacity *= self.dim[i];
        }

        capacity
    }
}
