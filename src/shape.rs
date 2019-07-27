#[derive(Debug, Clone)]
pub struct Shape{
    pub rank: usize,
    // if dim == None, Tensor which has this shape represents scalar type
    pub dim: Option<Vec<usize>>,
}

impl Shape{
    pub fn new(rank: usize, dim: Option<Vec<usize>>) -> Shape{
        match dim {
            None => Shape {rank: rank, dim: None},
            Some(v) => {
                if rank != v.len(){
                    panic!("Guess rank and length of dimention vector must be equal to each other, \
                            got rank = {}, length of dimention {}.", rank, v.len());
                }
                Shape {rank: rank, dim: Some(v)}
            },
        }
    }
}
