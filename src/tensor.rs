use std::fmt::{self, Display, Formatter, Debug};
use crate::numeric::Numeric;
use crate::shape::Shape;

#[derive(Debug, Clone)]
pub struct Tensor<T>{
    pub shape: Shape,
    pub longarray: Vec<T>,
    pub gradient: Vec<T>,
}

impl<T> Tensor<T>
where T: Numeric + Clone + Display + Debug
{
    pub fn zeros(dim: Vec<usize>) -> Tensor<T>{
        let shape = Shape::new(dim);
        let mut capacity = 1;

        let dim = &shape.dim;

        for i in dim {
            capacity *= i;
        }

        let longarray : Vec<T> = vec![T::zero(); capacity];
        let gradient : Vec<T> = longarray.clone();

        Tensor {
            shape: shape,
            longarray: longarray,
            gradient: gradient,
        }
    }
}

impl<T> Display for Tensor<T>
where T: Numeric + Clone + Display + Debug
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({:#?}, {:#?})", self.shape, self.longarray)
    }
}
