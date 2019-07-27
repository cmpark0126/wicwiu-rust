use std::fmt::{self, Display, Formatter, Debug};
use crate::numeric::Numeric;
use crate::shape::Shape;

#[derive(Debug, Clone)]
pub struct Tensor<T>{
    pub shape: Shape,
    pub longarray: Vec<T>,
}

impl<T: Numeric + Clone + Display + Debug> Tensor<T>{
    #[inline]
    pub fn zeros(dim: Vec<usize>) -> Tensor<T>{
        let shape = Shape::new(dim);
        let mut capacity = 1;

        let dim = &shape.dim;

        for i in dim {
            capacity *= i;
        }

        let longarray : Vec<T> = vec![T::zero(); capacity];

        Tensor {
            shape: shape,
            longarray: longarray,
        }
    }
}

impl<T: Numeric + Clone + Display + Debug> Display for Tensor<T>{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({:#?}, {:#?})", self.shape, self.longarray)
    }
}
