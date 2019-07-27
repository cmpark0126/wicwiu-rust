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
    pub fn zeros(dim: Option<Vec<usize>>) -> Tensor<T>{
        let shape = Shape::new(dim);
        let mut capacity = 1;

        match &shape.dim {
            None => {},
            Some(v) => {
                for i in v {
                    capacity *= i;
                }
            },
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
