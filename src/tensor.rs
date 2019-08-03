use std::fmt::{self, Display, Formatter, Debug};
use std::ops::{Index,IndexMut};
use std::rc::Rc;
use std::cell::RefCell;
use crate::numeric::Numeric;
use crate::shape::Shape;

#[derive(Debug, Clone)]
pub struct Tensor<T>{
    pub shape: Shape,
    pub longarray: Vec<T>,
    pub gradient: Option<Rc<RefCell<Tensor<T>>>>,
}

impl<T> Tensor<T>
where T: Numeric + Clone + Display + Debug
{
    pub fn zeros(dim: Vec<usize>, requires_grad: bool) -> Tensor<T>{
        let shape = Shape::new(dim);
        let mut capacity = 1;

        let dim = &shape.dim;

        for i in dim {
            capacity *= i;
        }

        let longarray : Vec<T> = vec![T::zero(); capacity];
        let mut gradient = None;

        if requires_grad == true{
            let gredient_t = Tensor {
                shape: shape.clone(),
                longarray: longarray.clone(),
                gradient: None,
            };
            gradient = Some(Rc::new(RefCell::new(gredient_t)));
        }

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

// impl<T> Index<usize> for Tensor<T>
// where
//     T: Numeric + Clone + Display + Debug
// {
//     type Output = T;
//
//     fn index<'a>(&'a self, index: usize) -> &'a Self::Output {
//         println!("Accessing {:?}-side of Tensor immutably", index);
//         match index {
//             i => &longarray
//         }
//     }
// }
//
// impl<T> IndexMut<usize> for Tensor<T>
// where
//     T: Numeric + Clone + Display + Debug
// {
//     fn index_mut<'a>(&'a mut self, index: Side) -> &'a mut Self::Output {
//         println!("Accessing {:?}-side of balance mutably", index);
//         match index {
//             Side::Left => &mut self.left,
//             Side::Right => &mut self.right,
//         }
//     }
// }
