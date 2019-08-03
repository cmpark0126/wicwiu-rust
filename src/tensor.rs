use std::fmt::{self, Display, Formatter, Debug};
use std::ops::{Index,IndexMut};
use std::rc::Rc;
use std::cell::RefCell;
use num::{Num, NumCast, Float, FromPrimitive};
use crate::shape::Shape;

#[derive(Debug, Clone)]
pub struct Tensor<T>{
    pub shape: Shape,
    pub longarray: Vec<T>,
    pub gradient: Option<Rc<RefCell<Tensor<T>>>>,
}

impl<T> Tensor<T>
where T: Num + NumCast + Float + Clone + FromPrimitive
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
                longarray: vec![T::zero(); capacity],
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

    pub fn ones(dim: Vec<usize>, requires_grad: bool) -> Tensor<T>{
        let shape = Shape::new(dim);
        let mut capacity = 1;

        let dim = &shape.dim;

        for i in dim {
            capacity *= i;
        }

        let longarray : Vec<T> = vec![T::one(); capacity];
        let mut gradient = None;

        if requires_grad == true{
            let gredient_t = Tensor {
                shape: shape.clone(),
                longarray: vec![T::zero(); capacity],
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

    pub fn item(&self, index: usize) -> &T {
        println!("Accessing {:?}-side of Tensor immutably", index);
        &self.longarray[index]
    }

    pub fn item_mut(&mut self, index: usize) -> &mut T {
        println!("Accessing {:?}-side of Tensor immutably", index);
        &mut self.longarray[index]
    }
}

// impl<T> Display for Tensor<T>
// where T: Num + NumCast + Float + Clone + FromPrimitive
// {
//     fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
//         write!(f, "({:#?}, {:#?})", self.shape, self.longarray)
//     }
// }

// impl<T> Index<usize> for Tensor<T>
// where
//     T: Num + NumCast + Float + Clone + FromPrimitive
// {
//     type Output = T;
//
//     fn index<'a>(&'a self, index: usize) -> &'a Self::Output {
//         println!("Accessing {:?}-side of Tensor immutably", index);
//         &self.longarray[index]
//     }
// }
//
// impl<T> IndexMut<usize> for Tensor<T>
// where
//     T: Num + NumCast + Float + Clone + FromPrimitive
// {
//     fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut Self::Output {
//         println!("Accessing {:?}-side of Tensor mutably", index);
//         &mut self.longarray[index]
//     }
// }
