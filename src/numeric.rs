pub trait Numeric {
    fn zero() -> Self;
    fn one() -> Self;
}

impl Numeric for i8{
    fn zero() -> Self{
        0 as Self
    }
    fn one() -> Self{
        1 as Self
    }
}
impl Numeric for i16{
    fn zero() -> Self{
        0 as Self
    }
    fn one() -> Self{
        1 as Self
    }
}
impl Numeric for i32{
    fn zero() -> Self{
        0 as Self
    }
    fn one() -> Self{
        1 as Self
    }
}
impl Numeric for i64{
    fn zero() -> Self{
        0 as Self
    }
    fn one() -> Self{
        1 as Self
    }
}
impl Numeric for f32{
    fn zero() -> Self{
        0 as Self
    }
    fn one() -> Self{
        1 as Self
    }
}
impl Numeric for f64{
    fn zero() -> Self{
        0 as Self
    }
    fn one() -> Self{
        1 as Self
    }
}
