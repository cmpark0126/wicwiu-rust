pub trait Floateric {
    fn zero() -> Self;
    fn one() -> Self;
}

impl Floateric for i8{
    fn zero() -> Self{
        0 as Self
    }
    fn one() -> Self{
        1 as Self
    }
}
impl Floateric for i16{
    fn zero() -> Self{
        0 as Self
    }
    fn one() -> Self{
        1 as Self
    }
}
impl Floateric for i32{
    fn zero() -> Self{
        0 as Self
    }
    fn one() -> Self{
        1 as Self
    }
}
impl Floateric for i64{
    fn zero() -> Self{
        0 as Self
    }
    fn one() -> Self{
        1 as Self
    }
}
impl Floateric for f32{
    fn zero() -> Self{
        0 as Self
    }
    fn one() -> Self{
        1 as Self
    }
}
impl Floateric for f64{
    fn zero() -> Self{
        0 as Self
    }
    fn one() -> Self{
        1 as Self
    }
}
