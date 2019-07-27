use wicwiu::shape::Shape;

#[test]
fn shape_define() {
    let s = Shape::new(0, None);
    assert_eq!(s.rank, 0);
    assert_eq!(s.dim, None);

    let s = Shape::new(1, Some(vec![1]));
    assert_eq!(s.rank, 1);
    assert_eq!(s.dim, Some(vec![1]));

    let s = Shape::new(2, Some(vec![1, 2]));
    assert_eq!(s.rank, 2);
    assert_eq!(s.dim, Some(vec![1, 2]));
}

#[should_panic(expected = "Guess rank and length of dimention vector must be equal to each other, \
                            got rank = 2, length of dimention 1.")]
#[test]
#[allow(unused_variables)]
fn shape_define_panic_case() {
    let s = Shape::new(2, Some(vec![1]));
}

#[test]
fn shape_clone() {
    let s = Shape::new(0, None);
    let s_clone = s.clone();

    drop(s);

    assert_eq!(s_clone.rank, 0);
    assert_eq!(s_clone.dim, None);

    let s = Shape::new(1, Some(vec![1]));
    let s_clone = s.clone();

    drop(s);

    assert_eq!(s_clone.rank, 1);
    assert_eq!(s_clone.dim, Some(vec![1]));
}
