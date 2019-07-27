use wicwiu::shape::Shape;

#[test]
fn shape_define() {
    let s = Shape::new(None);
    assert_eq!(s.rank, 0);
    assert_eq!(s.dim, None);

    let s = Shape::new(Some(vec![1]));
    assert_eq!(s.rank, 1);
    assert_eq!(s.dim, Some(vec![1]));

    let s = Shape::new(Some(vec![1, 2]));
    assert_eq!(s.rank, 2);
    assert_eq!(s.dim, Some(vec![1, 2]));
}

#[test]
fn shape_clone() {
    let s = Shape::new(None);
    let s_clone = s.clone();

    drop(s);

    assert_eq!(s_clone.rank, 0);
    assert_eq!(s_clone.dim, None);

    let s = Shape::new(Some(vec![1]));
    let s_clone = s.clone();

    drop(s);

    assert_eq!(s_clone.rank, 1);
    assert_eq!(s_clone.dim, Some(vec![1]));
}
