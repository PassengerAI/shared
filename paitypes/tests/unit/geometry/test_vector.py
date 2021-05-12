from hypothesis import given
from hypothesis.strategies import register_type_strategy
from hypothesis.strategies import from_type
from hypothesis.strategies import floats
from hypothesis.extra import numpy as np
from paitypes.geometry.Point import Point
from paitypes.geometry.Vector import Vector
from numpy.testing import assert_equal
import numpy

register_type_strategy(float, floats(allow_nan=False,
                                     max_value=1e10,
                                     min_value=-1e10,
                                     ))


@given(np.arrays(numpy.float32, 3))
def test_from_np(a: numpy.ndarray) -> None:
    assert_equal(Vector.from_np_array(a).to_np_array(), a)


@given(from_type(Vector))
def test_to_np(v: Vector) -> None:
    assert Vector.from_np_array(v.to_np_array()) == v


@given(from_type(Vector))
def test_add_zero(v: Vector) -> None:
    assert v + Vector(0, 0, 0) == v


@given(from_type(Vector), from_type(Vector))
def test_add_is_commutative(v1: Vector, v2: Vector) -> None:
    assert v1 + v2 == v2 + v1


@given(from_type(Vector))
def test_sub_zero(v: Vector) -> None:
    assert v - Vector(0, 0, 0) == v


@given(from_type(Vector))
def test_sub_self(v: Vector) -> None:
    assert v - v == Vector(0, 0, 0)


@given(from_type(Vector))
def test_sub_inverts_add(v: Vector) -> None:
    assert v - v == v + (-v)


@given(from_type(Vector))
def test_dot_zero(v: Vector) -> None:
    assert Vector.dot(v, Vector(0.0, 0.0, 0.0)) == 0.0


@given(from_type(Vector))
def test_cross_zero(v: Vector) -> None:
    assert Vector.cross(v, Vector(0.0, 0.0, 0.0)) == Vector(0.0, 0.0, 0.0)


@given(from_type(Vector), from_type(Vector))
def test_cross_is_anticommutative(v1: Vector, v2: Vector) -> None:
    assert Vector.cross(v1, v2) == -Vector.cross(v2, v1)


@given(from_type(Vector), from_type(Vector))
def test_planar_vectors_cross_is_perpendicular_to_plane(v1: Vector,
                                                        v2: Vector) -> None:
    v1.z = v2.z = 0
    cross = Vector.cross(v1, v2)
    assert cross.x == 0 and cross.y == 0


@given(from_type(Point))
def test_from_point_to_origin(p: Point) -> None:
    assert Vector.from_points(p, Point(0, 0)) == Vector(-p.x, -p.y)


@given(from_type(Point))
def test_from_origin_to_point(p: Point) -> None:
    assert Vector.from_points(Point(0, 0), p) == Vector(p.x, p.y)


def test_add() -> None:
    assert Vector(1, 2, 3) + Vector(4, 5, 6) == Vector(5, 7, 9)


def test_subtract() -> None:
    assert Vector(1, 2, 3) - Vector(4, 5, 6) == Vector(-3, -3, -3)


def test_cross() -> None:
    assert (Vector.cross(Vector(1, 2, 3), Vector(4, 5, 6)) ==
            Vector(-3, 6, -3))


def test_dot() -> None:
    assert Vector.dot(Vector(1, 2, 3), Vector(4, 5, 6)) == 32


def test_from_points() -> None:
    assert Vector.from_points(Point(1, 2), Point(-1, -2)) == Vector(-2, -4)
