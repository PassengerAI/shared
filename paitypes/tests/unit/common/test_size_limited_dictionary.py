import pytest
from typing import Any
from hypothesis import given, strategies

from paitypes.common.containers import SizeLimitedDictionary

_DEFAULT = 'default'


@given(
    input_dict=strategies.dictionaries(keys=strategies.integers(max_value=100,
                                                                min_value=0),
                                       values=strategies.just('value'),
                                       max_size=20))
def test_size_is_limited(input_dict: Any) -> None:
    d = SizeLimitedDictionary(10, None, input_dict)
    assert len(d) <= 10


def test_insert_missing_keys() -> None:
    d = SizeLimitedDictionary(10, lambda: 1)
    for i in range(100):
        _ = d[i]
    assert len(d) == 10


def test_no_default_factory_raise() -> None:
    d = SizeLimitedDictionary()
    with pytest.raises(KeyError):
        v = d['a']


def test_with_default_factory() -> None:
    d = SizeLimitedDictionary(default_factory=lambda: _DEFAULT)
    assert d['a'] == _DEFAULT


def test_old_keys_are_deleted_on_overflow() -> None:
    d = SizeLimitedDictionary(max_size=10)
    for i in range(100):
        d[i] = i

    assert list(d.keys()) == list(range(90, 100))


def test_old_keys_raise_without_default_factory() -> None:
    d = SizeLimitedDictionary(max_size=10)
    for i in range(100):
        d[i] = i

    with pytest.raises(KeyError):
        v = d[0]


def test_old_keys_set_to_default() -> None:
    d = SizeLimitedDictionary(max_size=10,
                              default_factory=lambda: _DEFAULT)
    for i in range(100):
        d[i] = i

    assert d[0] == _DEFAULT
