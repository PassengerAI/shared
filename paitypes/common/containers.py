from collections import OrderedDict
from typing import Any, Optional, Callable


class SizeLimitedDictionary(OrderedDict):
    """An ordered dictionary with limited size and default dict.

    This class is a combination of `defaultdict` and `OrderedDict`,
    but additionally has a `max_size` attribute which limits the size of the
    dictionary, to avoid memory leaks. Adding more items than `max_size` will
    remove previous items, starting from the oldest one.
    """

    def __init__(self,
                 max_size: int = 10,
                 default_factory: Optional[Callable[[], Any]] = None,
                 *args: Any,
                 **kwargs: Any) -> None:
        self._default_factory = default_factory
        self._max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        if len(self.items()) == self._max_size:
            self.popitem(last=False)
        super().__setitem__(key, value)

    def __missing__(self, key: Any) -> Any:
        if not self._default_factory:
            raise KeyError(key)
        else:
            value = self._default_factory()
            self.__setitem__(key, value)
            return value
