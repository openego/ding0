"""Hashes of ding0 objects that are the same in every run.

Python hashes an object by its memory address unless its class defines
otherwise, and it iterates a set in the order of the hashes of its elements.
ding0 iterates sets of its objects in many places, e.g. when it picks the
ring branch an urban stub is connected to or moves cable distributors onto
street nodes one after another, so the grid would depend on where Python
places the objects in memory. Objects of a class derived from
:class:`CreationOrderHash` are hashed by the number of their creation
instead. Equality stays identity.
"""

import itertools

_numbers = itertools.count()


def restart_creation_order():
    """Numbers the objects created from now on from 0 again.

    :class:`~ding0.core.NetworkDing0` calls this when it is created, so that
    a network gets the same hashes no matter what the process built before.
    """
    global _numbers
    _numbers = itertools.count()


class CreationOrderHash:
    """Hashes an object by the number of its creation.

    A copy (:func:`copy.copy`, :func:`copy.deepcopy`, pickle) is a new object
    and gets a number of its own when it is created. The number is not part
    of the copied state, so the hash of the copy does not change while the
    copy is put together, e.g. as a key of a dictionary it refers to.
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._creation_number = next(_numbers)
        return obj

    def __hash__(self):
        return self._creation_number

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_creation_number"]
        return state
