"""Hashes of ding0 objects, see `ding0.tools.creation_order`."""

import copy
import importlib
import inspect
import pickle
import pkgutil

import pytest

import ding0.core.network
import ding0.core.structure
import ding0.grid.mv_grid.models
from ding0.core import NetworkDing0
from ding0.core.network import BranchDing0, CableDistributorDing0
from ding0.tools.creation_order import CreationOrderHash, restart_creation_order


def classes_of(package):
    modules = [package] + [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + ".")]
    return {cls for module in modules
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if cls.__module__.startswith(package.__name__)}


@pytest.mark.parametrize("package", [
    ding0.core.network, ding0.core.structure, ding0.grid.mv_grid.models])
def test_no_grid_class_is_hashed_by_memory_address(package):
    assert {cls.__name__ for cls in classes_of(package)
            if not issubclass(cls, CreationOrderHash)} == set()


def test_hash_is_number_of_creation():
    restart_creation_order()
    objects = [BranchDing0(), CableDistributorDing0(), BranchDing0()]
    assert [hash(o) for o in objects] == [0, 1, 2]


def test_equality_stays_identity():
    restart_creation_order()
    a = BranchDing0()
    restart_creation_order()
    b = BranchDing0()
    assert hash(a) == hash(b)
    assert a != b
    assert len({a, b}) == 2


def test_new_network_counts_from_zero():
    BranchDing0()
    NetworkDing0(name="test")
    assert hash(BranchDing0()) == 0


@pytest.mark.parametrize("duplicate", [
    copy.deepcopy, lambda o: pickle.loads(pickle.dumps(o))])
def test_copy_is_found_in_a_dictionary_it_refers_to(duplicate):
    # the copy is a key of a dictionary that is copied along with it, like
    # a node in the graph of its grid
    original = BranchDing0(length=5)
    original.lookup = {original: "itself"}
    duplicated = duplicate(original)
    assert duplicated.lookup[duplicated] == "itself"
    assert duplicated.length == 5


@pytest.mark.parametrize("duplicate", [
    copy.copy, copy.deepcopy, lambda o: pickle.loads(pickle.dumps(o))])
def test_copy_gets_a_number_of_its_own(duplicate):
    original = BranchDing0()
    duplicated = duplicate(original)
    assert hash(duplicated) != hash(original)
    assert hash(BranchDing0()) == hash(duplicated) + 1
