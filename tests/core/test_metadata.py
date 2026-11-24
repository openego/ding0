"""The metadata of a grid name the ding0 code and the database it is built from."""
import shutil
import subprocess
from types import SimpleNamespace

from sqlalchemy.engine import make_url

from ding0.core import NetworkDing0, git_version


def git(path, *command):
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.org", *command],
        cwd=path, check=True, capture_output=True, text=True).stdout.strip()


def make_checkout(path):
    """A git repository with a committed package `ding0`; returns the package."""
    package = path / "ding0"
    package.mkdir()
    (package / "__init__.py").write_text("")
    git(path, "init", "-q")
    git(path, "add", ".")
    git(path, "commit", "-q", "-m", "init")
    return package


def test_git_version_of_a_checkout_is_its_commit(tmp_path):
    package = make_checkout(tmp_path)
    assert git_version(str(package)) == git(tmp_path, "rev-parse", "--short", "HEAD")


def test_git_version_marks_uncommitted_changes(tmp_path):
    package = make_checkout(tmp_path)
    (package / "__init__.py").write_text("x = 1\n")
    assert git_version(str(package)).endswith("-dirty")


def test_git_version_of_a_copy_inside_another_repository_is_none(tmp_path):
    # e.g. the code frozen into the folder of a run inside the ding0 repository
    package = make_checkout(tmp_path)
    copy = tmp_path / "results" / "run" / "code" / "ding0"
    shutil.copytree(package, copy)
    assert git_version(str(copy)) is None


def test_metadata_names_the_database():
    network = NetworkDing0(name="test")
    url = make_url("postgresql://user@localhost/regon_dev_06-11")
    network._session = SimpleNamespace(bind=SimpleNamespace(url=url))
    assert network.metadata["database"] == "regon_dev_06-11"
