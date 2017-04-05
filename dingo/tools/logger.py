import os

def create_dir(dirpath):
    """
    Create directory and report about it

    Parameters
    ----------
    dirpath : str
        Directory including path
    """

    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)

        print("We create a directory for you and your Dingo data: {}".format(
            dirpath))


def create_home_dir(dingo_path=None):
    """
    Check in ~/.dingo exists, otherwise create it

    Parameters
    ----------
    dingo_path : str
        Path to store Dingo related data (logging, etc)
    """

    if dingo_path is None:
        dingo_path = get_default_home_dir()

    create_dir(dingo_path)


def get_default_home_dir():
    """
    Return default home directory of Dingo

    Returns
    -------
    homedir : str
        Default home directory including its path
    """
    return os.path.join(os.path.expanduser('~'), '.dingo')