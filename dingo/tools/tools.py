def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.

    Parameters
    ----------
    x: dict
    y: dict

    Notes
    -----
    This function was originally proposed by
    http://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression

    Credits to Thomas Vander Stichele. Thanks for sharing ideas!

    Returns
    -------
    z: dict
        Merged dictionary keyed by top-level keys of both dicts
    '''

    z = x.copy()
    z.update(y)
    return z