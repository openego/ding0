import numpy as np
import pandas as pd

def compare_data_frames_by_tolerance(df_orig, df_comp, absolute_tolerance = 1e-5, relative_tolerance = 1e-3):
    '''
    Function that compares pandas.DataFrames by absolute and relative tolerance. Returns True if inserted dataframes
    fulfill defined either absolute or relative tolerances.

    Parameters
    ----------
    df_orig: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe which counts as original to compare to
    df_comp: :pandas:`pandas.DataFrame<dataframe>`
        Dataframe being compared to original
    absolute_tolerance: float
        absolute tolerance with which values count as same, by default 1e-5
    relative_tolerance: float
        relative tolerance with which values count as same, by default 0.1%

    Returns
    -------
    is_equal: bool
        Boolean that indicates whether dataframes are seen as equal given the selected tolerances
    count_of_unequal entries: int
    '''

    # check which entries are exactly the same
    is_same = (df_orig == df_comp)._values
    # separately check for nan
    is_nan = (pd.isnull(df_orig)._values*pd.isnull(df_comp)._values)
    if(sum(sum(pd.isnull(df_orig)._values != pd.isnull(df_comp)._values))):
        return False
    # check which entries fulfill absolute tolerance
    is_almost_equal = (abs(np.subtract(df_comp,df_orig)) < absolute_tolerance)._values
    # check which entries fulfill relative tolerance
    correction_factor = (df_orig == 0) * 1e-5
    is_relatively_equal = (abs(np.subtract(df_comp,df_orig))/(df_orig+correction_factor) < relative_tolerance)._values
    # if either relative or absolute tolerance are fulfilled or values are the same, the entry counts as equal;
    count_of_unequal_entries = sum(sum(~is_same*~is_nan*~is_almost_equal*~is_relatively_equal))
    # check weather all entries are equal
    is_equal = count_of_unequal_entries == 0
    return is_equal, count_of_unequal_entries