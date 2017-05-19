# reinforcement measures according to dena


def parallel_branch(grid, node_target):
    """ Reinforce MV or LV grid by installing a new parallel branch according to dena

    Parameters
    ----------
        grid : GridDingo 
            Grid identifier.
        node_target: int
            node where the parallel cable (starting from HV/MV substation) is connected to (used when grid is a MV grid)

    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
    """
    pass


def split_ring(grid):
    """ Reinforce MV grid by splitting a critical ring into two new rings according to dena
    Parameters
    ----------
        grid : MVGridDingo 
            Grid identifier.
          
    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
    """
    pass


def extend_substation(grid):
    """ Reinforce MV or LV substation by exchanging the existing trafo and installing a parallel one if necessary with according to dena

    Parameters
    ----------
        grid : GridDingo 
            Grid identifier.

    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
    """
    pass


def new_substation(grid):
    """ Reinforce MV grid by installing a new primary substation opposite to the existing one according to dena

    Parameters
    ----------
        grid : MVGridDingo 
            Grid identifier.
            
    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
    """