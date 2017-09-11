"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


import pickle
import os
import pandas as pd
import time
import numpy as np

import oemof.db as db
from ding0.tools.results import load_nd_from_pickle
from ding0.core import GeneratorDing0

########################################################
def validate_generation(conn, file = 'ding0_tests_grids_1.pkl'):
    '''Validate if total generation of a grid in a pkl file is what expected.
    
    Parameters
    ----------
    conn:
        The connection
    file: str
        name of file containing a grid to compare.
        
    Returns
    -------
    DataFrame
        compare_by_level
    DataFrame
        compare_by_type
    '''
    #load network
    nw = load_nd_from_pickle(filename=file)

    #config network intern variables
    nw._config = nw.import_config()
    nw._pf_config = nw.import_pf_config()
    nw._static_data = nw.import_static_data()
    nw._orm = nw.import_orm()

    #rescue generation from input table
    generation_input = nw.list_generators(conn)

    #make table of generators that are in the grid
    gen_idx = 0
    gen_dict = {}
    for mv_district in nw.mv_grid_districts():
        #search over MV grid
        for node in mv_district.mv_grid.graph_nodes_sorted():
            if isinstance(node, GeneratorDing0):
                gen_idx+=1
                subtype = node.subtype
                if subtype == None:
                    subtype = 'other'
                type = node.type
                if type == None:
                    type = 'other'
                gen_dict[gen_idx] = {
                    'v_level':node.v_level,
                    'type':type,
                    'subtype':subtype,
                    'GenCap':node.capacity,
                }

        #search over LV grids
        for LA in mv_district.lv_load_areas():
            for lv_district in LA.lv_grid_districts():
                # generation capacity
                for g in lv_district.lv_grid.generators():
                    gen_idx+=1
                    subtype = g.subtype
                    if subtype == None:
                        subtype = 'other'
                    type = g.type
                    if type == None:
                        type = 'other'
                    gen_dict[gen_idx] = {
                        'v_level':g.v_level,
                        'type':type,
                        'subtype':subtype,
                        'GenCap':g.capacity,
                    }

    generation_effective = pd.DataFrame.from_dict(gen_dict, orient='index')

    #compare by voltage level
    input_by_level = generation_input.groupby('v_level').sum()['GenCap'].apply(lambda x: np.round(x,3))
    effective_by_level = generation_effective.groupby('v_level').sum()['GenCap'].apply(lambda x: np.round(x,3))

    compare_by_level = pd.concat([input_by_level,effective_by_level,input_by_level==effective_by_level],axis=1)
    compare_by_level.columns = ['before','after','equal?']

    #compare by type/subtype
    generation_input['type'] =generation_input['type']+'/'+generation_input['subtype']
    generation_effective['type'] =generation_effective['type']+'/'+generation_effective['subtype']

    input_by_type = generation_input.groupby('type').sum()['GenCap'].apply(lambda x: np.round(x,3))
    effective_by_type = generation_effective.groupby('type').sum()['GenCap'].apply(lambda x: np.round(x,3))

    compare_by_type = pd.concat([input_by_type,effective_by_type,input_by_type==effective_by_type],axis=1)
    compare_by_type.columns = ['before','after','equal?']
    compare_by_type.index.names = ['type/subtype']

    return compare_by_level, compare_by_type

########################################################
def validate_load_areas(conn, file = 'ding0_tests_grids_1.pkl'):
    '''Validate if total load of a grid in a pkl file is what expected.
    
    Parameters
    ----------
    conn:
        The connection
    file: str
        name of file containing a grid to compare.
        
    Returns
    -------
    DataFrame
        compare_by_sector
    '''
    #load network
    nw = load_nd_from_pickle(filename=file)

    #config network intern variables
    nw._config = nw.import_config()
    nw._pf_config = nw.import_pf_config()
    nw._static_data = nw.import_static_data()
    nw._orm = nw.import_orm()

    #rescue generation from input table
    load_input = nw.list_load_areas(conn)

    print(load_input)






########################################################
if __name__ == "__main__":
    conn = db.connection(section='oedb')

    #compare_by_level, compare_by_type = validate_generation(conn)
    #print(compare_by_level)
    #print(compare_by_type)
    validate_load_areas(conn)

    conn.close()
