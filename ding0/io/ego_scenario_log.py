"""
Write entry into scenario log table
"""

__copyright__   = "Reiner Lemoine Institut gGmbH"
__license__     = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__         = "https://github.com/openego/data_processing/blob/master/LICENSE"
__author__ 	    = "nesnoj, Ludee"


from datetime import datetime
from sqlalchemy.orm import sessionmaker
from egoio.db_tables.model_draft import ScenarioLog as orm_scenario_log


def write_scenario_log(conn, project, version, io, schema, table,
                           script, entries=None, comment=None, metadata=None):
    """
    Write entry into scenario log table
    
    Parameters
    ----------
    conn: SQLAlchemy connection object
    project: str
        Project name
    version: str
             Version number
    io: str
        IO-type (input, output, temp)
    schema: str
            Database schema
    table: str
           Database table
    script: str
            Script name
    entries: int
             Number of entries
    comment: str
             Comment
    metadata: str
             Meta data
    
    Example
    -------
    write_scenario_log(conn=conn,
                           project='eGoDP'
                           version='v0.3.0',
                           io='output',
                           schema='model_draft',
                           table='ego_demand_loadarea_peak_load',
                           script='peak_load_per_load_area.py',
                           entries=1000)
    """

    Session = sessionmaker(bind=conn)
    session = Session()
    
    # extract user from connection details
    # is there a better way?
    try:
        conn_details = conn.connection.connection.dsn
        for entry in conn_details.split(' '):
            if entry.split('=')[0] == 'user':
                user = entry.split('=')[1]
                break
    except:
        user = 'unknown'
    
    # Add data to orm object
    log_entry = orm_scenario_log(project=project,
                                 version=version,
                                 io=io,
                                 schema_name=schema,
                                 table_name=table,
                                 script_name=script,
                                 entries=entries,
                                 comment=comment,
                                 user_name=user,
                                 timestamp=datetime.now(),
                                 meta_data=metadata)

    # Commit to DB
    session.add(log_entry)
    session.commit()
