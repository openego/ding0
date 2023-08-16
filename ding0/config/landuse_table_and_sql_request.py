# landuse has to be fetched and stored in DB as selct * from osm data where osmdata.landuse is not null;

class Landuse(Base):
    """Landuse Model"""

    __tablename__ = "landuse"

    osm_id = Column(Integer, primary_key=True)
    landuse = Column(String(50))
    natural = Column(String(50))
    geometry = Column(Geometry('POLYGON'))
    area = Column(Float)



def get_osm_landuse(geo_area, session_osm):
    """ load ways from db for given polygon as geo_area_wkt """
 
    landuse = session_osm.query(Landuse).filter(
        func.st_intersects(func.ST_GeomFromText(geo_area, srid), Landuse.geometry))

    landuse_sql_df = pd.read_sql(
                landuse.statement,
                con=session_osm.bind)
# con=engine_osm both ways are working. select the easier/ more appropriate one
    return landuse_sql_df
