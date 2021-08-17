/*
 * 
 * Create table for merged ways and lines to combine way (geo-info) with nodes
 * EPSG is transformed from 3857 to 4326
 */

DROP TABLE if exists ways_preprocessed

-- create table containing merged mandatory ways
CREATE TABLE ways_preprocessed AS
	select ways.osm_id, ways.highway, ways.geometry, ways.nodes, ST_Length(ways.geometry::geography) as length 
		from ( select ways.osm_id, ways.highway, ways.geometry, pow.id , pow.nodes
			from ( select l.osm_id as osm_id, l.highway as highway, st_transform(ST_SetSRID(l.way, 3857), 4326) as geometry
					from public.planet_osm_line l
					where l.highway like 'service'
					or l.highway like 'residential'
					or l.highway like 'secondary'
					or l.highway like 'tertiary'
					or l.highway like 'unclassified'
					or l.highway like 'primary'
					or l.highway like 'living_street'
					or l.highway like 'motorway'
					or l.highway like 'pedestrian'
					or l.highway like 'primary_link'
					or l.highway like 'motorway_link'
					or l.highway like 'trunk_link'
					or l.highway like 'secondary_link'
					or l.highway like 'tertiary_link'
					or l.highway like 'disused'
					or l.highway like 'trunk'
				) ways
			left join public.planet_osm_ways pow 
			on ways.osm_id = pow.id
	) ways
	
	

CREATE INDEX ON ways_preprocessed USING gist (geometry);


drop table if exists ways_with_segments



/*
 * 1st ST_DumpPoints to separate linestring into points
 * 2nd ST_MakeLine to make lines from linestring segments
 * 3rd group by osm_id and aggregate w. array_agg()
 *
 * with linestring segments:
 *
 * SELECT way_w_segments.osm_id, way_w_segments.highway, way_w_segments.nodes, way_w_segments.geometry, array_agg(way_w_segments.linestring_segment) as linestring_segments, array_agg(ST_Length(way_w_segments.linestring_segment::geography)) as length_segments

 *
 */
CREATE TABLE ways_with_segments_3035 as
	select ways.osm_id, ways.nodes, ways.highway, st_transform(ways.geometry, 3035) as geometry, ways.length_segments
	from (
		with way as (
			SELECT wp.osm_id, wp.highway, wp.nodes, wp.geometry, ST_DumpPoints(wp.geometry) as geo_dump FROM ways_preprocessed wp)
		SELECT way_w_segments.osm_id, way_w_segments.nodes, way_w_segments.highway, way_w_segments.geometry, array_agg(ST_Length(way_w_segments.linestring_segment::geography)) as length_segments
		FROM (
			SELECT way.osm_id, way.highway, way.nodes, way.geometry, ST_AsText(ST_MakeLine(lag((geo_dump).geom, 1, NULL) OVER (PARTITION BY way.osm_id ORDER BY way.osm_id, (geo_dump).path), (geo_dump).geom)) AS linestring_segment from way) way_w_segments
		WHERE way_w_segments.linestring_segment IS NOT null
		GROUP BY way_w_segments.osm_id, way_w_segments.highway, way_w_segments.nodes, way_w_segments.geometry
	) ways
	where ways.nodes is not null



CREATE INDEX ON ways_with_segments_3035 USING gist (geometry);


