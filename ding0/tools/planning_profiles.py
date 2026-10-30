"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io

Planning profiles: sets of planning principles (thresholds, load factors,
voltage limits, equipment) that ding0 uses to build the grids.

The parameters are stored in `config/planning_parameters.csv`, the allowed
equipment in `config/equipment_selection.csv`, with one column per profile.
The column `default` holds general planning principles in Germany as used for
the eGon grids. Other profiles only contain the values that differ from
`default`; an empty cell inherits the `default` value.

Applying a profile writes its parameters into ding0's config (sections as in
`config_calc.cfg`, read with :func:`ding0.tools.config.get`) and into the
parameters returned by :func:`ding0.config.config_lv_grids_osm.get_config_osm`.
"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj"

import csv
import logging
import os

import ding0
from ding0.tools import config as cfg_ding0

logger = logging.getLogger(__name__)

PARAMETERS_FILE = "planning_parameters.csv"
EQUIPMENT_SELECTION_FILE = "equipment_selection.csv"
DEFAULT_PROFILE = "default"
NO_LIMIT = "none"

# Section of parameters returned by `get_config_osm()` instead of the config
OSM_SECTION = "lv_grids_osm"

PARAMETERS_META_COLUMNS = ["section", "key", "unit", "description"]
EQUIPMENT_META_COLUMNS = ["type", "name"]

# Column (or index if None) of the equipment tables in
# `NetworkDing0.static_data` that holds the name used in the selection table
EQUIPMENT_NAME_COLUMNS = {
    "MV_trafos": "S_nom",
    "MV_cables": "name",
    "MV_overhead_lines": "name",
    "LV_trafos": "S_nom",
    "LV_cables": None,
}

_active_profile = DEFAULT_PROFILE
_osm_parameters = {}


def _read_table(filename, meta_columns):
    """
    Reads a profile table (csv module, as it is read when loading the config).

    Returns the profile names and the rows as dicts with stripped strings.
    """
    path = os.path.join(ding0.__path__[0], "config", filename)
    with open(path, newline="") as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    columns = [c.strip() for c in reader.fieldnames]
    rows = [
        {key.strip(): (value or "").strip() for key, value in row.items()}
        for row in reader
    ]

    missing = [c for c in meta_columns + [DEFAULT_PROFILE] if c not in columns]
    if missing:
        raise ValueError(f"Columns {missing} missing in '{filename}'.")
    ids = [(row[meta_columns[0]], row[meta_columns[1]]) for row in rows]
    duplicates = sorted({id_ for id_ in ids if ids.count(id_) > 1})
    if duplicates:
        raise ValueError(f"Duplicate entries in '{filename}': {duplicates}.")
    empty = [id_ for id_, row in zip(ids, rows) if not row[DEFAULT_PROFILE]]
    if empty:
        raise ValueError(
            f"Column '{DEFAULT_PROFILE}' in '{filename}' must not have empty cells: {empty}."
        )
    profiles = [c for c in columns if c not in meta_columns]
    return profiles, rows


def _profile_value(row, profile):
    """Value of `profile` in a table row, the default value if not set."""
    return row[profile] or row[DEFAULT_PROFILE]


def _check_profile(profile, profiles, filename):
    if profile not in profiles:
        raise ValueError(
            f"Unknown planning profile '{profile}' in '{filename}'. "
            f"Available profiles: {profiles}."
        )


def _parse_value(value):
    """Converts a value string into int, float, None (`NO_LIMIT`) or str."""
    if value.lower() == NO_LIMIT:
        return None
    for type_ in (int, float):
        try:
            return type_(value)
        except ValueError:
            pass
    return value


def available_profiles():
    """
    Returns the names of all planning profiles.

    Returns
    -------
    :obj:`list` of :obj:`str`
        Profile names
    """
    profiles, _ = _read_table(PARAMETERS_FILE, PARAMETERS_META_COLUMNS)
    return profiles


def get_parameters(profile=DEFAULT_PROFILE):
    """
    Returns the planning parameters of a profile.

    Parameters
    ----------
    profile : :obj:`str`
        Name of the planning profile

    Returns
    -------
    :obj:`dict`
        Parameter values as strings, keys are (section, key) tuples. Values
        that are not set in the profile are taken from the default profile.
    """
    profiles, rows = _read_table(PARAMETERS_FILE, PARAMETERS_META_COLUMNS)
    _check_profile(profile, profiles, PARAMETERS_FILE)
    return {(row["section"], row["key"]): _profile_value(row, profile) for row in rows}


def get_equipment_selection(profile=DEFAULT_PROFILE):
    """
    Returns the equipment that may be used in a profile.

    Parameters
    ----------
    profile : :obj:`str`
        Name of the planning profile

    Returns
    -------
    :obj:`dict`
        Names of the selected equipment (:obj:`set` of :obj:`str`) per
        equipment type, e.g. 'LV_cables'
    """
    profiles, rows = _read_table(EQUIPMENT_SELECTION_FILE, EQUIPMENT_META_COLUMNS)
    _check_profile(profile, profiles, EQUIPMENT_SELECTION_FILE)

    selection = {type_: set() for type_ in EQUIPMENT_NAME_COLUMNS}
    for row in rows:
        if row["type"] not in selection:
            raise ValueError(
                f"Unknown equipment type '{row['type']}' in '{EQUIPMENT_SELECTION_FILE}'."
            )
        value = _profile_value(row, profile).lower()
        if value not in ["true", "false"]:
            raise ValueError(
                f"Values in '{EQUIPMENT_SELECTION_FILE}' must be 'true' or 'false', "
                f"got '{value}' for {row['type']} '{row['name']}'."
            )
        if value == "true":
            selection[row["type"]].add(row["name"])
    return selection


def _listed_equipment():
    """All equipment names per type listed in the selection table."""
    _, rows = _read_table(EQUIPMENT_SELECTION_FILE, EQUIPMENT_META_COLUMNS)
    listed = {type_: set() for type_ in EQUIPMENT_NAME_COLUMNS}
    for row in rows:
        listed.setdefault(row["type"], set()).add(row["name"])
    return listed


def apply_profile(profile=DEFAULT_PROFILE):
    """
    Applies the parameters of a planning profile to ding0's config.

    The parameters are written into the config sections read with
    :func:`ding0.tools.config.get` and into the parameters returned by
    :func:`ding0.config.config_lv_grids_osm.get_config_osm`. The profile is
    applied again whenever `config_calc.cfg` is (re)loaded.

    Parameters
    ----------
    profile : :obj:`str`
        Name of the planning profile
    """
    global _active_profile

    parameters = get_parameters(profile)
    lv_standard_line = parameters[("assumptions", "lv_standard_line")]
    if lv_standard_line not in get_equipment_selection(profile)["LV_cables"]:
        raise ValueError(
            f"lv_standard_line '{lv_standard_line}' of planning profile '{profile}' "
            f"is not selected in '{EQUIPMENT_SELECTION_FILE}'."
        )

    osm_parameters = {}
    for (section, key), value in parameters.items():
        if section == OSM_SECTION:
            osm_parameters[key] = _parse_value(value)
        else:
            if not cfg_ding0.cfg.has_section(section):
                cfg_ding0.cfg.add_section(section)
            cfg_ding0.cfg.set(section, key, value)

    _osm_parameters.clear()
    _osm_parameters.update(osm_parameters)
    _active_profile = profile
    logger.info(f"Planning profile '{profile}' applied.")


def get_active_profile():
    """
    Returns the name of the planning profile that is currently applied.

    Returns
    -------
    :obj:`str`
        Profile name
    """
    return _active_profile


def get_osm_parameter(key):
    """
    Returns a planning parameter of the section `lv_grids_osm`.

    Applies the active profile first if this has not been done yet.

    Parameters
    ----------
    key : :obj:`str`
        Parameter name

    Returns
    -------
    :obj:`int`, :obj:`float`, :obj:`str` or None
        Parameter value
    """
    if not _osm_parameters:
        apply_profile(_active_profile)
    return _osm_parameters[key]


def get_limit(section, key):
    """
    Returns a numeric limit from the config, None if set to 'none' (no limit).

    Parameters
    ----------
    section : :obj:`str`
        Config section
    key : :obj:`str`
        Parameter name

    Returns
    -------
    :obj:`float` or None
        Limit
    """
    value = cfg_ding0.get(section, key)
    if isinstance(value, str) and value.lower() == NO_LIMIT:
        return None
    return value


def select_equipment(static_data, profile=DEFAULT_PROFILE):
    """
    Removes equipment that is not selected in a planning profile.

    Parameters
    ----------
    static_data : :obj:`dict`
        Equipment tables, see :attr:`~.core.NetworkDing0.static_data`
    profile : :obj:`str`
        Name of the planning profile

    Returns
    -------
    :obj:`dict`
        Equipment tables with the selected equipment only
    """
    listed_equipment = _listed_equipment()
    selection = get_equipment_selection(profile)

    for type_, name_column in EQUIPMENT_NAME_COLUMNS.items():
        equipment = static_data[type_]
        if name_column is None:
            names = equipment.index.astype(str)
        else:
            names = equipment[name_column].astype(str)

        listed = listed_equipment[type_]
        not_listed = set(names) - listed
        unknown = listed - set(names)
        if not_listed or unknown:
            raise ValueError(
                f"'{EQUIPMENT_SELECTION_FILE}' does not match the equipment data of "
                f"'{type_}': not listed {sorted(not_listed)}, unknown {sorted(unknown)}."
            )

        selected = equipment.loc[[name in selection[type_] for name in names]]
        if selected.empty:
            raise ValueError(f"No equipment of type '{type_}' selected in planning profile '{profile}'.")
        if name_column is not None:
            selected = selected.reset_index(drop=True)
        static_data[type_] = selected

    return static_data
