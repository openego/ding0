"""Invariants of ding0 grids: physics, geography and completeness.

Every check returns a list of violations (strings), empty if the invariant
holds. A check reports all violations it finds, not only the first one, so a
failing test shows the whole picture. The checks run on any ding0 network:
the synthetic grid districts (`tests/synthetic_district.py`) and grids built
from an egon-data database.

* `physics`: Kirchhoff, Ohm, energy balance, voltage band, thermal limits,
  transformer sizing, radial operation, equipment data
* `geo`: coordinate reference system, containment, line geometries
* `completeness`: input data versus grid versus export
* `powerflow`: an independent power flow on the exported grid
"""


def assert_no_violations(violations, limit=30):
    """Fails with the first `limit` violations, if there are any."""
    if violations:
        shown = "\n".join(str(v) for v in violations[:limit])
        more = f"\n... and {len(violations) - limit} more" if len(violations) > limit else ""
        raise AssertionError(f"{len(violations)} violation(s):\n{shown}{more}")
