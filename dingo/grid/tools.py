def cable_type(nom_power, nom_voltage, avail_cables):
    """
    Determine suitable type of cable for given nominal power

    Based on maximum occurring current which is derived from nominal power
    (either peak load or max. generation capacity) a suitable cable type is
    chosen. Thus, no line overloading issues should occur.

    Parameters
    ----------
    nom_power : numeric
        Nominal power of generators or loads connected via a cable
    nom_voltage : numeric
        Nominal voltage in kV
    avail_cables : pandas.DataFrame
        Available cable types including it's electrical parameters
    Returns
    -------
    cable_type : pandas.DataFrame
        Parameters of cable type
    """

    I_max_load = nom_power / (3 ** 0.5 * nom_voltage)

    # determine suitable cable for this current
    suitable_cables = avail_cables[avail_cables['I_max_th'] > I_max_load]
    cable_type = suitable_cables.ix[suitable_cables['I_max_th'].idxmin()]

    return cable_type