Calculation principles
~~~~~~~~~~~~~~~~~~~~~~

Reactance
#########

We assume all cables and overhead lines to be short lines. Thus, the capacity
is not considered in calculation of reactance of overhead lines and cables.

.. math::
    x = \omega * L


Apparent power
##############

* Given maximum thermal current `I_th_amx` (`I_L`) is given per conducter (of three
  cables in a system)/per phase.

* We assume to have delta connection. Thus, nominal voltage per conducted can be
  applied to calculate apparent power `s_nom` and conductor current `I_L` has to
  be transformed to `I_delta` respectively to `I` by

  .. math::
      I = I_{delta} = \sqrt{3} \cdot I_L

* Apparent `S` power is calculated to

  .. math::
      S = U \cdot I = U \cdot I_{th,max}

