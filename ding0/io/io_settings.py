__copyright__ = "Reiner Lemoine Institut gGmbH"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__ = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__ = "jh-RLI"

import os
from configobj import ConfigObj


io_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ding0_config_path = os.path.join(io_base_dir, 'config', 'exporter_config.cfg')
exporter_config = ConfigObj(ding0_config_path)
