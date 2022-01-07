"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from ding0.tools import config as cfg_ding0
from ding0.tools.logger import get_default_home_dir

import os


class AnimationDing0:
    """ Class for visual animation of the routing process (solving CVRP).
    
    (basically a central place to store information about output file and count of saved images).
    Use argument 'animation=True' of method 'NetworkDing0.mv_routing()' to enable image export.
    The images are exported to ding0's home dir which is usually ~/.ding0/ .
    
    Subsequently, FFMPEG can be used to convert images to animation, e.g.

        ffmpeg -r 5 -i mv-routing_ani_%04d.png -vframes 200 -r 15 -vcodec libx264 -y -an mv-routing_ani.mp4 -s 640x480
    
    See Also
    --------
    ding0.core.NetworkDing0.mv_routing() :
    """

    def __init__(self, **kwargs):
        output_animation_file_prefix = cfg_ding0.get('output', 'animation_file_prefix')

        self.file_path = os.path.join(get_default_home_dir(), 'animation/')
        self.file_prefix = output_animation_file_prefix
        self.counter = 1
