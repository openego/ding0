import dingo
from dingo.tools import config as cfg_dingo

import os


class AnimationDingo:
    """ Class for visual animation of routing process (basically a central place to store information about output file
        and count of saved images). Use argument 'animation=True' of method 'NetworkDingo.mv_routing()' to enable image
        export. Subsequently, FFMPEG can be used to convert images to animation, e.g.
            ffmpeg -r 10 -i mv-routing_ani_%04d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p mv-routing_ani.mp4
    """

    def __init__(self, **kwargs):
        output_animation_file_prefix = cfg_dingo.get('output', 'animation_file_prefix')
        package_path = dingo.__path__[0]

        self.file_path = os.path.join(package_path, 'output/animation/')
        self.file_prefix = output_animation_file_prefix
        self.counter = 1
