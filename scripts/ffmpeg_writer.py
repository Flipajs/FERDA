from __future__ import unicode_literals
from builtins import str
from builtins import object
import subprocess
import numpy as np


class VideoSink(object) :

    def __init__( self, size, filename="output", rate=30, byteorder="bgra" ) :
            self.size = size
            str_size = str(size[1])+'x'+str(size[0])

            cmdstring  = ('avconv',
                            '-y', # (optional) overwrite output file if it exists
                            '-f', 'rawvideo',
                            '-vcodec', 'rawvideo',
                            '-s', str_size, # size of one frame
                            '-pix_fmt', 'bgr24',
                            '-r', str(rate), # frames per second
                            '-i', '-', # The imput comes from a pipe
                            '-an', # Tells FFMPEG not to expect any audio
                            '-force_fps',
                            'my_output_videofile12.mp4'
                    )
            self.p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=False)

    def run(self, image) :
            assert image.shape == self.size
            self.p.stdin.write(image.tostring())

    def close(self) :
            self.p.stdin.close()