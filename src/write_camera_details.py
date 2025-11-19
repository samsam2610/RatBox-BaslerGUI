"""
Camera Control
Copyright M. Mathis Lab
Written by  Gary Kane - https://github.com/gkane26
post-doctoral fellow @ the Adaptive Motor Control Lab
https://github.com/AdaptiveMotorControlLab

create json parameter file
"""

import os
from pathlib import Path
import json
import numpy as np

path = Path(os.path.realpath(__file__))
# Navigate to the outer parent directory and join the filename
out = os.path.normpath(str(path.parents[2] / 'config-files' / 'camera_details.json'))
# out = os.path.normpath(path.parent.absolute() + 'camera_details.json')

# Crop, rotation, and exposure are default parameters. Can be changed in the GUI.

cam_0 = {'name': 'cam1',
         'crop': {'offset_X': 0, 'offset_Y': 0, 'width': 1376, 'height': 960},
         'rotate': 0,
         'exposure': 1500, # in microseconds
         'gain': 30, # in db
         'offset': {'x': 0, 'y': 100},
         'output_dir': 'C:\\Data folder'}

cam_1 = {'name': 'cam2',
         'crop': {'offset_X': 130, 'offset_Y': 92, 'height': 400, 'width': 704},
         'rotate': 0,
         'exposure': 1500, # in microseconds
         'gain': 30, # in db
         'offset': {'x': 366, 'y': 300},
         'output_dir': 'C:\\Data folder'}

subs = ['MAP1', 'test2', 'test3']  # optional, can manually enter subject for each session.
details = {'cams': 2,
           '0': cam_0,
           '1': cam_1,
           'subjects': subs}

with open(out, 'w') as handle:
    json.dump(details, handle)
