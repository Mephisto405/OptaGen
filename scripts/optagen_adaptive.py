import os
import random
import argparse
import numpy as np 
import os.path as path
from subprocess import check_output, STDOUT, CalledProcessError

recipe = {
    'bathroom-2': (4, 32000),
    'dining-room-2': (5, 32000),
    'hyperion': (3, 32000),
    'lamp': (6, 32000),
    'living-room': (4, 32000),
    'hyperion': (3, 32000),
    'living-room-2': (2, 32000),
    'living-room-3': (9, 32000),
    'material-testball': (6, 32000),
    'spaceship': (8, 32000),
    'staircase': (14, 16000),
    'staircase-2': (14, 16000),
}

for scene in recipe:
    print(scene)
    cmd = [
            'C:\\Users\\Dorian\\cuda_datagen\\OptaGen\\build\\bin\\Release\\OptaGen.exe',
            '-M', '2',
            '-s', 'C:\\Users\\Dorian\\scenes\\optagen\\train\\{}\\scene.scene'.format(scene),
            '-i', 'D:\\LLPM\\train\\input\\{}.npy'.format(scene),
            '-o', 'D:\\LLPM\\train\\gt\\{}.npy'.format(scene),
            '-n', str(recipe[scene][0]),
            #TODO(iycho) should change this part
            '-c', '14',
            '-m', str(recipe[scene][1]),
            '-w', '1280',
            '-v', '0',
            '--device', '0'
        ]

    try:
        cmd_out = check_output(cmd, stderr=STDOUT)
    except CalledProcessError as exc:
        print('error!')
        print(exc.output)
    else:
        continue

print("[] Processing done.")