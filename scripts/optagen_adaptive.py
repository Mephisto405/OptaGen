import os
import random
import argparse
import numpy as np 
import os.path as path
from subprocess import check_output, STDOUT, CalledProcessError

recipe = {
    'living-room-3': (9, 32000),
}

for scene in recipe:
    print(scene)
    cmd = [
            'C:\\Users\\Dorian\\cuda_datagen\\OptaGen\\build\\bin\\Release\\OptaGen.exe',
            '-M', '2',
            '-s', 'C:\\Users\\Dorian\\scenes\\optagen\\train\\{}\\scene.scene'.format(scene),
            '-d', 'C:\\Users\\Dorian\\scenes\\optagen\\hdri',
            '-i', 'D:\\LLPM\\train\\input\\{}.npy'.format(scene),
            '-o', 'D:\\LLPM\\train\\gt\\{}.npy'.format(scene),
            '-n', str(recipe[scene][0]),
            #TODO(iycho) should change this part
            '-c', '22',
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