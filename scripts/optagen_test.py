import os
import random
import argparse
import numpy as np 
import os.path as path
from subprocess import check_output, STDOUT, CalledProcessError

recipe1 = {
    #'bathroom': 32000,
    #'car': 32000,
    'car2': 32000,
}

recipe2 = {
    #'chair': 16000,
    #'gharbi': 32000,
    #'tableware': 16000,
    'teapot': 64000
}

recipe = recipe1

for scene in recipe:
    print(scene)
    cmd = [
            'C:\\Users\\dorian\\OptaGen\\build\\bin\\Release\\OptaGen.exe',
            '-M', '2',
            '-n', '1',
            '-w', '512',
            '-v', '0',
            '-p', '4',
            '-s', 'C:\\Users\\dorian\\scenes\\optagen\\test\\{}\\scene.scene'.format(scene),
            '-m', str(recipe[scene]),
            '-i', 'D:\\p-buffer\\test\\input\\{}.npy'.format(scene),
            '-o', 'D:\\p-buffer\\test\\gt\\{}.npy'.format(scene),
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