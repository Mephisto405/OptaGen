import os
import random
import argparse
import numpy as np 
import os.path as path
from subprocess import check_output, STDOUT, CalledProcessError

recipe1 = {
    'bathroom': 32000,
    'car': 32000,
    'car2': 32000,
}

recipe2 = {
    'chair': 16000,
    'gharbi': 32000,
    'tableware': 16000,
    'teapot': 64000,
    'bathroom': 32000,
    'car': 32000,
    'car2': 32000,
}

recipe3 = {
    'path-manifold': 32000
}

recipe = recipe3

for i in range(8):
    for scene in recipe:
        print(scene)
        cmd = [
                'OptaGen.exe',
                '-M', '1',
                '-n', '1',
                '-w', '512',
                '-v', '0',
                '-s', 'C:\\Users\\dorian\\scenes\\optagen\\test\\{}\\scene.scene'.format(scene),
                '-m', str(recipe[scene]),
                '-i', 'D:\\LLPM\\test\\input\\{}_{}.npy'.format(scene, i),
                '-o', 'D:\\LLPM\\test\\gt\\{}_{}.npy'.format(scene, i),
                '--device', '1'
            ]

        if i == 0:
            cmd[cmd.index('-i')+1] = 'D:\\LLPM\\test\\input\\{}.npy'.format(scene)
            cmd[cmd.index('-o')+1] = 'D:\\LLPM\\test\\gt\\{}.npy'.format(scene)
        print(cmd)

        try:
            cmd_out = check_output(cmd, stderr=STDOUT)
        except CalledProcessError as exc:
            print('error!')
            print(exc.output)
        else:
            continue

print("[] Processing done.")