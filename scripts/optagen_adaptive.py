import os
import random
import argparse
import numpy as np 
import os.path as path
from subprocess import check_output, STDOUT, CalledProcessError

recipe1 = {
    #'bathroom': (1, 64*1024),
    #'car2': (1, 64*1024),
    'teapot': (1, 128*1024),
}

recipe2 = {
    'car': (1, 128*1024),
    #'chair': (1, 16*1024),
    #'gharbi': (1, 16*1024),
    #'tableware': (1, 64*1024),
}

recipe = recipe1

for scene in recipe:
    print(scene)
    cmd = [
            'OptaGen.exe',
            '-M', '2',
            '-s', 'C:\\Users\\dorian\\scenes\\optagen\\test\\{}\\scene.scene'.format(scene),
            '-d', 'C:\\Users\\dorian\\scenes\\optagen\\hdri',
            '-i', 'D:\\LLPM\\test\\input\\{}.npy'.format(scene),
            '-o', 'D:\\LLPM\\test\\gt\\{}.npy'.format(scene),
            '-n', str(recipe[scene][0]),
            #TODO(iycho) should change this part
            #'-c', '22',
            '-m', str(recipe[scene][1]),
            '-w', '1024',
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