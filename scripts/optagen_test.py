import os
import random
import argparse
import numpy as np 
import os.path as path
from subprocess import check_output, STDOUT, CalledProcessError

from npys_to_pngs import merge_file

recipe1 = {
    # 'bathroom': 32000,
    # 'car': 32000,
    'car2': 32000,
    'chair': 16000
}

recipe2 = {
    'gharbi': 32000,
    'tableware': 16000,
    'teapot': 64000,
}

recipe3 = {
    'bathroom': 32000,
    'car': 32000,
    'car2': 32000,
    'chair': 16000,
    'gharbi': 32000,
    'tableware': 16000,
    'teapot': 64000,
}

recipe4 = {
    # 'bathroom': 64,
    # 'car': 32,
    # 'car2': 32,
    'chair': 64
}

recipe = recipe4

# base_spp = 2
# target_spp = 8
# for i in range(target_spp // base_spp):
#     for scene in recipe:
#         print(scene + '_' + str(i))
#         cmd = [
#             'C:\\Users\\SGVR\\Desktop\\OptaGen\\build\\bin\\Release\\OptaGen.exe',
#             # '-M', '2' if i==0 else '1',
#             '-M', '0',
#             '-n', '1',
#             '-w', '1280',
#             '-v', '0',
#             '-s', 'C:\\Users\\SGVR\\Downloads\\test\\test\\{}\\scene.scene'.format(scene),
#             # '-s', 'C:\\Users\\SGVR\\Downloads\\test_suite_2\\test_suite_2\\{}\\scene.scene'.format(scene),
#             '-m', str(recipe[scene]),
#             # '-i', 'C:\\Users\\SGVR\\Desktop\\{}.npy'.format(scene),
#             # '-o', 'C:\\Users\\SG VR\\Desktop\\{}.npy'.format(scene),
#             '-i', 'D:\\p-buf\\test\\input\\{}.npy'.format(scene + '_spp' + str(i)),
#             '-o', 'D:\\p-buf\\test\\gt\\{}.npy'.format(scene),
#             '--device', '1'
#         ]

#         # if i == 0:
#         #     cmd[cmd.index('-i')+1] = 'C:\\Users\\SGVR\\Desktop\\{}.npy'.format(scene)
#         #     cmd[cmd.index('-o')+1] = 'C:\\Users\\SGVR\\Desktop\\{}.npy'.format(scene)
#         print(cmd)

#         try:
#             cmd_out = check_output(cmd, stderr=STDOUT)
#         except CalledProcessError as exc:
#             print('error!')
#             print(exc.output)
#         else:
#             continue

# for scene in recipe:
#         print(scene)
#         cmd = [
#             'C:\\Users\\SGVR\\Desktop\\OptaGen\\build\\bin\\Release\\OptaGen.exe',
#             '-M', '2',
#             '-n', '1',
#             '-w', '1280',
#             '-v', '0',
#             '-s', 'C:\\Users\\SGVR\\Downloads\\test\\test\\{}\\scene.scene'.format(scene),
#             # '-s', 'C:\\Users\\SGVR\\Downloads\\test_suite_2\\test_suite_2\\{}\\scene.scene'.format(scene),
#             '-m', str(recipe[scene]),
#             # '-i', 'C:\\Users\\SGVR\\Desktop\\{}.npy'.format(scene),
#             # '-o', 'C:\\Users\\SG VR\\Desktop\\{}.npy'.format(scene),
#             '-i', 'D:\\p-buf\\test\\input\\{}.npy'.format(scene),
#             '-o', 'D:\\p-buf\\test\\gt\\{}.npy'.format(scene),
#             '--device', '1'
#         ]

#         # if i == 0:
#         #     cmd[cmd.index('-i')+1] = 'C:\\Users\\SGVR\\Desktop\\{}.npy'.format(scene)
#         #     cmd[cmd.index('-o')+1] = 'C:\\Users\\SGVR\\Desktop\\{}.npy'.format(scene)
#         print(cmd)

#         try:
#             cmd_out = check_output(cmd, stderr=STDOUT)
#         except CalledProcessError as exc:
#             print('error!')
#             print(exc.output)
#         else:
#             continue

# print("[] Rendering done.")
merge_file(dir="D:\\SBMC\\train")