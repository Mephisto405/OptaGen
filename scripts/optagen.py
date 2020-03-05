import os
import random
import argparse
import numpy as np 
import os.path as path
from subprocess import check_output, STDOUT, CalledProcessError

##
# Input
parser = argparse.ArgumentParser(description='generate a training dataset.')
parser.add_argument('--exe', type=str, required=True, help='OptaGen renderer.')
parser.add_argument('--mode', type=int, required=True, help='0: reference-only, 1: feature-only, 2: both')
parser.add_argument('--root', type=str, required=True, help='directory which contains all OptaGen scene files to use.')
parser.add_argument('--hdr', type=str, required=True, help='directory containing all hdr environmental map files to use.')
parser.add_argument('--save', type=str, required=True, help='parent directory to save patches and reference images.')
parser.add_argument('--num', type=int, required=False, default=258, help='total number of patches to generate.')
parser.add_argument('--spp', type=int, required=False, default=4, help='sample per pixel.')
parser.add_argument('--mspp', type=int, required=False, default=1024, help='maximum number of sample per pixel to render the reference image.')
parser.add_argument('--roc', type=float, required=False, default=0.9, help='if the `rate of change` of relMSE is higher than this value, stop rendering the reference image.')
args = parser.parse_args()

assert os.path.isfile(args.exe), 'EXE is not a valid executable file.'
assert (args.mode in [0, 1, 2]), '0: reference-only, 1: feature-only, 2: both'
assert os.path.isdir(args.root), 'ROOT does not exist.'
assert os.path.isdir(args.hdr), 'HDR does not exist.' 
if not os.path.isdir(args.save):
    os.mkdir(args.save)
input_dir = path.join(args.save, 'input')
if not os.path.isdir(input_dir):
    os.mkdir(input_dir)
ref_dir = path.join(args.save, 'ref')
if not os.path.isdir(ref_dir):
    os.mkdir(ref_dir)
assert args.num >= 10, 'NUM < 10.'
assert args.spp >= 1 and args.spp <= 32, 'SPP < 1 or SPP > 32. CUDA memory error might occur.'
assert args.mspp >= 32 and args.mspp <= 1000000, 'MSPP < 1000 or MSPP > 1000000.'
assert args.roc > 0.0 and args.roc < 1.0, 'ROC <= 0.0 or ROC >= 1.0.'
#.\build\bin\Release\OptaGen.exe -m 0 -s C:\Users\Dorian\data_scenes\optagen\bathroom\scene.scene -d C:\Users\Dorian\data_scenes\optagen\HDRS -i C:\Users\Dorian\data_scenes\optagen\bathroom\scene.
##
# Aux. Configurations
scenes = []
for root, dirs, files in os.walk(args.root, topdown=True):
    for name in files:
        if '.scene' in name:
            scenes.append(path.join(root, name))
patches_per_scenes = (args.num + len(scenes) - 1) // len(scenes)
print('[] Number of scenes: {}'.format(len(scenes)))
print('[] Patches per scene: {}'.format(patches_per_scenes))
print('[] Expected number of patches to generate: {}'.format(patches_per_scenes * len(scenes)))

hdrs = []
for root, dirs, files in os.walk(args.hdr, topdown=True):
    for name in files:
        if '.hdr' in name:
            hdrs.append(path.join(root, name))
patches_per_hdrs = 100
print('[] Number of HDRIs: {}'.format(len(hdrs))); print('')

##
# Rendering
print('[] Rendering start...')
for scene in scenes:
    try:
        arr = np.load(path.join(input_dir, scene.split('\\')[-2] + '.npy'))
        if arr.shape != (1280, 1280, 4, 54):
            raise ValueError()
    except ValueError:
        cmd = [
            args.exe,
            '-M', str(args.mode),
            '-s', scene,
            '-d', args.hdr,
            '-i', path.join(input_dir, scene.split('\\')[-2] + '.npy'),
            '-o', path.join(ref_dir, scene.split('\\')[-2] + '.npy'),
            '-n', str(patches_per_scenes),
            '-p', str(args.spp),
            '-m', str(args.mspp),
            '-r', str(args.roc),
            '-w', "1280",
            '-v', "0"
        ]

        try:
            cmd_out = check_output(cmd, stderr=STDOUT)
        except CalledProcessError as exc:
            print(exc.output)
        else:
            assert 0
    else:
        continue