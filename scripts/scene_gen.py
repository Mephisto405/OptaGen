import os
import subprocess
import os.path as path

scene_home = 'C://Users//Dorian//data_scenes//optagen'
optagen = 'C://Users//Dorian//cuda_datagen//Optix-PathTracer' \
          + '//build//bin//Release//optixPathTracer.exe'

def randomize_cam_params(data):
    pass



do_loop = True
for root, dirs, files in os.walk(scene_home, topdown=True):
    if not do_loop:
        break
    for name in files:
        if '.scene' in name:
            output = path.join(root, 'test.png')
            tmp_scene = path.join(root, 'tmp.scene')
            spp = '16'

            with open(path.join(root, name)) as f, open(tmp_scene, 'w') as o:
                data = f.read()
                #data = data.replace('fov 55', 'fov 1')
                data = data.replace('width 1024', 'width 128')
                data = data.replace('height 1024', 'height 128')
                o.write(data)

            cmd = [optagen, '-f', output, '-n', '-s', tmp_scene, '-p', spp]
            cmd_out = subprocess.check_output(cmd)
            os.remove(tmp_scene)
            do_loop = False