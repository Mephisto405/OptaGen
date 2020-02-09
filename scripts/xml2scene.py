import argparse
import re
import os

parser = argparse.ArgumentParser(description='convert a xml file to a scene file.')
parser.add_argument('--src', type=str, required=True, help='source file')
parser.add_argument('--dst', type=str, required=True, help='destination file')

args = parser.parse_args()
assert args.src[-4:] == '.xml', 'error: SRC should be a xml file'

if os.path.isdir(args.dst):
    args.dst = os.path.join(args.dst, args.src.split('/')[-1][:-4] + '.scene')
elif os.path.isfile(args.dst) and args.dst[-6:] == '.scene':
    pass
else:
    assert True, 'error: DST should be a scene file or a valid directory'

writefile = open(args.dst, 'w+')

def getValue(line, key):
    if key not in line:
        return ""
    start_i = line.find('"', line.find(key))
    end_i = line.find('"', start_i+1)
    return line[start_i+1:end_i]


with open(args.src, newline='') as lines:
    lines = list(lines)
    i = 0
    while i < len(lines):
        if '<integrator' in lines[i]:
            writefile.write('properties\n')
            writefile.write('{\n')
            while '</sensor>' not in lines[i]:
                if 'maxDepth' in lines[i]:
                    writefile.write('\tmax_depth {}\n'.format(getValue(lines[i], 'value')))
                if 'width' in lines[i]:
                    writefile.write('\twidth {}\n'.format(getValue(lines[i], 'value')))
                if 'height' in lines[i]:
                    writefile.write('\theight {}\n'.format(getValue(lines[i], 'value')))
                if 'fov' in lines[i]:
                    writefile.write('\tfov {}\n'.format(getValue(lines[i], 'value')))
                if '<transform' in lines[i]:
                    while '<matrix' not in lines[i]:
                        i += 1
                    mat = getValue(lines[i], 'value').split(" ")
                    mat = [float(i) for i in mat]
                    eye = [mat[3], mat[7], mat[11]]
                    at = [mat[3] + mat[2], mat[7] + mat[6], mat[11] + mat[10]] # eye + z
                    up = [mat[1], mat[5], mat[9]] # y
                    writefile.write('\tposition {:.6f} {:.6f} {:.6f}\n'.format(*eye))
                    writefile.write('\tlook_at {:.6f} {:.6f} {:.6f}\n'.format(*at))
                    writefile.write('\tup {:.6f} {:.6f} {:.6f}\n'.format(*up))
                i += 1
            writefile.write('}\n\n')
        
        if '<bsdf' in lines[i] and 'id' in lines[i]:
            writefile.write('material ')
            while '</bsdf' not in lines[i]:
                if 'type' not in lines[i]:
                    i += 1
                    continue
                type = getValue(lines[i], 'type')

                if 'id=' in lines[i]:
                    id = getValue(lines[i], 'id=')
                    writefile.write('{}\n'.format(id))
                    writefile.write('{\n')
                    
                if type == 'diffuse':
                    while '<rgb' not in lines[i] and '<texture' not in lines[i]:
                        i += 1
                    if '<rgb' in lines[i]:
                        rgb = getValue(lines[i], 'value').split(', ')
                        rgb = [float(c) for c in rgb]
                        writefile.write('\tcolor {:.6f} {:.6f} {:.6f}\n'.format(*rgb))
                    elif '<texture' in lines[i]:
                        while 'filename' not in lines[i]:
                            i += 1
                        path = getValue(lines[i], 'value')
                        writefile.write('\talbedoTex {}\n'.format(path))

                elif type == 'plastic':
                    i += 1
                    param_dict = {}
                    while '</bsdf' not in lines[i]:
                        name = getValue(lines[i], 'name')
                        value = getValue(lines[i], 'value')
                        if name != '' and value != '':
                            param_dict[name] = value
                        i += 1
                    
                    params = param_dict.keys()
                    if 'diffuseReflectance' in params:
                        rgb = param_dict['diffuseReflectance'].split(', ')
                        rgb = [float(c) for c in rgb]
                        writefile.write('\tcolor {:.6f} {:.6f} {:.6f}\n'.format(*rgb))
                    writefile.write('\troughness 0.0\n')
                    i -= 1

                elif type == 'roughplastic':
                    while '</bsdf' not in lines[i]:
                        if 'alpha' in lines[i]:
                            writefile.write('\troughness {}\n'.format(getValue(lines[i], 'value')))
                        elif '<texture' in lines[i]:
                            while 'filename' not in lines[i]:
                                i += 1
                            path = getValue(lines[i], 'value')
                            writefile.write('\talbedoTex {}\n'.format(path))
                        elif 'diffuseReflectance' in lines[i]:
                            rgb = [float(c) for c in getValue(lines[i], 'value').split(', ')]
                            writefile.write('\tcolor {:.6f} {:.6f} {:.6f}\n'.format(*rgb))
                        i += 1

                elif type == 'conductor':
                    while '</bsdf' not in lines[i]:
                        i += 1

                    writefile.write('\tcolor 1.0 1.0 1.0\n')
                    writefile.write('\troughness 0.0\n')
                    writefile.write('\tmetallic 1.0\n')

                elif type == 'roughconductor':
                    while '</bsdf' not in lines[i]:
                        if 'alpha' in lines[i]:
                            writefile.write('\troughness {}\n'.format(getValue(lines[i], 'value')))
                        if 'specularReflectance' in lines[i]:
                            rgb = getValue(lines[i], 'value').split(', ')
                            rgb = [float(c) for c in rgb]
                            writefile.write('\tcolor {:.6f} {:.6f} {:.6f}\n'.format(*rgb))
                        i += 1
                    writefile.write('\tmetallic 1.0\n')

                elif type == 'dielectric':
                    writefile.write('\tbrdf GLASS\n')
                    i += 1
                    param_dict = {}
                    while '</bsdf' not in lines[i]:
                        name = getValue(lines[i], 'name')
                        value = getValue(lines[i], 'value')
                        if name != '' and value != '':
                            param_dict[name] = value
                        i += 1
                    
                    params = param_dict.keys()
                    writefile.write('\tintIOR {:.6f}\n'
                        .format(float(param_dict['intIOR']) if 'intIOR' in params else '1.5046'))
                    writefile.write('\textIOR {:.6f}\n'
                        .format(float(param_dict['extIOR']) if 'extIOR' in params else '1.000277'))
                    if 'specularReflectance' in params:
                        rgb = param_dict['specularReflectance'].split(', ')
                        rgb = [float(c) for c in rgb]
                        writefile.write('\tcolor {:.6f} {:.6f} {:.6f}\n'.format(*rgb))
                    elif 'specularTransmittance' in params:
                        rgb = param_dict['specularTransmittance'].split(', ')
                        rgb = [float(c) for c in rgb]
                        writefile.write('\tcolor {:.6f} {:.6f} {:.6f}\n'.format(*rgb))
                    else:
                        writefile.write('\tcolor 1.0 1.0 1.0\n')
                    i -= 1

                elif type == 'roughdielectric':
                    writefile.write('\tbrdf ROUGHDIELECTRIC\n')
                    i += 1
                    param_dict = {}
                    while '</bsdf' not in lines[i]:
                        name = getValue(lines[i], 'name')
                        value = getValue(lines[i], 'value') # texture는 value가 아니라 type으로 잡아야 함
                        if name != '' and value != '':
                            param_dict[name] = value
                        i += 1
                    
                    params = param_dict.keys()
                    if 'distribution' in params:
                        if param_dict['distribution'] == 'as':
                            raise NotImplementedError('!roughdielectric: distribution as')
                        else:
                            writefile.write('\tdist {}\n'.format(param_dict['distribution']))
                    else:
                        writefile.write('\tdist beckmann\n')
                    if 'alpha' in params:
                        writefile.write('\troughness {}\n'.format(param_dict['alpha']))
                    else:
                        writefile.write('\troughness {}\n'.format(1.0))
                    writefile.write('\tintIOR {:.6f}\n'
                        .format(float(param_dict['intIOR']) if 'intIOR' in params else '1.5046'))
                    writefile.write('\textIOR {:.6f}\n'
                        .format(float(param_dict['extIOR']) if 'extIOR' in params else '1.000277'))
                    if 'specularReflectance' in params:
                        rgb = param_dict['specularReflectance'].split(', ')
                        rgb = [float(c) for c in rgb]
                        writefile.write('\tcolor {:.6f} {:.6f} {:.6f}\n'.format(*rgb))
                    elif 'specularTransmittance' in params:
                        rgb = param_dict['specularTransmittance'].split(', ')
                        rgb = [float(c) for c in rgb]
                        writefile.write('\tcolor {:.6f} {:.6f} {:.6f}\n'.format(*rgb))
                    else:
                        writefile.write('\tcolor 1.0 1.0 1.0\n')
                    i -= 1

                i += 1
            writefile.write('}\n\n')
        
        if '<shape' in lines[i]:
            type = getValue(lines[i], 'type')

            j = i
            isEmitter = False
            emit_type = ""
            while '</shape' not in lines[j]:
                if '<emitter' in lines[j]:
                    isEmitter = True
                    emit_type = getValue(lines[j], 'type')
                j += 1
            
            if not isEmitter:
                writefile.write('mesh\n')
                writefile.write('{\n')
                if type == 'rectangle':
                    writefile.write('\tfile models/Rectangle.obj\n')
                elif type != 'obj':
                    NotImplementedError('Shape {} is not implemented yet.'.format(type))

                while '</shape' not in lines[i]:
                    if 'filename' in lines[i]:
                        path = getValue(lines[i], 'value')
                        writefile.write('\tfile {}\n'.format(path))
                    if '<transform' in lines[i]:
                        while '<matrix' not in lines[i]:
                            i += 1
                        mat = getValue(lines[i], 'value')
                        if mat != '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1':
                            writefile.write('\ttransform {}\n'.format(mat)) # transform
                        while '</transform' not in lines[i]:
                            i += 1
                    if '<ref' in lines[i]:
                        id = getValue(lines[i], 'id')
                        writefile.write('\tmaterial {}\n'.format(id))
                    i += 1
            elif isEmitter and type == 'rectangle' and emit_type == 'area':
                writefile.write('light\n')
                writefile.write('{\n')
                while '</shape' not in lines[i]:
                    if '<transform' in lines[i]:
                        while '<matrix' not in lines[i]:
                            i += 1
                        mat = getValue(lines[i], 'value').split(" ")
                        mat = [float(m) for m in mat]
                        pos = [- mat[0] - mat[1] + mat[3], 
                               - mat[4] - mat[5] + mat[7], 
                               - mat[8] - mat[9] + mat[11]]
                        v1 = [pos[0] + 2 * mat[0], pos[1] + 2 * mat[4], pos[2] + 2 * mat[8]]
                        v2 = [pos[0] + 2 * mat[1], pos[1] + 2 * mat[5], pos[2] + 2 * mat[9]]
                        writefile.write('\tposition {:.6f} {:.6f} {:.6f}\n'.format(*pos))
                        writefile.write('\tv1 {:.6f} {:.6f} {:.6f}\n'.format(*v1))
                        writefile.write('\tv2 {:.6f} {:.6f} {:.6f}\n'.format(*v2))
                        while '</transform' not in lines[i]:
                            i += 1
                    if '<emitter' in lines[i]:
                        while '<rgb' not in lines[i]:
                            i += 1
                        rgb = getValue(lines[i], 'value').split(', ')
                        rgb = [float(c) for c in rgb]
                        writefile.write('\temission {:.6f} {:.6f} {:.6f}\n'.format(*rgb))
                    i += 1
                writefile.write('\ttype Quad\n')
            elif isEmitter and type == 'sphere' and emit_type == 'area':
                writefile.write('light\n')
                writefile.write('{\n')
                i += 1
                param_dict = {}
                while '</shape' not in lines[i]:
                    name = getValue(lines[i], 'name')
                    if name == '':
                        i += 1
                        continue
                    if 'point' in lines[i]:
                        value = "{} {} {}".format(getValue(lines[i], 'x'), getValue(lines[i], 'y'), getValue(lines[i], 'z'))
                    else:
                        value = getValue(lines[i], 'value') # texture는 value가 아니라 type으로 잡아야 함
                    if name != '' and value != '':
                        param_dict[name] = value
                    i += 1

                params = param_dict.keys()
                if 'radius' in params:
                    rad = param_dict['radius']
                    writefile.write('\tradius {}\n'.format(rad))
                else:
                    writefile.write('\tradius 1.0\n')
                if 'center' in params:
                    writefile.write('\tposition {}\n'.format(param_dict['center']))
                else:
                    writefile.write('\tposition 0.0 0.0 0.0\n')
                if 'radiance' in params:
                    rgb = param_dict['radiance'].split(', ')
                    rgb = [float(c) for c in rgb]
                    writefile.write('\temission {:.6f} {:.6f} {:.6f}\n'.format(*rgb))
                else:
                    raise NotImplementedError('Emitter: Radiance is not specified')
                i -= 1
                writefile.write('\ttype Sphere\n')

            else: # isEmitter and type != 'rectangle'
                print(isEmitter)
                print(type)
                print(emit_type)
                raise NotImplementedError('Non-rectangle type emitter is not implemented yet')
            writefile.write('}\n\n')
        if '<emitter' in lines[i]:
            raise NotImplementedError('Env map is not implemented yet')
        
        i += 1