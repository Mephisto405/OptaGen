import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#mpl.rcParams['figure.dpi'] = 250

def ToneMap(c, limit):
    # c: (W, H, C=3)
    luminance = 0.3 * c[:,:,0] + 0.6 * c[:,:,1] + 0.1 * c[:,:,2]
    col = c[:,:,:]
    col[:,:,0] /=  (1.0 + luminance / limit)
    col[:,:,1] /=  (1.0 + luminance / limit)
    col[:,:,2] /=  (1.0 + luminance / limit)
    return col

def LinearToSrgb(c):
    # c: (W, H, C=3)
    kInvGamma = 1.0 / 2.2
    return np.clip(c ** kInvGamma, 0.0, 1.0)

def main():
    parser = argparse.ArgumentParser(description='Path descriptor visualizer.')
    parser.add_argument('--npy', type=str, required=True, help='Path descriptor (.npy).')
    args = parser.parse_args()

    filename = args.npy
    arr = np.load(filename)

    if len(arr.shape) == 4:
        print("Path descriptor")

        MAX_DEPTH = 5

        subpixel_x = arr[:,:,:,0]
        subpixel_y = arr[:,:,:,1]
        radiance = arr[:,:,:,2:5]
        radiance_diffuse = arr[:,:,:,5:8]
        albedo_at_first = arr[:,:,:,8:11]
        albedo = arr[:,:,:,11:14]
        normal_at_first = arr[:,:,:,14:17]
        normal = arr[:,:,:,17:20]
        depth_at_first = arr[:,:,:,20]
        depth = arr[:,:,:,21]
        visibility = arr[:,:,:,22]
        hasHit = arr[:,:,:,23]

        probabilities = arr[:,:,:,24:24+(MAX_DEPTH+1)*4]
        light_directions = arr[:,:,:,24+(MAX_DEPTH+1)*4:24+(MAX_DEPTH+1)*6]
        bounce_types = arr[:,:,:,24+(MAX_DEPTH+1)*6:24+(MAX_DEPTH+1)*7]

        path_weight = arr[:,:,:,24+(MAX_DEPTH+1)*7]
        radiance_wo_weight = arr[:,:,:,25+(MAX_DEPTH+1)*7:27+(MAX_DEPTH+1)*7]
        light_intensity = arr[:,:,:,27+(MAX_DEPTH+1)*7:31+(MAX_DEPTH+1)*7]

        throughputs = arr[:,:,:,31+(MAX_DEPTH+1)*7:31+(MAX_DEPTH+1)*10]
        roughnesses = arr[:,:,:,31+(MAX_DEPTH+1)*10:31+(MAX_DEPTH+1)*11]

        plt.imshow(LinearToSrgb(ToneMap(np.mean(radiance, 2), 1.5)))
        plt.show()
        plt.imshow(np.mean(albedo_at_first, 2))
        plt.show()
        plt.imshow(np.mean(normal_at_first, 2))
        plt.show()
        plt.imshow(np.mean(hasHit, 2), cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()