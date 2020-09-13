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
    return np.clip(c, 0.0, 1.0) ** kInvGamma

def main():
    parser = argparse.ArgumentParser(description='Path descriptor visualizer.')
    parser.add_argument('--npy', type=str, required=True, help='Path descriptor (.npy).')
    args = parser.parse_args()

    filename = args.npy
    arr = np.load(filename)

    if len(arr.shape) == 4:
        MAX_DEPTH = 5


        """ SBMC features """
        subpixel_x = arr[:,:,:,0]
        subpixel_y = arr[:,:,:,1]
        radiance = arr[:,:,:,2:5]
        radiance_diffuse = arr[:,:,:,5:8]
        radiance_specular = radiance - radiance_diffuse
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


        """ KPCN features """
        albedo_at_diff = arr[:,:,:,24+(MAX_DEPTH+1)*7:27+(MAX_DEPTH+1)*7]
        normal_at_diff = arr[:,:,:,27+(MAX_DEPTH+1)*7:30+(MAX_DEPTH+1)*7]
        depth_at_diff = arr[:,:,:,30+(MAX_DEPTH+1)*7]


        """ LLPM features """
        path_weight = arr[:,:,:,31+(MAX_DEPTH+1)*7]
        radiance_wo_weight = arr[:,:,:,32+(MAX_DEPTH+1)*7:35+(MAX_DEPTH+1)*7]
        light_intensity = arr[:,:,:,35+(MAX_DEPTH+1)*7:38+(MAX_DEPTH+1)*7]

        throughputs = arr[:,:,:,38+(MAX_DEPTH+1)*7:38+(MAX_DEPTH+1)*10]
        roughnesses = arr[:,:,:,38+(MAX_DEPTH+1)*10:38+(MAX_DEPTH+1)*11]

        print(np.max(roughnesses))
        print(np.min(roughnesses))

        """ SBMC features 
        plt.imshow(np.mean(subpixel_x, 2), cmap='gray')
        plt.show()
        plt.imshow(np.mean(subpixel_y, 2), cmap='gray')
        plt.show()
        plt.imshow(LinearToSrgb(ToneMap(np.mean(radiance, 2), 1.5)))
        plt.show()
        plt.imshow(LinearToSrgb(ToneMap(np.mean(radiance_diffuse, 2), 1.5)))
        plt.show()
        plt.imshow(LinearToSrgb(ToneMap(np.mean(radiance_diffuse, 2) / np.mean(albedo_at_diff + 0.00316, 2), 1.5)))
        plt.show()
        plt.imshow(LinearToSrgb(ToneMap(np.mean(radiance_specular, 2), 1.5)))
        plt.show()
        plt.imshow(np.mean(albedo_at_first, 2))
        plt.show()
        plt.imshow(np.mean(albedo, 2))
        plt.show()
        plt.imshow(np.mean(normal_at_first * 0.5 + 0.5, 2))
        plt.show()
        plt.imshow(np.mean(normal * 0.5 + 0.5, 2))
        plt.show()"""
        plt.imshow(np.mean(depth_at_first, 2), cmap='binary', vmax = np.max(depth), vmin = np.min(depth))
        plt.show()
        """plt.imshow(np.mean(depth, 2), cmap='binary', vmax = np.max(depth), vmin = np.min(depth))
        plt.show()
        plt.imshow(np.mean(visibility, 2), cmap='gray')
        plt.show()
        plt.imshow(np.mean(hasHit, 2), cmap='gray')
        plt.show()

        for i in range(MAX_DEPTH+1):
            plt.subplot(2,3,i+1)
            plt.title(str(i+1))
            plt.imshow((np.mean(probabilities[:,:,:,4*i], 2) / (1 + np.mean(probabilities[:,:,:,4*i], 2) / 1.5)) ** 0.45, cmap='gray')
        plt.show()
        for i in range(MAX_DEPTH+1):
            plt.subplot(2,3,i+1)
            plt.title(str(i+1))
            plt.imshow((np.mean(probabilities[:,:,:,4*i+1], 2) / (1 + np.mean(probabilities[:,:,:,4*i+1], 2) / 1.5)) ** 0.45, cmap='gray')
        plt.show()
        for i in range(MAX_DEPTH+1):
            plt.subplot(2,3,i+1)
            plt.title(str(i+1))
            plt.imshow((np.mean(probabilities[:,:,:,4*i+2], 2) / (1 + np.mean(probabilities[:,:,:,4*i+2], 2) / 1.5)) ** 0.45, cmap='gray')
        plt.show()
        for i in range(MAX_DEPTH+1):
            plt.subplot(2,3,i+1)
            plt.title(str(i+1))
            plt.imshow((np.mean(probabilities[:,:,:,4*i+3], 2) / (1 + np.mean(probabilities[:,:,:,4*i+3], 2) / 1.5)) ** 0.45, cmap='gray')
        plt.show()
        for i in range(MAX_DEPTH+1):
            plt.subplot(2,3,i+1)
            plt.title(str(i+1))
            plt.imshow(np.mean(light_directions[:,:,:,2*i], 2), cmap='binary')
        plt.show()
        for i in range(MAX_DEPTH+1):
            plt.subplot(2,3,i+1)
            plt.title(str(i+1))
            plt.imshow(np.mean(light_directions[:,:,:,2*i+1], 2), cmap='binary')
        plt.show()
        plt.imshow(np.mean(bounce_types[:,:,:,0], 2), cmap='gray')
        plt.show()"""


        """ KPCN features 
        plt.imshow(np.mean(albedo_at_diff, 2))
        plt.show()
        plt.imshow(np.mean(normal_at_diff * 0.5 + 0.5, 2))
        plt.show()
        plt.imshow(np.mean(depth_at_diff, 2), cmap='binary', vmax = np.max(depth), vmin = np.min(depth))
        plt.show()"""

        """ LLPM features 
        plt.imshow((np.mean(path_weight, 2) / (1 + np.mean(path_weight, 2) / 1.5)) ** 0.45, cmap='gray')
        plt.show()
        radiance_wo_weight[:,:,:,0] /= path_weight + 0.00316
        radiance_wo_weight[:,:,:,1] /= path_weight + 0.00316
        radiance_wo_weight[:,:,:,2] /= path_weight + 0.00316
        plt.imshow(LinearToSrgb(ToneMap(np.mean(radiance_wo_weight, 2), 1.5)))
        plt.show()
        plt.imshow(LinearToSrgb(ToneMap(np.mean(light_intensity, 2), 1.5)))
        plt.show()

        for i in range(MAX_DEPTH+1):
            plt.subplot(2,3,i+1)
            plt.title(str(i+1))
            plt.imshow(LinearToSrgb(ToneMap(np.mean(throughputs[:,:,:,3*i:3*i+3], 2), 1.5)))
        plt.show()
        for i in range(MAX_DEPTH+1):
            plt.subplot(2,3,i+1)
            plt.title(str(i+1))
            plt.imshow((np.mean(roughnesses[:,:,:,i], 2) / (1 + np.mean(roughnesses[:,:,:,i], 2) / 1.5)) ** 0.45, cmap='binary')
        plt.show()"""


if __name__ == '__main__':
    main()