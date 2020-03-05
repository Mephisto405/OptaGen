import argparse
import numpy as np
import matplotlib.pyplot as plt

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

parser = argparse.ArgumentParser(description='MBPF visualizer.')
parser.add_argument('--npy', type=str, required=True, help='multi-bounce path feature (.npy).')
args = parser.parse_args()

filename = args.npy
arr = np.load(filename)
arr = np.flip(arr, 0)

if len(arr.shape) == 4:
    rad = [arr[:,:,:,0:3], arr[:,:,:,3:6], arr[:,:,:,6:9], arr[:,:,:,9:12], arr[:,:,:,12:15], arr[:,:,:,15:18]]
    alb = [arr[:,:,:,18:21], arr[:,:,:,21:24], arr[:,:,:,24:27], arr[:,:,:,27:30], arr[:,:,:,30:33], arr[:,:,:,33:36]]
    nor = [arr[:,:,:,36:39], arr[:,:,:,39:42], arr[:,:,:,42:45], arr[:,:,:,45:48], arr[:,:,:,48:51], arr[:,:,:,51:54]]

    plt.subplot(3,6,1)
    plt.imshow(LinearToSrgb(ToneMap(np.mean(rad[0], 2), 1.5)))
    plt.subplot(3,6,2)
    plt.imshow(LinearToSrgb(ToneMap(np.mean(rad[1], 2), 1.5)))
    plt.subplot(3,6,3)
    plt.imshow(LinearToSrgb(ToneMap(np.mean(rad[2], 2), 1.5)))
    plt.subplot(3,6,4)
    plt.imshow(LinearToSrgb(ToneMap(np.mean(rad[3], 2), 1.5)))
    plt.subplot(3,6,5)
    plt.imshow(LinearToSrgb(ToneMap(np.mean(rad[4], 2), 1.5)))
    plt.subplot(3,6,6)
    plt.imshow(LinearToSrgb(ToneMap(np.mean(rad[5], 2), 1.5)))

    plt.subplot(3,6,7)
    plt.imshow(np.mean(alb[0], 2))
    plt.subplot(3,6,8)
    plt.imshow(np.mean(alb[1], 2))
    plt.subplot(3,6,9)
    plt.imshow(np.mean(alb[2], 2))
    plt.subplot(3,6,10)
    plt.imshow(np.mean(alb[3], 2))
    plt.subplot(3,6,11)
    plt.imshow(np.mean(alb[4], 2))
    plt.subplot(3,6,12)
    plt.imshow(np.mean(alb[5], 2))

    plt.subplot(3,6,13)
    plt.imshow(np.mean(nor[0], 2))
    plt.subplot(3,6,14)
    plt.imshow(np.mean(nor[1], 2))
    plt.subplot(3,6,15)
    plt.imshow(np.mean(nor[2], 2))
    plt.subplot(3,6,16)
    plt.imshow(np.mean(nor[3], 2))
    plt.subplot(3,6,17)
    plt.imshow(np.mean(nor[4], 2))
    plt.subplot(3,6,18)
    plt.imshow(np.mean(nor[5], 2))

    plt.show()
elif len(arr.shape) == 3:
    plt.imshow(LinearToSrgb(ToneMap(arr, 1.5))); plt.show()
    