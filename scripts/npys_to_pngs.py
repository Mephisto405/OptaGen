import os
import numpy as np
import matplotlib.pyplot as plt

#mpl.rcParams['figure.dpi'] = 250

def ToneMap(c, limit=1.5):
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


"""
cnt = 0
for root, dirs, files in os.walk("D:\\LLPM\\train\\gt", topdown=False):
    for name in files:
        npy_fn = os.path.join(root, name)
        png_fn = os.path.join("D:\\LLPM\\train\\gt_imgs", name)[:-3]+"png"
        if os.path.isfile(png_fn):
            continue
        else:
            img_np = np.load(npy_fn)[:,:,:3]
            plt.imsave(os.path.join("D:\\LLPM\\train\\gt_imgs", name)[:-3]+"png", LinearToSrgb(ToneMap(img_np)))
            cnt += 1
print(cnt)
"""

cnt = 0
for root, dirs, files in os.walk("D:\\LLPM\\train\\gt", topdown=False):
    for name in files:
        npy_fn = os.path.join(root, name)
        png_fn = os.path.join("D:\\LLPM\\train\\gt_imgs", name)[:-3]+"png"
        if os.path.isfile(png_fn):
            continue
        else:
            os.remove(npy_fn)
            os.remove(os.path.join("D:\\LLPM\\train\\input", name)[:-4]+"_a.npy")
            os.remove(os.path.join("D:\\LLPM\\train\\input", name)[:-4]+"_b.npy")
            cnt += 1
print(cnt)

"""
for root, dirs, files, in os.walk("D:\\p-buffer\\train\\input", topdown=False):
    for name in files:
        png_fn = os.path.join(root, name)
        npy_fn = os.path.join("D:\\p-buffer\\train\\gt", name)[:-3]+"npy"
        if not os.path.isfile(npy_fn):
            assert False

for root, dirs, files, in os.walk("D:\\p-buffer\\train\\gt", topdown=False):
    for name in files:
        png_fn = os.path.join(root, name)
        npy_fn = os.path.join("D:\\p-buffer\\train\\input", name)[:-3]+"npy"
        if not os.path.isfile(npy_fn):
            assert False
"""