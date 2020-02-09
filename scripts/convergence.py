import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

mit_imgs = []
for i in [4,8,32,128]:
    img = plt.imread('./coffee_mitsuba/{}spp.png'.format(i))
    mit_imgs.append(img)

opx_imgs = []
for i in [4,8,32,128]:
    img = plt.imread('./coffee_optix/{}spp.png'.format(i))
    opx_imgs.append(img)

mit_ref = plt.imread('./coffee_mitsuba/reference.png')
opx_ref = plt.imread('./coffee_optix/reference.png')

def PSNR(input, target):
    mse = (input - target)**2
    mse = mse.mean()
    psnr = 4 * np.log2(255) - 2 * np.log2(mse)
    return psnr

y_mit = []
for img in mit_imgs:
    y_mit.append(PSNR(img, mit_ref))

y_opx = []
for img in opx_imgs:
    y_opx.append(PSNR(img, opx_ref))

x = np.log2(np.array([4,8,32,128]))
y_mit = np.array(y_mit)
y_opx = np.array(y_opx)

s_mit, _, _, _, _ = stats.linregress(x, y_mit)
s_opx, _, _, _, _ = stats.linregress(x, y_opx)

plt.plot(x, y_mit, 'b-')
plt.plot(x, y_opx, 'r-')
plt.show()