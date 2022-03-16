from copy import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import random
# import tqdm

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

def check_dir_exists(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            print(dir, 'does not exists')


def merge_file(dir="D:\\p-buf\\test"):
    dir_input = os.path.join(dir, 'input')
    cnt = 0
    for root, dirs, files in os.walk(dir_input, topdown=False):
        for name in files:
            if not name.endswith('_a.npy'):
                continue
            
            print('merging', name[:-6])
            feat_a = np.load(os.path.join(dir_input, name))
            feat_b = np.load(os.path.join(dir_input, name)[:-6]+"_b.npy")

            feat_b = np.concatenate((feat_a, feat_b), axis=0)
            np.save(os.path.join(dir_input, name)[:-6]+".npy", feat_b)

            os.remove(os.path.join(dir_input, name))
            os.remove(os.path.join(dir_input, name)[:-6]+"_b.npy")
            cnt += 1
            print('merged', name[:-6])
    print(cnt)
    cnt = 0
    for root, dirs, files in os.walk(dir_input, topdown=False):
        for name in files:
            if not name.endswith('_spp0.npy'):
                continue

            print('merging', name[:-9])
            # TODO
            # Extend to multiple divided 
            feat_0 = np.load(os.path.join(dir_input, name))
            feat_1 = np.load(os.path.join(dir_input, name)[:-9]+"_spp1.npy")
            
            feat_1 = np.concatenate((feat_0, feat_1), axis=2)
            np.save(os.path.join(dir_input, name)[:-9]+".npy", feat_1)

            os.remove(os.path.join(dir_input, name))
            os.remove(os.path.join(dir_input, name)[:-9]+"_spp1.npy")

            cnt += 1
            print('merged', name[:-9])
    print(cnt)

# merge_file(dir="D:\\newdata\\train")


# cnt = 0
# for root, dirs, files in os.walk("D:\\SBMC\\train\\input", topdown=False):
#     for name in files:
#         if 'kpcn' in name:
#             # print(name)
#             os.remove(os.path.join("D:\\SBMC\\train\\input", name))
#             cnt += 1

# for root, dirs, files in os.walk("D:\\newdata\\train\\input", topdown=False):
#     random.shuffle(files)
#     l = len(files) // 4
#     # copy_cnt = (l * 4) // 7
#     copy_cnt = 5
#     print('copy {} files from total {}'.format(copy_cnt, l))
#     cnt = 0
#     for name in files:
#         if 'llpm' in name:
#             print('copying', name[:-9])

#             llpm = os.path.join("D:\\newdata\\train\\input", name)
#             prob_imp = os.path.join("D:\\newdata\\train\\input", name)[:-9] + "_prob_imp.npy"
#             sbmc_p = os.path.join("D:\\newdata\\train\\input", name)[:-9] + "_sbmc_p.npy"
#             sbmc_s = os.path.join("D:\\newdata\\train\\input", name)[:-9] + "_sbmc_s.npy"
#             # gt = os.path.join("D:\\SBMC\\train1\\gt", name)[:-9] + ".npy"
#             check_dir_exists([llpm, prob_imp, sbmc_p, sbmc_s])

#             new_llpm = os.path.join("D:\\train2\\input", name)
#             new_prob_imp = os.path.join("D:\\train2\\input", name)[:-9] + "_prob_imp.npy"
#             new_sbmc_p = os.path.join("D:\\train2\\input", name)[:-9] + "_sbmc_p.npy"
#             new_sbmc_s = os.path.join("D:\\train2\\input", name)[:-9] + "_sbmc_s.npy"
#             # new_gt = os.path.join("D:\\SBMC\\train1\\gt", name)[:-9] + ".npy"

#             shutil.move(llpm, new_llpm)
#             shutil.move(prob_imp, new_prob_imp)
#             shutil.move(sbmc_p, new_sbmc_p)
#             shutil.move(sbmc_s, new_sbmc_s)
#             # shutil.copyfile(gt, new_gt)
#             cnt += 1
#             if cnt > copy_cnt: break
# print(cnt)

# for root, dirs, files in os.walk("D:\\newdata\\train\\gt", topdown=False):
#     # random.shuffle(files)
#     cnt = 0
#     for name in files:
#         fn = name[:-4]
#         # print(fn)
#         llpm_fn_0 = os.path.join("D:\\newdata\\train\\input", fn+"_llpm.npy")
#         prob_imp = os.path.join("D:\\newdata\\train\\input", fn + "_prob_imp.npy")
#         sbmc_p = os.path.join("D:\\newdata\\train\\input", fn + "_sbmc_p.npy")
#         sbmc_s = os.path.join("D:\\newdata\\train\\input", fn + "_sbmc_s.npy")
#         print(llpm_fn_0, prob_imp, sbmc_p, sbmc_s)
#         if not os.path.isfile(llpm_fn_0) or not os.path.isfile(prob_imp) or not os.path.isfile(sbmc_p) or not os.path.isfile(sbmc_s): 
#             # print(fn)
#             cnt += 1
#             # os.remove(os.path.join("D:\\SBMC\\train\\gt", name))
#             print('missing', fn)
#         # cnt += 1
# print(cnt)


# cnt = 0
# for root, dirs, files in os.walk("D:\\p-buf\\test\\input", topdown=False):
#     for name in files:
#         if not name.endswith('_a.npy'):
#             continue
        
#         print('merging', name[:-6])
#         feat_a = np.load(os.path.join("D:\\p-buf\\test\\input", name))
#         feat_b = np.load(os.path.join("D:\\p-buf\\test\\input", name)[:-6]+"_b.npy")

#         feat_b = np.concatenate((feat_a, feat_b), axis=0)
#         np.save(os.path.join("D:\\p-buf\\test\\input", name)[:-6]+".npy", feat_b)

#         os.remove(os.path.join("D:\\p-buf\\test\\input", name))
#         os.remove(os.path.join("D:\\p-buf\\test\\input", name)[:-6]+"_b.npy")
#         cnt += 1
#         print('merged', name[:-6])


# print(cnt)

# cnt = 0
# for root, dirs, files in os.walk("D:\\p-buf\\test\\input", topdown=False):
#     for name in files:
#         if not name.endswith('_spp0.npy'):
#             continue

#         print('merging', name[:-9])
#         # TODO
#         # Extend to multiple divided 
#         feat_0 = np.load(os.path.join("D:\\p-buf\\test\\input", name))
#         feat_1 = np.load(os.path.join("D:\\p-buf\\test\\input", name)[:-9]+"_spp1.npy")
        
#         feat_1 = np.concatenate((feat_0, feat_1), axis=2)
#         np.save(os.path.join("D:\\p-buf\\test\\input", name)[:-9]+".npy", feat_1)

#         os.remove(os.path.join("D:\\p-buf\\test\\input", name))
#         os.remove(os.path.join("D:\\p-buf\\test\\input", name)[:-9]+"_spp1.npy")

#         cnt += 1
#         print('merged', name[:-9])
# print(cnt)



# cnt = 0
# for root, dirs, files in os.walk("D:\\p-buf\\test\\gt", topdown=False):
#     for name in files:
#         npy_fn = os.path.join(root, name)
#         png_fn = os.path.join("D:\\p-buf\\test\\gt_imgs", name)[:-3]+"png"
#         if os.path.isfile(png_fn):
#             continue
#         else:
#             img_np = np.load(npy_fn)[:,:,:3]
#             plt.imsave(os.path.join("D:\\p-buf\\test\\gt_imgs", name)[:-3]+"png", LinearToSrgb(ToneMap(img_np)))
#             cnt += 1
# print(cnt)



"""
cnt = 0
for root, dirs, files in os.walk("D:\\LLPM\\test\\gt", topdown=False):
    for name in files:
        feat_a = np.load(os.path.join("D:\\LLPM\\test\\input", name)[:-4]+"_a.npy")
        feat_b = np.load(os.path.join("D:\\LLPM\\test\\input", name)[:-4]+"_b.npy")

        feat_b = np.concatenate((feat_a, feat_b), axis=0)
        np.save(os.path.join("D:\\LLPM\\test\\input", name), feat_b)
        plt.imsave(os.path.join("D:\\LLPM\\test\\input_imgs", name)[:-3]+"png", feat_b[:,:,:,8:11].mean(2))
        
        os.remove(os.path.join("D:\\LLPM\\test\\input", name)[:-4]+"_a.npy")
        os.remove(os.path.join("D:\\LLPM\\test\\input", name)[:-4]+"_b.npy")
        
        cnt += 1
print(cnt)


cnt = 0
for root, dirs, files in os.walk("D:\\LLPM\\test\\gt", topdown=False):
    for name in files:
        npy_fn = os.path.join(root, name)
        png_fn = os.path.join("D:\\LLPM\\test\\gt_imgs", name)[:-3]+"png"
        if os.path.isfile(png_fn):
            continue
        else:
            img_np = np.load(npy_fn)[:,:,:3]
            plt.imsave(os.path.join("D:\\LLPM\\test\\gt_imgs", name)[:-3]+"png", LinearToSrgb(ToneMap(img_np)))
            cnt += 1
print(cnt)

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