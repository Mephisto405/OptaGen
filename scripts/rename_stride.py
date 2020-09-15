import os

cnt = 0
for root, dirs, files in os.walk("D:\\LLPM\\train\\gt", topdown=False):
    for name in files:
        a = name
        dst = a[:a.find('_')+1] + str(int(a[a.find('_')+1:a.find('.')]) + 40) + a[a.find('.'):]

        os.rename(os.path.join(root, name), os.path.join(root, dst))
        os.rename(os.path.join("D:\\LLPM\\train\\input", name), 
                  os.path.join("D:\\LLPM\\train\\input", dst))
        os.rename(os.path.join("D:\\LLPM\\train\\gt_imgs", name[:-3]+"png"), 
                  os.path.join("D:\\LLPM\\train\\gt_imgs", dst[:-3]+"png"))
        os.rename(os.path.join("D:\\LLPM\\train\\input_imgs", name[:-3]+"png"), 
                  os.path.join("D:\\LLPM\\train\\input_imgs", dst[:-3]+"png"))

        cnt += 1
print(cnt)