import numpy as np
from PIL import Image
import os


def get_grid(imgs, args, epoch, it):
    grid_w, grid_h = args.grid_w, args.grid_h
    img_w, img_h = imgs[0].shape[2], imgs[0].shape[1]
    # nimgs = Image.new('L', (grid_h * img_h, grid_w * img_w))
    nimgs = np.zeros(shape=(grid_h*img_h, grid_w*img_w))
    for idx, x in enumerate(imgs):
        x = x.reshape(img_h, img_w)
        # x = x * 0.5 + 0.5
        ww, hh = idx % grid_w, idx // grid_w
        nimgs[hh*img_h: hh*img_h+img_h, ww*img_w: ww*img_w+img_w] = x[:, :]
        # print(nimgs[hh*img_h: hh*img_h+img_h, ww*img_w: ww*img_w+img_w])
    # nimgs = nimgs.reshape(1, grid_h*img_h, grid_w*img_w)
    nimgs = nimgs*255
    nimgs = Image.fromarray(nimgs.astype('uint8'), mode='L')
    nimgs.save(os.path.join(args.output, "{}_{}_{:03d}_{:04d}.jpg".format(args.model, args.dataset, epoch, it)))
    # nimgs.show()
    # plt.axis('off')
    # plt.imshow(nimgs, cmap='gray')
    # plt.savefig(os.path.join(args.output, "{}_{:03d}_{:04d}.jpg".format(args.model, epoch, it)), format='jpg')
    return nimgs
