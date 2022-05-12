#!/usr/bin/env python
# coding: utf-8


import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm


def srgb_to_rgb(srgb):
    ret = torch.zeros(srgb.size()).type(srgb.type())
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = torch.pow((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret

def rgb_to_srgb(rgb):
    ret = torch.zeros(rgb.size()).type(rgb.type())
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = torch.pow(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret

def normalize_surface_normal(normal):
    length = torch.sum(normal ** 2, dim=0, keepdim=True) ** 0.5
    normal = normal / (length + 1e-6)

    return normal

def swap_channels(rgb):
    (r, g, b) = cv2.split(rgb)
    return cv2.merge([b, g, r])

def transfer_single(data_root, name):
    fn = name + '.pth.tar'
    output_name = os.path.join(data_root, os.path.split(name)[-1][:-11] + '.yml')
    if not os.path.exists(fn):
            raise FileExistsError(fn)
        
    t = torch.load(fn)

    input_srgb = t['input_srgb']
    rgb = srgb_to_rgb(input_srgb)        
    #rgb = input_srgb

    pred_S = t['pred_S']

    pred_R = t['pred_R']
    pred_R = torch.clamp(pred_R, 0, 1)

    img = rgb.permute(1, 2, 0).numpy() * 255
    # rgb to bgr
    img = swap_channels(img)    
    # cv2.waitKey(0)
    #cv2.imwrite(result_dir + 'rgb.png', img)

    # rgbout is still rgb
    rgbout = rgb.permute(1, 2, 0).numpy()
    predSout = pred_S.permute(1, 2, 0).numpy()
    predRout = pred_R.permute(1, 2, 0).numpy()
    # predLout = pred_L.permute(1, 2, 0).numpy()
    # predNout = pred_N.permute(1, 2, 0).numpy()
    f = cv2.FileStorage(output_name, cv2.FILE_STORAGE_WRITE)                
    #print(np.max(rgbout))
    #print(np.max(predSout))

    f.write("rgb", rgbout)
    f.write("pred_S", predSout)
    f.write("pred_R", predRout)
    # f.write("pred_L", predLout)
    # f.write("pred_N", predNout)

    f.release()
    return output_name

def transfer(input_file, output_dir):
    # name = "../build/multi-lighting-result/multi-lighting/0336/00/decompose_results_1.0/0407_scale_1.0"
    
    lines = sum(1 for i in open(input_file, "r"))
    #assert args.i is not None
    #fns = []
    #fns.append(args.i)
    yml_paths = []
    bar = tqdm(open(input_file, 'r'), total=lines)
    for line in bar:        
        # print(line)
        # result_dir = "./results/"
        # name = "../build/multi-lighting-result/multi-lighting/0473/00/decompose_results_1.0/0001_scale_1.0"
        name = line.rstrip('\n')
        bar.set_description('Transfer %s' % os.path.split(name)[-1][:-11])
        fn = name + ".pth.tar"
        output_name = os.path.join(output_dir, os.path.split(name)[-1][:-11] + '.yml')
        yml_paths.append(output_name)
        
        if not os.path.exists(fn):
            raise FileExistsError(fn)
        
        t = torch.load(fn)
        # rendered_img = t['rendered_img']

        input_srgb = t['input_srgb']
        rgb = srgb_to_rgb(input_srgb)        

        # pred_N = t['pred_N']
        # pred_N = normalize_surface_normal(pred_N)

        pred_S = t['pred_S']

        #pred_L = t['pred_L']

        pred_R = t['pred_R']
        pred_R = torch.clamp(pred_R, 0, 1)


        # In[5]:
        #fig = plt.figure()
        #fig.add_subplot(1, 2, 1)
        #plt.imshow(input_srgb.permute(1, 2, 0))
        #fig.add_subplot(1, 2, 2)
        #plt.imshow(rgb.permute(1, 2, 0))
        # plt.imshow(rgb_to_srgb(input_srgb.permute(1, 2, 0)))
        # plt.imshow(rgb_to_srgb(rendered_img.permute(1, 2, 0)))
        # plt.show()


        # In[7]:


        img = rgb.permute(1, 2, 0).numpy() * 255
        # rgb to bgr
        img = swap_channels(img)
        # cv2.imshow("img", img / 255)
        # cv2.waitKey(0)
        #cv2.imwrite(result_dir + 'rgb.png', img)

        # rgbout is still rgb
        rgbout = rgb.permute(1, 2, 0).numpy()
        predSout = pred_S.permute(1, 2, 0).numpy()
        predRout = pred_R.permute(1, 2, 0).numpy()
        # predLout = pred_L.permute(1, 2, 0).numpy()
        # predNout = pred_N.permute(1, 2, 0).numpy()
        f = cv2.FileStorage(output_name, cv2.FILE_STORAGE_WRITE)                
        #print(np.max(rgbout))
        #print(np.max(predSout))

        f.write("rgb", rgbout)
        f.write("pred_S", predSout)
        f.write("pred_R", predRout)
        # f.write("pred_L", predLout)
        # f.write("pred_N", predNout)

        f.release()
        
    return yml_paths