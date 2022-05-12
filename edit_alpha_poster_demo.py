import cv2
import numpy as np
import math 
import torch
from skimage import io 
from matplotlib import pyplot as plt
import os
# LIFE path
import sys 
from third_party.LIFE.config import get_life_args, get_raft_args
from third_party.LIFE.flow_estimator import Flow_estimator
from third_party.LIFE.core.utils.utils import image_flow_warp

from decompose_to_yml import decompose_to_yml
DEBUG = True

def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret

def rgb_to_srgb(rgb):
    ret = np.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = np.power(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret

def alpha_image_editing(target_path:str,
                        scene_yml_path:str,
                        ssrc_path:str,
                        data_root:str):
    # step0: Load model
    args = get_life_args()
    args.model = "../LIFE/model/pretrain.pth"
    print(args)
    estimator = Flow_estimator(args, method='life')

    # step1: process target & source images
    H, W = 480, 640

    target = cv2.imread(target_path, cv2.IMREAD_UNCHANGED) #bgr    
    target = np.float64(target)
    target = target / 255.0
    starget = rgb_to_srgb(target)
    starget = (starget*255.0).astype(np.uint8)
    starget = cv2.resize(starget, (W, H))

    ssrc = cv2.imread(ssrc_path, cv2.IMREAD_UNCHANGED) #bgra
    if ssrc.shape[2] == 4:        
        print ("with alpha channel")    
        src_alpha = ssrc[:,:,3] #alpha channel
        src = ssrc[:,:,:3] #bgr    
    elif ssrc.shape[2] == 3:
        raise ValueError("no alpha channel")        
    else:
        raise ValueError("replace image should have 3 or 4 channels")
    src = cv2.resize(src, (W, H))
    src_alpha = cv2.resize(src_alpha, (W, H))

    if not os.path.exists(scene_yml_path):
        raise FileExistsError(scene_yml_path)
      
    print(scene_yml_path)
    ifs = cv2.FileStorage(scene_yml_path, cv2.FILE_STORAGE_READ)
    dst = ifs.getNode("rgb").mat()
    dst = np.float64(dst)
    dst = dst[:,:,::-1] #rgb to bgr  
    sdst = rgb_to_srgb(dst)    
    sdst = (sdst*255.0).astype(np.uint8)

    #step2: find flow target -> scene
    ## resize images
    ori_H, ori_W = dst.shape[:2]
    sdst = cv2.resize(sdst, (W, H))
    
    ## estimate
    flow = estimator.estimate(sdst, starget)    
    ## warp
    out = image_flow_warp(src, flow[0].permute([1,2,0]))
    out_alpha = image_flow_warp(src_alpha[:,:,np.newaxis], flow[0].permute([1,2,0]))
    
    intensity = np.linalg.norm(out, axis=2)
    mask = (intensity == 0)[:,:,np.newaxis]        
    result = (out*(1-mask) + sdst*mask).astype(np.uint8)
    result = cv2.resize(result, (ori_W, ori_H))
    out = cv2.resize(out, (ori_W, ori_H))

    out_alpha = cv2.resize(out_alpha, (ori_W, ori_H))

    #ofs = cv2.FileStorage("./data/alpha.yml", cv2.FILE_STORAGE_WRITE)
    #ofs.write("alpha", src_alpha)
    #cv2.imshow("alpha", src_alpha) #(0, 255)
    #cv2.imshow("mask", out_alpha) # mask means alpha mask
    #cv2.imshow("out", out)
    #cv2.imshow("result", result)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    #return 

    cv2.imwrite(os.path.join(data_root,"mask.jpg"), out_alpha) #(0, 255)
    #cv2.imwrite("./data/replace.jpg", result) #(0, 255)
    ofs = cv2.FileStorage(os.path.join(data_root, "replace.yml"), cv2.FILE_STORAGE_WRITE)
    ofs.write("replace", result)    
    return result

    
def alpha_images_editing(target_path: str, 
                        scene_yml_paths:str, 
                        ssrc_path:str,
                        data_root: str):
    '''
    Args:
        target: image in scene to replace
        ssrc: replace target with ssrc (srgb)
        scene(dst): scene image
    '''
    # step0: Load model
    args = get_life_args()
    args.model = "../LIFE/model/pretrain.pth"
    print(args)
    estimator = Flow_estimator(args, method='life')

    # step1: process target & source images
    H, W = 480, 640

    target = cv2.imread(target_path, cv2.IMREAD_UNCHANGED) #bgr    
    target = np.float64(target)
    target = target / 255.0
    starget = rgb_to_srgb(target)
    starget = (starget*255.0).astype(np.uint8)
    starget = cv2.resize(starget, (W, H))

    ssrc = cv2.imread(ssrc_path, cv2.IMREAD_UNCHANGED) #bgra
    if ssrc.shape[2] == 4:        
        print ("with alpha channel")    
        src_alpha = ssrc[:,:,3] #alpha channel
        src = ssrc[:,:,:3] #bgr    
    elif ssrc.shape[2] == 3:
        raise ValueError("no alpha channel")        
    else:
        raise ValueError("replace image should have 3 or 4 channels")
    src = cv2.resize(src, (W, H))
    src_alpha = cv2.resize(src_alpha, (W, H))


    if not os.path.exists(scene_yml_paths):
        raise FileExistsError(scene_yml_paths)  
    results = []  
    for scene_yml_path in scene_yml_paths:
        print(scene_yml_path)
        ifs = cv2.FileStorage(scene_yml_path, cv2.FILE_STORAGE_READ)
        dst = ifs.getNode("rgb").mat()
        dst = np.float64(dst)
        dst = dst[:,:,::-1] #rgb to bgr  
        sdst = rgb_to_srgb(dst)    
        sdst = (sdst*255.0).astype(np.uint8)
        
        '''
        cv2.imshow("target", starget)
        cv2.imshow("scene", sdst)
        cv2.imshow("src", src)
        cv2.waitKey()
        cv2.destroyAllWindows()   
        print(starget.max(), sdst.max(), src.max()) 
        '''
        #step2: find flow target -> scene
        ## resize images
        ori_H, ori_W = dst.shape[:2]
        sdst = cv2.resize(sdst, (W, H))
        
        ## estimate
        flow = estimator.estimate(sdst, starget)    
        ## warp
        out = image_flow_warp(src, flow[0].permute([1,2,0]))
        out_alpha = image_flow_warp(src_alpha[:,:,np.newaxis], flow[0].permute([1,2,0]))
        
        intensity = np.linalg.norm(out, axis=2)
        mask = (intensity == 0)[:,:,np.newaxis]        
        result = (out*(1-mask) + sdst*mask).astype(np.uint8)
        result = cv2.resize(result, (ori_W, ori_H))
        out = cv2.resize(out, (ori_W, ori_H))

        out_alpha = cv2.resize(out_alpha, (ori_W, ori_H))

        #ofs = cv2.FileStorage("./data/alpha.yml", cv2.FILE_STORAGE_WRITE)
        #ofs.write("alpha", src_alpha)
        #cv2.imshow("alpha", src_alpha) #(0, 255)
        #cv2.imshow("mask", out_alpha) # mask means alpha mask
        #cv2.imshow("out", out)
        #cv2.imshow("result", result)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        #return 

        cv2.imwrite(os.path.join(data_root,"mask.jpg"), out_alpha) #(0, 255)
        #cv2.imwrite("./data/replace.jpg", result) #(0, 255)
        ofs = cv2.FileStorage(os.path.join(data_root, "replace.yml"), cv2.FILE_STORAGE_WRITE)
        ofs.write("replace", result)
        results.append(result)
    return results
    

if __name__ == '__main__'   :
    data_root = './data/exp1/'
    target_path = os.path.join(data_root, 'target.jpg')
    ssrc_path = os.path.join(data_root, 'source.jpg')
    # scene_yml_path = "./data/scene_decomposed.yml"    
    datadir_path = os.path.join(data_root, 'images')
    scene_yml_paths = decompose_to_yml(datadir_path, False)    
    alpha_images_editing(target_path, scene_yml_paths, ssrc_path, data_root)
