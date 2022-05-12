from click import edit
import cv2
import numpy as np
import math 
import os 
SHOW = False

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

def split(fn, data_root):
    #if os.path.exists(fn):
    #    raise FileExistsError(fn+" not exist")
    ifs = cv2.FileStorage(fn, cv2.FILE_STORAGE_READ)
    rgb = ifs.getNode("rgb").mat()
    s = ifs.getNode("pred_S").mat()
    r = ifs.getNode("pred_R").mat()
    rgb = rgb[:,:,::-1].astype(np.float64)
    s = s[:,:,::-1].astype(np.float64)
    r = r[:,:,::-1].astype(np.float64)
    

    '''
    render = cv2.multiply(s, r)
    if SHOW:
        cv2.imshow("srgb", rgb_to_srgb(rgb))
        cv2.imshow("s", rgb_to_srgb(s))
        cv2.imshow("r", rgb_to_srgb(r))
        cv2.imshow("render", render)
        cv2.imshow("srender", rgb_to_srgb(render))
        cv2.waitKey()
        cv2.destroyAllWindows()
    '''
    ofs = cv2.FileStorage(os.path.join(data_root, "tmp.yml"), cv2.FILE_STORAGE_WRITE)
    ofs.write("illumination", s)
    ofs.write("reflectance", r)
    ofs.write("rgb", rgb)
    

def render(poster_brightness: float, data_root: str, prefix: str):
    ifs_tmp = cv2.FileStorage(os.path.join(data_root, "tmp.yml"), cv2.FILE_STORAGE_READ)
    ifs_replace = cv2.FileStorage(os.path.join(data_root, "replace.yml"), cv2.FILE_STORAGE_READ)
    replace = ifs_replace.getNode("replace").mat().astype(np.float64)#(0,255)
    illumination = ifs_tmp.getNode("illumination").mat().astype(np.float64) #(0,1)
    reflectance = ifs_tmp.getNode("reflectance").mat().astype(np.float64)*255 #(0,255)
    rgb = ifs_tmp.getNode("rgb").mat()*255 #(0,255)
    mask = cv2.imread(os.path.join(data_root, "mask.jpg"), cv2.IMREAD_UNCHANGED)    
    
    rgb_sum = rgb.sum(axis=-1)
    r_sum = replace.sum(axis=-1)
    ref_sum = reflectance.sum(axis=-1)
    tot = np.count_nonzero(r_sum)
    rgbi = rgb_sum.sum()/(3.0*tot)
    ri = r_sum.sum()/(3.0*tot)
    refi = ref_sum.sum()/(3.0*tot)
    refi = refi * poster_brightness
    
    mask = mask[:,:,np.newaxis]
    dst = (255-mask)/255.0 * rgb + mask/255.0 * replace * illumination * refi / ri 
    #dst = replace * illumination * refi / ri
    #dst = dst / 255
    #sdst = (rgb_to_srgb(dst)*255).astype(np.uint8)
    sdst = dst.astype(np.uint8)

    #output_root = os.path.join(data_root, 'output')
    #cv2.imwrite(os.path.join(output_root, prefix+'_soutput.jpg'), sdst)
    #cv2.imwrite(os.path.join(output_root, prefix+'_output.jpg'), (dst*255).astype(np.uint8))
    return sdst

    #cv2.imshow("sdst", sdst)
    #cv2.waitKey()
    #cv2.destroyAllWindows()