import argparse
import enum
import os 
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import sys
sys.path.append('./third_party/NIID')
sys.path.append('./third_party/LIFE')
sys.path.append('./third_party/LIFE/core')

from decompose_to_yml import decompose_to_yml
from edit_alpha_poster_demo import alpha_image_editing
from edit_poster import image_editing
import cv2
from tempfile import TemporaryDirectory

import argparse

def resize_rewrite(img_path):
    frame = cv2.imread(img_path)
    if frame.shape[0] > frame.shape[1]: # H > W
        frame = cv2.resize(frame, (480, 640))     #size = (W, H)
    else: # H <= W
        frame = cv2.resize(frame, (640, 480))     #size = (W, H)               
    resize_name = '%s_resize.jpg' % (img_path[:img_path.rfind('.')])
    cv2.imwrite(resize_name, frame)
    return resize_name
    

def process_image(exp_dir, scene_name, target_name, source_name, alpha=False):
    data_root = exp_dir    
    # scene_yml_path = "./data/scene_decomposed.yml"  
    scene_path = os.path.join(data_root, scene_name)    
    target_path = os.path.join(data_root, target_name)
    src_path = os.path.join(data_root, source_name)         
    if not (os.path.exists(scene_path) and os.path.exists(target_path) and os.path.exists(src_path)):
        raise FileNotFoundError()    

    
    scene_path = resize_rewrite(scene_path)        
    target_path = resize_rewrite(target_path)
    src_path = resize_rewrite(src_path)
    
    scene_name = os.path.split(scene_path)[-1]
    print('scene: '+scene_path)
    print('target: '+target_path)
    print('source: '+src_path) 

    scene_yml_path = decompose_to_yml(data_root, scene_name, True)    
    
    if alpha:
        alpha_image_editing(target_path, scene_yml_path, src_path, data_root)
    else:
        result = image_editing(target_path, scene_yml_path, src_path, data_root)

    prefix=os.path.split(scene_yml_path)[1][:-4]
    cv2.imwrite(os.path.join(data_root, prefix+'_soutput.jpg'), result)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--exp_dir', type=str, default='./data/exp7')
    args.add_argument('--scene_name', type=str, default='scene.jpg')    
    args.add_argument('--target_name', type=str, default='fantastic_beast.jpg')
    args.add_argument('--source_name', type=str, default='sonic.jpg')
    parser = args.parse_args()

    alpha = False     
    
    exp_dir = parser.exp_dir

    
    process_image(exp_dir=exp_dir, scene_name=parser.scene_name, target_name=parser.target_name, source_name=parser.source_name, alpha=False)

    
