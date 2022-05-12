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

from decompose_to_yml import decompose_to_ymls
from edit_alpha_poster_demo import alpha_images_editing
from edit_poster import images_editing
import cv2
from tempfile import TemporaryDirectory

import argparse

def process_images(exp_dir, alpha=False):
    data_root = exp_dir
    target_path = os.path.join(data_root, 'target.jpg')
    src_path = os.path.join(data_root, 'source.jpg')
    # scene_yml_path = "./data/scene_decomposed.yml"    
    datadir_path = os.path.join(data_root, 'scenes')
    scene_yml_paths = decompose_to_ymls(datadir_path, True)    
    
    if alpha:
        alpha_images_editing(target_path, scene_yml_path, src_path, data_root)
    else:
        results = images_editing(target_path, scene_yml_path, src_path, data_root)

    output_root = os.path.join(data_root, 'output')
    for i, result in enumerate(results):
        scene_yml_path = scene_yml_paths[i]
        prefix=os.path.split(scene_yml_path)[1][:-4]
        cv2.imwrite(os.path.join(output_root, prefix+'_soutput.jpg'), result)


def read_video(video_path, W=640, H=480):
    cap = cv2.VideoCapture(video_path)
    imgs = []    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (W, H))
            imgs.append(frame)            
        else:
            break
    fps = cap.get(cv2.CAP_PROP_FPS)    
    cap.release()
    print(len(imgs))
    return imgs, fps

def save_video(imgs, size, video_path, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, size)
    for frame in imgs:
        out.write(frame)
    out.release()
    print('video saved')

def process_video(exp_dir, video_path, output_path, target_path, src_path, alpha=False):
    data_root = exp_dir
    
    # read video, save to temporary directory
    cap = cv2.VideoCapture(video_path)
    with TemporaryDirectory() as datadir_path:
        print('frame save dir: ', datadir_path)
        id = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if frame.shape[0] > frame.shape[1]: # H > W
                    frame = cv2.resize(frame, (480, 640))     #size = (W, H)
                else: # H <= W
                    frame = cv2.resize(frame, (640, 480))     #size = (W, H)
                image_path = os.path.join(datadir_path, str(id)+'.jpg')                
                cv2.imwrite(image_path, frame)
                id = id + 1
                #imgs.append(frame)            
            else:
                break
        fps = cap.get(cv2.CAP_PROP_FPS)    
        cap.release()

        
        # scene_yml_path = "./data/scene_decomposed.yml"            
        scene_yml_paths = decompose_to_ymls(data_root, datadir_path, True)
        print('decompose '+str(len(scene_yml_paths))+' frames')
        
        if alpha:
            results = alpha_images_editing(target_path, scene_yml_paths, src_path, data_root)
        else:
            results = images_editing(target_path, scene_yml_paths, src_path, data_root)
               
        save_video(results, fps=fps, size=(results[0].shape[1], results[0].shape[0]), video_path=output_path)

    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--exp_dir', type=str, default='./data/exp2')
    args.add_argument('--video_name', type=str, default='input.mp4')
    args.add_argument('--output_name', type=str, default='output.avi')
    args.add_argument('--target_name', type=str, default='target.jpg')
    args.add_argument('--source_name', type=str, default='source.jpg')
    parser = args.parse_args()

    alpha = True     
    
    exp_dir = parser.exp_dir
    video_path = os.path.join(exp_dir, parser.video_name)    
    target_path = os.path.join(exp_dir, parser.target_name)
    src_path = os.path.join(exp_dir, parser.source_name)
    print('input video: '+video_path)
    print('target: '+target_path)
    print('source: '+src_path)
    if not (os.path.exists(video_path) and os.path.exists(target_path) and os.path.exists(src_path)):
        raise FileNotFoundError()

    output_path = os.path.join(exp_dir, parser.output_name)
    process_video(exp_dir=exp_dir, video_path=video_path, output_path=output_path, target_path=target_path, src_path=src_path, alpha=False)

    '''
    data_root = parser.exp_dir 
    target_path = os.path.join(data_root, parser.target_name) 
    src_path = os.path.join(data_root, parser.source_name)
    scene_yml_paths = decompose_to_yml(data_root, "", True)
              
    #scene_yml_paths = []
    #for i in range(50):
    #    scene_yml_paths.append(data_root+'scenes_yml_50/'+str(i)+'.yml')
    #print('decompose '+str(len(scene_yml_paths))+' frames')
    

    if alpha:
        results = alpha_image_editing(target_path, scene_yml_paths, src_path, data_root)
    else:
        results = image_editing(target_path, scene_yml_paths, src_path, data_root)
        
    save_video(results, size=(results[0].shape[1], results[0].shape[0]), video_path=os.path.join(data_root, 'output.avi'))
    '''

    os.system('rm -rf '+os.path.join(exp_dir, 'decompose'))
    os.system('rm -rf '+os.path.join(exp_dir, 'scenes_yml'))
    
