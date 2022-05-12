import os
import sys
from third_party.NIID.config import TestOptions
from transfer import transfer, transfer_single
from third_party.NIID.decompose import decompose_images, decompose_image

from tempfile import TemporaryDirectory
from tempfile import NamedTemporaryFile
from os import listdir


def create_fns_file(decompose_path, fns_path):
    file_list = listdir(decompose_path)
    included_list = []

    for f in file_list:            
        if f.endswith('.pth.tar'):
            included_list.append(os.path.join(decompose_path, f[:-8])+'\n')                  
    included_list.sort(key = lambda x : int(os.path.split(x)[1][:-12]))    
    print(included_list)

    with open(fns_path, 'w') as fns:
        for path in included_list:
            fns.write(path)

    return included_list

def decompose_to_ymls(data_root, datadir_path="", save_decomposed=False):
    decompose_path = os.path.join(data_root,'decompose')
    if save_decomposed:
        if not os.path.exists(decompose_path):
            decompose_dir = os.mkdir(decompose_path)
    else:
        decompose_dir = TemporaryDirectory()
        decompose_path = decompose_dir.name 
    print('decompose path: '+decompose_path)
    
    decompose_images(datadir_path, 
                    decompose_path,
                    os.path.join(data_root, 'decompose'),
                    False,
                    **{
                        'pretrained_file': 'pretrained_model/final.pth.tar',
                        'offline': True,
                        'gpu_devices': [0],
                        }
                    )
    
    fns_file = NamedTemporaryFile()
    include_list = create_fns_file(decompose_path, fns_file.name)    
    #print('fns path: ', fns_file.name)
    #print(include_list)

    yml_output_dir_path = os.path.join(data_root, 'scenes_yml')
    if not os.path.exists(yml_output_dir_path):
        os.mkdir(yml_output_dir_path)
   
    yml_paths = transfer(fns_file.name, yml_output_dir_path)
    print(yml_paths)
    os.system('cp '+fns_file.name+' '+os.path.join(data_root, 'fns.txt'))
    return yml_paths
    
def decompose_to_yml(data_root, img_name, save_decomposed=False):
    decompose_path = os.path.join(data_root, 'decompose')
    if save_decomposed:
        if not os.path.exists(decompose_path):
            decompose_dir = os.mkdir(decompose_path)
    else:
        decompose_dir = TemporaryDirectory()
        decompose_path = decompose_dir.name
    print('decompose path: '+decompose_path)

    decompose_image(data_root, img_name, decompose_path, False,                    
                    **{
                        'pretrained_file': 'pretrained_model/final.pth.tar',
                        'offline': True,
                        'gpu_devices': [0],
                        }
                    )
    fn = os.path.join(data_root, '%s_decomposed' % (img_name[:img_name.rfind('.')]))
    print('file name: '+fn)
    yml_path = transfer_single(data_root, fn)
    return yml_path



if __name__ == '__main__':
    data_root = './data/exp1/'
    datadir_path = os.path.join(data_root, 'scenes')
    print(decompose_to_ymls(datadir_path, True))