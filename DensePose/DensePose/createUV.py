import apply_net as an

import unpack_IUV 
import torch 
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import cv2  
import os
from pathlib import Path

import pickle

configpath = 'configs/densepose_rcnn_R_50_FPN_s1x.yaml'
model = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl'


def infere_image(image, imdir = None):
    create_mask(image, imdir)
    file, dir = create_pkl(image, imdir)
    texture(pkl_file = file, pkl_dir = dir, directory = imdir)

def infere_images(directory):
    for image in os.listdir(directory):
        f = os.path.join(directory, image)
        # checking if it is a file
        if os.path.isfile(f):
            infere_image(image, directory)

def create_pkl(image , directory = None, savedir = 'data/pkl_files/'):
    filename = image.split('.')[0]
    outpath = savedir + filename+ '.pkl'
    impath = directory + '/' + image
    
    parser = an.create_argument_parser()
    args = parser.parse_args(['dump',configpath, model, impath, '--output',outpath])

    verbosity = args.verbosity if hasattr(args, "verbosity") else None
    global logger
    logger = an.setup_logger(name=an.LOGGER_NAME)
    logger.setLevel(an.verbosity_to_level(verbosity))
    args.func(args)

    return filename+ '.pkl', savedir

def create_mask(image , directory = None, visualization = 'dp_contour,bbox' , savedir = 'data/mask/' ): # 'dp_contour,bbox' , 'bbox,dp_segm' 
    filename = image.split('.')[0]
    outpath = savedir + filename+ '_mask.jpg'
    impath = directory + '/' + image

    parser = an.create_argument_parser()
    args = parser.parse_args(['show',configpath, model, impath, visualization,'-v', '--output',outpath])
    
    verbosity = args.verbosity if hasattr(args, "verbosity") else None
    global logger
    logger = an.setup_logger(name=an.LOGGER_NAME)
    logger.setLevel(an.verbosity_to_level(verbosity))
    args.func(args)

    return savedir + filename + '.jpg'

# def loadall(filename):
#     with open(filename, "rb") as f:
#         while True:
#             try:
#                 yield pickle.load(f)
#             except EOFError:
#                 break

def texture(pkl_file, pkl_dir = None, directory = None, savedir = 'data/textures/'):

    filename = pkl_file.split('.')[0]
    outpath = savedir + filename+ '.jpg'
    impath = directory + '/' + filename + '.jpg'

    with open(pkl_dir + pkl_file, 'rb') as f:
        data = torch.load(f, map_location=device)
        
    if 'pred_densepose' in  data[0]:
        i = data[0]['pred_densepose'][0].labels.cpu().numpy()
        uv = data[0]['pred_densepose'][0].uv.cpu().numpy()
        # I assume the data are stored in pickle, and you are able to read them 
        results = data[0]
        iuv = unpack_IUV.parse_iuv(results)
        bbox = unpack_IUV.parse_bbox(results)
        image = cv2.imread(impath)[:, :, ::-1]
        uv_texture = unpack_IUV.get_texture(image, iuv, bbox)
        
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.imsave(outpath,uv_texture)
        
    else:
        array = np.zeros((800,1200), dtype='float')
        plt.imsave(outpath, array.astype('uint8'), cmap=matplotlib.cm.gray, vmin=0, vmax=255)
        print(outpath)
        
    return outpath

if __name__ == '__main__':
    infere_images('data/images')
    