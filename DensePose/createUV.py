import apply_net as an
import torch 
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import cv2  
import os


def parse_iuv(result):
    i = result['pred_densepose'][0].labels.cpu().numpy().astype(float)
    uv = (result['pred_densepose'][0].uv.cpu().numpy() * 255.0).astype(float)
    iuv = np.stack((uv[1, :, :], uv[0, :, :], i))
    iuv = np.transpose(iuv, (1, 2, 0))
    return iuv


def parse_bbox(result):
    return result["pred_boxes_XYXY"][0].cpu().numpy()


def concat_textures(array):
    texture = []
    for i in range(4):
        tmp = array[6 * i]
        for j in range(6 * i + 1, 6 * i + 6):
            tmp = np.concatenate((tmp, array[j]), axis=1)
        texture = tmp if len(texture) == 0 else np.concatenate((texture, tmp), axis=0)
    return texture


def interpolate_tex(tex):
    # code is adopted from https://github.com/facebookresearch/DensePose/issues/68
    valid_mask = np.array((tex.sum(0) != 0) * 1, dtype='uint8')
    radius_increase = 10
    kernel = np.ones((radius_increase, radius_increase), np.uint8)
    dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
    region_to_fill = dilated_mask - valid_mask
    invalid_region = 1 - valid_mask
    actual_part_max = tex.max()
    actual_part_min = tex.min()
    actual_part_uint = np.array((tex - actual_part_min) / (actual_part_max - actual_part_min) * 255, dtype='uint8')
    actual_part_uint = cv2.inpaint(actual_part_uint.transpose((1, 2, 0)), invalid_region, 1,
                               cv2.INPAINT_TELEA).transpose((2, 0, 1))
    actual_part = (actual_part_uint / 255.0) * (actual_part_max - actual_part_min) + actual_part_min
    # only use dilated part
    actual_part = actual_part * dilated_mask

    return actual_part


def get_texture(im, iuv, bbox, tex_part_size=200):
    # this part of code creates iuv image which corresponds
    # to the size of original image (iuv from densepose is placed
    # within pose bounding box).
    im = im.transpose(2, 1, 0) / 255
    image_w, image_h = im.shape[1], im.shape[2]
    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]
    x, y, w, h = [int(v) for v in bbox]
    bg = np.zeros((image_h, image_w, 3))
    bg[y:y + h, x:x + w, :] = iuv
    iuv = bg
    iuv = iuv.transpose((2, 1, 0))
    i, u, v = iuv[2], iuv[1], iuv[0]

    # following part of code iterate over parts and creates textures
    # of size `tex_part_size x tex_part_size`
    n_parts = 24
    texture = np.zeros((n_parts, 3, tex_part_size, tex_part_size))
    
    for part_id in range(1, n_parts + 1):
        generated = np.zeros((3, tex_part_size, tex_part_size))

        x, y = u[i == part_id], v[i == part_id]
        # transform uv coodrinates to current UV texture coordinates:
        tex_u_coo = (x * (tex_part_size - 1) / 255).astype(int)
        tex_v_coo = (y * (tex_part_size - 1) / 255).astype(int)
        
        # clipping due to issues encountered in denspose output;
        # for unknown reason, some `uv` coos are out of bound [0, 1]
        tex_u_coo = np.clip(tex_u_coo, 0, tex_part_size - 1)
        tex_v_coo = np.clip(tex_v_coo, 0, tex_part_size - 1)
        
        # write corresponding pixels from original image to UV texture
        # iterate in range(3) due to 3 chanels
        for channel in range(3):
            generated[channel][tex_v_coo, tex_u_coo] = im[channel][i == part_id]
        
        # this part is not crucial, but gives you better results 
        # (texture comes out more smooth)
        if np.sum(generated) > 0:
            generated = interpolate_tex(generated)

        # assign part to final texture carrier
        texture[part_id - 1] = generated[:, ::-1, :]
    
    # concatenate textures and create 2D plane (UV)
    tex_concat = np.zeros((24, tex_part_size, tex_part_size, 3))
    for i in range(texture.shape[0]):
        tex_concat[i] = texture[i].transpose(2, 1, 0)
    tex = concat_textures(tex_concat)
    return tex

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
        iuv = parse_iuv(results)
        bbox = parse_bbox(results)
        image = cv2.imread(impath)[:, :, ::-1]
        uv_texture = get_texture(image, iuv, bbox)
        
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
    