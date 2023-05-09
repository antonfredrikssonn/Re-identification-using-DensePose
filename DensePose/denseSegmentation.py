import os
import createUV as cuv
import cv2
from pathlib import Path
import json
import torch 
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def make_folders(person_dir):
    if not os.path.isdir(person_dir):
        Path(person_dir + "/crops").mkdir(exist_ok=True, parents=True)  
        Path(person_dir + "/pkl_files").mkdir(exist_ok=True, parents=True)  
        Path(person_dir + "/uv_maps").mkdir(exist_ok=True, parents=True)  
            
def fill_folders(name,crop,person_dir):
    cv2.imwrite(f"{person_dir}/crops/{name}.jpg", crop)
    pkl_file , pkl_dir = cuv.create_pkl(f"{name}.jpg",directory = f"{person_dir}/crops",savedir = f"{person_dir}/pkl_files/")             
    cuv.texture(pkl_file , pkl_dir, directory = f"{person_dir}/crops",savedir = f"{person_dir}/uv_maps/") 

def _crop(im, bbox):
    ih, iw, _ = im.shape
    b = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    x, y, w, h = [int(v * r) for v, r in zip(b, [iw, ih, iw, ih])]
    return im[y : y + h, x : x + w, :]

def parse_train_json(train, data_dir, output_dir):
    with open(train, 'r') as f:
        data = json.load(f)
    
        for person in data:        
            person_id = person["person_id"].replace("/", "-")       
            person_dir = f"{output_dir}/{person_id}"

            make_folders(person_dir)

            for idx, sample in enumerate(person["samples"]):
                im = cv2.imread(str(data_dir + '/' + sample["image_path"]))
                bbox = [sample[k] for k in ("xtl", "ytl", "xbr", "ybr")]
                crop = _crop(im, bbox)  
                fill_folders(f"{idx:06}",crop,person_dir)
            
def create_segmap(output_dir, images_dir):
    for image in os.listdir(images_dir):
        f = os.path.join(images_dir, image)
        if os.path.isfile(f):
            pkl_file , pkl_dir = cuv.create_pkl(image,images_dir,f"{output_dir}/pkl_files_query/")             
            cuv.texture(pkl_file , pkl_dir, images_dir, f"{output_dir}/uv_maps_query/") 

def fix(output_dir,images_dir):
    for image in os.listdir(images_dir):
        f = os.path.join(images_dir, image)
        texture = os.path.join(output_dir,"uv_maps_test", image)
        if os.path.isfile(f):
            if not os.path.isfile(texture):
                if not image == "Thumbs.db":
                    print(texture)
                    pkl_file , pkl_dir = cuv.create_pkl(image,images_dir,f"{output_dir}/pkl_files_test/")             
                    cuv.texture(pkl_file , pkl_dir, images_dir, f"{output_dir}/uv_maps_test/") 

def remove(images_dir):
    for image in os.listdir(images_dir):
        f = os.path.join(images_dir, image)
        if os.path.isfile(f):
            id = image.split('_')[0]
            if id == '-1':
                os.remove(f)
            
if __name__ == '__main__':    
    output_dir= 'mnt/analyticsvideo/DensePoseData/market1501'
    images_dir = 'mnt/analyticsvideo/DensePoseData/market1501/query'
    ica_train= "/n/analyticsdata/polyaxon/data"
    ica_train_json = "/n/analyticsdata/polyaxon/data/ICA/ICA_train.json"
    ica_train_out = '../Axis/ICATrain'
    Axis_lobby_train = "/n/analyticsdata/polyaxon/data"
    Axis_lobby_train_json = "/n/analyticsdata/polyaxon/data/AxisLobby2017/AxisLobby2017_train.json"
    Axis_lobby_train_out = '../Axis/AxisLobby2017Train'
    create_segmap(output_dir, images_dir)

    