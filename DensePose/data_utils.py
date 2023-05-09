import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from random_erasing import RandomErasing
import numpy as np
import shutil

def random_erase(impath, outpath):

    MEAN = np.array([0.486, 0.459, 0.408])
    STD = np.array([0.229, 0.224, 0.225])

    trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN,STD),
                    RandomErasing(1, mean=[0.0, 0.0, 0.0]),
                    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / STD),
                    transforms.Normalize(mean=-MEAN, std=[1.0, 1.0, 1.0]),
                    transforms.ToPILImage(),
                ])

    Path(outpath).mkdir(exist_ok=True, parents=True)  

    images = sorted(os.listdir(impath))
    for image in images:
        f = os.path.join(impath, image)
        lb_ids = image.split('_')[0]    
        
        imnr = image.split('_')[2][0:4]
        
        store = os.path.join(outpath,lb_ids +'_c0_0000.jpg')
        store2 = os.path.join(outpath,lb_ids +'_c1_0001.jpg')
        print(store,store2)
        im = Image.open(f)
        if imnr== '0000':
            im = trans(im)
            im.save(store)
        else:
            im.save(store2)

# assign directory
def reshapeMarket(directory):

	# iterate over files in
	# that directory

	for id in os.listdir(directory):
		dir = os.path.join(directory,id, "uv_maps")
		for uvmap in os.listdir(dir):
			f = os.path.join(dir, uvmap)
			if os.path.isfile(f):
				dir2 = os.path.join(directory,"uv_maps")
				if not os.path.isdir(dir2):
					Path(dir2).mkdir(exist_ok=True, parents=True)  
				new_file = os.path.join(dir2,uvmap)
				shutil.copyfile(f,new_file)

def reshapeAxis(directory,idx):
	idx = 3000
	for id in os.listdir(directory):
		dir = os.path.join(directory,id, "uv_maps")
		for uvmap in os.listdir(dir):
			f = os.path.join(dir, uvmap)
			if os.path.isfile(f):
				dir2 = os.path.join(directory,"uv_maps")
				if not os.path.isdir(dir2):
					Path(dir2).mkdir(exist_ok=True, parents=True)  
				new_file = os.path.join(dir2,f"{idx}_"+uvmap)
				shutil.copyfile(f,new_file)
		idx += 1
	print(idx)
	
def split_train_val(impath, outpath):
    i = 0
    id_old = -10
    for image in os.listdir(impath):
        print(image)
        f = os.path.join(impath, image)
        val = os.path.join(outpath + "_val")
        tra = os.path.join(outpath + "_tra")
        
        Path(tra).mkdir(exist_ok=True, parents=True)  
        Path(val).mkdir(exist_ok=True, parents=True)  
        
        if os.path.isfile(f):
            id = image.split('_')[0]
            if id == id_old:
                i += 1
            else:     
                i = 0
                id_old = id
            if i == 4 or i == 5:
                shutil.copy(f,val)
            else:
                shutil.copy(f,tra)
def split_train_val2(impath, outpath):
    i = 0
    id_old = -10
    images = sorted(os.listdir(impath))
    for image in images:
        
        f = os.path.join(impath, image)
        val = os.path.join(outpath + "_val")
        tra = os.path.join(outpath + "_tra")
        
        Path(tra).mkdir(exist_ok=True, parents=True)  
        Path(val).mkdir(exist_ok=True, parents=True)  
        
        if os.path.isfile(f):
            id = image.split('_')[0]
            if id == id_old:
                i += 1
            else:     
                i = 0
                id_old = id
            if i == 0:
                image0 = f
            elif i == 1:
                image1 = f        
            elif i in range(2,5):
                shutil.copy(f,tra)
            elif i == 6:
                shutil.copy(image0,val)
                shutil.copy(image1,val)
                shutil.copy(f,tra)    
            else:
                shutil.copy(f,tra)    

def split_query_gallery(impath, outpath):
    i = 0
    id_old = -10
    images = sorted(os.listdir(impath))
    for image in images:
        
        f = os.path.join(impath, image)
        val = os.path.join(outpath + "_train_query")
        tra = os.path.join(outpath + "_train_gallery")
        
        Path(tra).mkdir(exist_ok=True, parents=True)  
        Path(val).mkdir(exist_ok=True, parents=True)  
        
        if os.path.isfile(f):
            id = image.split('_')[0]
            if id == id_old:
                i += 1
            else:     
                i = 0
                id_old = id
            if i == 0 or i == 1:
                shutil.copy(f,val)
            else:
                shutil.copy(f,tra)    