import os, shutil
from pathlib import Path

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

                
                
split_query_gallery("/mnt/analyticsvideo/DensePoseData/market1501/bounding_box_train", "/mnt/analyticsvideo/DensePoseData/market1501/bbox")
#split_train_val2("/mnt/analyticsvideo/DensePoseData/market1501/uv_maps_train", "/mnt/analyticsvideo/DensePoseData/market1501/uv_maps")

        