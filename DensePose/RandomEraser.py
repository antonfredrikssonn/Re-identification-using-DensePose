
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from random_erasing import RandomErasing
import numpy as np
import cv2

impath = 'chaxis100'
outpath = 'chaxis100re'


mean = (0.486, 0.459, 0.408)
std = (0.229, 0.224, 0.225)

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
