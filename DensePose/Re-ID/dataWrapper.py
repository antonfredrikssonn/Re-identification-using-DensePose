import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from random_erasing import RandomErasing


class Wrapper(Dataset):
    '''
    A wrapper for our datasets
    '''
    def __init__(self, data_path, data_path_dense = None, is_train = True, ran_er = False,  *args, **kwargs):
        super(Wrapper, self).__init__(*args, **kwargs)
        self.is_train = is_train
        self.data_path = data_path
        self.data_path_dense = data_path_dense
        self.imgs = sorted(os.listdir(data_path))
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.jpg']
        self.imgs_dense = sorted(os.listdir(data_path_dense))
        self.imgs_dense = [el for el in self.imgs_dense if os.path.splitext(el)[1] == '.jpg']
        self.lb_ids = [int(el.split('_')[0]) for el in self.imgs]
        self.lb_cams = [int(el.split('_')[1][1]) for el in self.imgs]
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]
        self.imgs_dense = [os.path.join(data_path_dense, el) for el in self.imgs_dense]

        if is_train:
            if ran_er:
                self.trans = transforms.Compose([
                    transforms.Resize((288, 144)),
                    transforms.RandomCrop((256, 128)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
                    RandomErasing(0.5, mean=[0.0, 0.0, 0.0])
                ])
            else:
                self.trans = transforms.Compose([
                    transforms.Resize((288, 144)),
                    transforms.RandomCrop((256, 128)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
                ])

            self.trans_dense = transforms.Compose([
                transforms.Resize((128, 192)),
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
            ])
        else:
            self.trans_tuple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
                ])
            self.Lambda = transforms.Lambda(
                lambda crops: [self.trans_tuple(crop) for crop in crops])
            self.trans = transforms.Compose([
                transforms.Resize((288, 144)),
                transforms.TenCrop((256, 128)),
                self.Lambda,
            ])

            self.trans_tuple_dense = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
                ])
            self.Lambda_dense = transforms.Lambda(
                lambda crops: [self.trans_tuple_dense(crop) for crop in crops])
            self.trans_dense = transforms.Compose([
                transforms.Resize((128, 192)),
                transforms.TenCrop((128, 192)),
                self.Lambda_dense,
            ])

        # useful for sampler
        self.lb_img_dict = dict()
        self.lb_ids_uniq = set(self.lb_ids)

        lb_array = np.array(self.lb_ids)
        for lb in self.lb_ids_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = self.trans(img)
        if self.data_path_dense != None:
            img_dense = Image.open(self.imgs_dense[idx])
            img_dense = self.trans_dense(img_dense)
            return img, img_dense, self.lb_ids[idx], self.lb_cams[idx]
        else:
            return img, self.lb_ids[idx], self.lb_cams[idx]
