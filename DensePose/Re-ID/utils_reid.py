import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from net import EmbedNetwork, EmbedNetwork_DSAG, Classifier 
from loss import TripletLoss, CenterLoss, CentroidTripletLoss
from batch_sampler import BatchSampler
from dataWrapper import Wrapper
from optimizer import AdamOptimWrapper
from logger import logger
from pytorch_metric_learning import losses
import math

def loadNetwork(net_init = None):
    logger.info('setting up backbone model and loss')
    net = EmbedNetwork().cuda()
    net_DSAG = EmbedNetwork_DSAG().cuda()
    classifier1 = Classifier(in_features=128).cuda()
    classifier2 = Classifier(in_features=128).cuda()
    classifier3 = Classifier(in_features=128).cuda()
    classifier4 = Classifier(in_features=128).cuda()
    if not net_init == None:
        print(net_init)
        model = torch.load("res2/" + net_init + ".pkl")
        net.load_state_dict(model["net"])
        if not net_init == 'iguanadon_full':
            net_DSAG.load_state_dict(model["net_DSAG"])
        classifier1.load_state_dict(model["classifier1"])
        classifier2.load_state_dict(model["classifier2"])
        if not net_init == 'iguanadon_full':
            classifier3.load_state_dict(model["classifier3"])
            classifier4.load_state_dict(model["classifier4"])
    net = nn.DataParallel(net)
    net_DSAG = nn.DataParallel(net_DSAG)
    classifier1 = nn.DataParallel(classifier1)
    classifier2 = nn.DataParallel(classifier2)
    classifier3 = nn.DataParallel(classifier3)
    classifier4 = nn.DataParallel(classifier4)
    return [net, net_DSAG, classifier1, classifier2, classifier3, classifier4]

def createOptimizer(model):
    optimizer = {
    "net": AdamOptimWrapper(model[0].parameters(), lr = 5e-4, wd = 0, t0 = 15000, t1 = 25000),
    "net_DSAG": AdamOptimWrapper(model[1].parameters(), lr = 5e-4, wd = 0, t0 = 15000, t1 = 25000),
    "classifier1": AdamOptimWrapper(model[2].parameters(), lr = 5e-4, wd = 0, t0 = 15000, t1 = 25000),
    "classifier2": AdamOptimWrapper(model[3].parameters(), lr = 5e-4, wd = 0, t0 = 15000, t1 = 25000),
    "classifier3": AdamOptimWrapper(model[4].parameters(), lr = 5e-4, wd = 0, t0 = 15000, t1 = 25000),
    "classifier4": AdamOptimWrapper(model[5].parameters(), lr = 5e-4, wd = 0, t0 = 15000, t1 = 25000)
    }
    return optimizer

def getDataloader(load_path, load_path_dense, ran_er = False, is_train = True, n_classes = 18, n_num = 4):
    ds = Wrapper(load_path, data_path_dense=load_path_dense, ran_er=ran_er, is_train=is_train)
    sampler = BatchSampler(ds, n_classes, n_num)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    diter = iter(dl)
    return diter, dl, len(ds.imgs)

def defineLoss():
    num_classes=1502
    embedding_size=128
    loss_dict = {
        "arc": losses.ArcFaceLoss(num_classes, embedding_size, margin=0.4*180/math.pi, scale=50).cuda(), # 28.6 and 64
        "trip": TripletLoss(margin = 0.3).cuda(),
        "id": nn.CrossEntropyLoss().cuda(),
        "center": CenterLoss(feat_dim=128).cuda(),
        "centroid": CentroidTripletLoss().cuda()
    }
    return loss_dict

def getEmbds(imgs, imgs_dense, model):
    mainGlobalEmbds, mainLocalEmbds = model[0](imgs)
    DSAGGlobalEmbds, DSAGLocalEmbds = model[1](imgs_dense)
    globalEmbds = mainGlobalEmbds + DSAGGlobalEmbds
    localEmbds = mainLocalEmbds + DSAGLocalEmbds
    return [mainGlobalEmbds, mainLocalEmbds, globalEmbds, localEmbds]

def updateLoss(bools, loss_funcs, embds, lbs, selector, classifiers):
    loss = 0
    losses = {
        "arc": 0,
        "trip": 0,
        "id": 0,
        "center": 0,
        "centroid": 0
    }

    if bools[0][0] or bools[0][1]:
        if bools[0][0]:
            global_main_arcface_loss = loss_funcs["arc"](embds[0],lbs)
            local_main_arcface_loss = loss_funcs["arc"](embds[1],lbs)
            arc_main_loss = bools[0][2]*(global_main_arcface_loss+local_main_arcface_loss)
            loss += arc_main_loss
            losses["arc"] += arc_main_loss
        if bools[0][1]:
            global_arcface_loss = loss_funcs["arc"](embds[2], lbs)
            local_arcface_loss = loss_funcs["arc"](embds[3],lbs)
            arc_fused_loss = bools[0][3]*(global_arcface_loss+local_arcface_loss)
            loss += arc_fused_loss
            losses["arc"] += arc_fused_loss
    if bools[1][0] or bools[1][1]:
        if bools[1][0]:
            anchor, positives, negatives = selector(embds[0], lbs)
            trip_main_global_loss = loss_funcs["trip"](anchor, positives, negatives)
            anchor, positives, negatives = selector(embds[1], lbs)
            trip_main_local_loss = loss_funcs["trip"](anchor, positives, negatives)
            trip_main_loss =  bools[1][2]*(trip_main_global_loss+trip_main_local_loss)
            loss += trip_main_loss
            losses["trip"] += trip_main_loss
        if bools[1][1]:
            anchor, positives, negatives = selector(embds[2], lbs)
            trip_global_loss = loss_funcs["trip"](anchor, positives, negatives)
            anchor, positives, negatives = selector(embds[3], lbs)
            trip_local_loss = loss_funcs["trip"](anchor, positives, negatives)
            trip_fused_loss = bools[1][3]*(trip_global_loss+trip_local_loss)
            loss += trip_fused_loss
            losses["trip"] += trip_fused_loss
    if bools[2][0] or bools[2][1]:
        if bools[2][0]:
            global_main_ID_loss = loss_funcs["id"](classifiers[0](embds[0]),lbs)
            local_main_ID_loss = loss_funcs["id"](classifiers[1](embds[1]),lbs)
            ID_main_loss = bools[2][2]*(global_main_ID_loss + local_main_ID_loss)
            loss += ID_main_loss
            losses["id"] += ID_main_loss
        if bools[2][1]:
            global_ID_loss = loss_funcs["id"](classifiers[2](embds[2]),lbs)
            local_ID_loss = loss_funcs["id"](classifiers[3](embds[3]),lbs)
            ID_fused_loss = bools[2][3]*(global_ID_loss + local_ID_loss)
            loss += ID_fused_loss
            losses["id"] += ID_fused_loss
    if bools[3][0] or bools[3][1]:
        if bools[3][0]:
            global_main_center_loss = loss_funcs["center"](embds[0], lbs)
            local_main_center_loss = loss_funcs["center"](embds[1], lbs)
            center_main_loss = bools[3][2]*(global_main_center_loss + local_main_center_loss)
            loss += center_main_loss
            losses["center"] += center_main_loss
        if bools[3][1]:
            global_center_loss = loss_funcs["center"](embds[2], lbs)
            local_center_loss = loss_funcs["center"](embds[3], lbs)
            center_fused_loss = bools[3][3]*(global_center_loss + local_center_loss)
            loss += center_fused_loss
            losses["center"] += center_fused_loss
    if bools[4][0] or bools[4][1]:
        if bools[4][0]:
            embds[0] = embds[0]/torch.norm(embds[0])
            embds[1] = embds[1]/torch.norm(embds[1])
            global_main_centroid_loss = loss_funcs["centroid"](embds[0], lbs)
            local_main_centroid_loss = loss_funcs["centroid"](embds[1], lbs)
            centroid_main_loss = bools[4][2]*(global_main_centroid_loss + local_main_centroid_loss)
            loss += centroid_main_loss
            losses["centroid"] += centroid_main_loss
        if bools[4][1]:
            embds[2] = embds[2]/torch.norm(embds[2])
            embds[3] = embds[3]/torch.norm(embds[3])
            global_centroid_loss = loss_funcs["centroid"](embds[2], lbs)
            local_centroid_loss = loss_funcs["centroid"](embds[3], lbs)
            centroid_fused_loss = bools[4][3]*(global_centroid_loss + local_centroid_loss)
            loss += centroid_fused_loss
            losses["centroid"] += centroid_fused_loss
    return loss, losses

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

def pdist_arc(emb1, emb2):
    '''
    compute the arc distance matrix between embeddings1 and embeddings2 using cpu
    '''
    # u*v = |u||v|*cos(theta)
    emb1_abs = np.linalg.norm(emb1, axis = 1)
    emb2_abs = np.linalg.norm(emb2, axis = 1)
    norms = np.outer(emb1_abs,emb2_abs.T)

    dist_mtx = np.matmul(emb1, emb2.T)
    dist_mtx = 1 - dist_mtx / norms
    # dist_mtx = np.absolute(np.arccos(dist_mtx / norms))

    return dist_mtx

def save_model(state, directory="./checkpoints", filename=None):
    if os.path.isdir(directory):
        pkl_filename = os.path.join(directory, filename)
        torch.save(state, pkl_filename)
        #print('Save "{:}" in {:} successful'.format(pkl_filename, directory))
    else:
        print(' "{:}" directory is not exsits!'.format(directory))



if __name__ == "__main__":
    a = np.arange(4*128).reshape(4, 128)
    b = np.arange(10, 10 + 5*128).reshape(5, 128)
    r1 = pdist_np(a, b)
    print(r1.shape)
    print(r1)

    a = torch.Tensor(a)
    b = torch.Tensor(b)
    r2 = pdist_torch(a, b)
    print(r2.shape)
    print(r2)