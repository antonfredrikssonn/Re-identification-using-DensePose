import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from net import EmbedNetwork, Classifier
from loss import TripletLoss, CenterLoss, CentroidTripletLoss
from triplet_selector import BatchHardTripletSelector
from batch_sampler import BatchSampler
from dataWrapper import Wrapper
from optimizer import AdamOptimWrapper
from logger import logger
from utils_reid import save_model
from pytorch_metric_learning import losses
import math

def loadNetwork(net_init):
    logger.info('setting up backbone model and loss')
    net = EmbedNetwork().cuda()
    classifier1 = Classifier().cuda()
    classifier2 = Classifier().cuda()
    if not net_init == None:
        model = torch.load("res2/" + net_init + ".pkl")
        net.load_state_dict(model["net"])
        classifier1.load_state_dict(model["classifier1"])
        classifier2.load_state_dict(model["classifier2"])
    net = nn.DataParallel(net)
    classifier1 = nn.DataParallel(classifier1)
    classifier2 = nn.DataParallel(classifier2)
    return [net, classifier1, classifier2]

def createOptimizer(model):
    optimizer = {
    "net": AdamOptimWrapper(model[0].parameters(), lr = 5e-4, wd = 0, t0 = 15000, t1 = 25000),
    "classifier1": AdamOptimWrapper(model[1].parameters(), lr = 5e-4, wd = 0, t0 = 15000, t1 = 25000),
    "classifier2": AdamOptimWrapper(model[2].parameters(), lr = 5e-4, wd = 0, t0 = 15000, t1 = 25000)
    }
    return optimizer

def getDataloader(load_path, ran_er = False, is_train = True, n_classes = 18, n_num = 4):
    ds = Wrapper(load_path, is_train=is_train, ran_er=ran_er)
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
        "center": CenterLoss().cuda(),
        "centroid": CentroidTripletLoss().cuda()
    }
    return loss_dict

def getEmbds(imgs, model):
    mainGlobalEmbds, mainLocalEmbds = model[0](imgs)
    return [mainGlobalEmbds, mainLocalEmbds]

def updateLoss(bools, loss_funcs, embds, lbs, selector, local, classifiers):
    loss = 0
    losses = {
        "arc": 0,
        "trip": 0,
        "id": 0,
        "center": 0,
        "centroid": 0
    }

    if bools[0][0]:
        global_main_arcface_loss = loss_funcs["arc"](embds[0],lbs)
        local_main_arcface_loss = loss_funcs["arc"](embds[1],lbs)
        arc_main_loss = bools[0][2]*(global_main_arcface_loss+local_main_arcface_loss) if local else bools[0][2]*(global_main_arcface_loss)
        loss += arc_main_loss
        losses["arc"] += arc_main_loss
    if bools[1][0]:
        anchor, positives, negatives = selector(embds[0], lbs)
        trip_main_global_loss = loss_funcs["trip"](anchor, positives, negatives)
        anchor, positives, negatives = selector(embds[1], lbs)
        trip_main_local_loss = loss_funcs["trip"](anchor, positives, negatives)
        trip_main_loss =  bools[1][2]*(trip_main_global_loss+trip_main_local_loss) if local else bools[1][2]*(trip_main_global_loss)
        loss += trip_main_loss
        losses["trip"] += trip_main_loss
    if bools[2][0]:
        global_main_ID_loss = loss_funcs["id"](classifiers[0](embds[0]), lbs)
        local_main_ID_loss = loss_funcs["id"](classifiers[1](embds[1]), lbs)
        ID_main_loss = bools[2][2]*(global_main_ID_loss + local_main_ID_loss) if local else bools[2][2]*(global_main_ID_loss) 
        loss += ID_main_loss
        losses["id"] += ID_main_loss
    if bools[3][0]:
        global_main_center_loss = loss_funcs["center"](embds[0], lbs)
        local_main_center_loss = loss_funcs["center"](embds[1], lbs)
        center_main_loss = bools[3][2]*(global_main_center_loss + local_main_center_loss) if local else bools[3][2]*(global_main_center_loss)
        loss += center_main_loss
        losses["center"] += center_main_loss
    if bools[4][0]:
        embds[0] = embds[0]/torch.norm(embds[0])
        embds[1] = embds[1]/torch.norm(embds[1])
        global_main_centroid_loss = loss_funcs["centroid"](embds[0], lbs)
        local_main_centroid_loss = loss_funcs["centroid"](embds[1], lbs)
        centroid_main_loss = bools[4][2]*(global_main_centroid_loss + local_main_centroid_loss) if local else bools[4][2]*(global_main_centroid_loss)
        loss += centroid_main_loss
        losses["centroid"] += centroid_main_loss
    return loss, losses

def trainMain(name, load_path, bools, net_init, local, num_ep, ran_er):
    # Setup
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists('./res2'): os.makedirs('./res2')

    # Initialize model
    logger.info('setting up backbone model and loss')
    model = loadNetwork(net_init)

    # Loss design
    loss_funcs = defineLoss()

    # Optimizer
    logger.info('creating optimizer')
    model_name = ["net", "classifier1", "classifier2"]
    optimizer = createOptimizer(model)


    # Train dataloader
    logger.info("creating dataloaders")
    selector = BatchHardTripletSelector()
    n_classes, n_num = 18, 4
    batch_size = n_classes*n_num
    if bools[0][0] or bools[0][1] or bools[2][0] or bools[2][1] or bools[3][0] or bools[3][1]:
        diter, dl, ds_len = getDataloader(load_path + 'bbox_tra', n_classes=n_classes, n_num=n_num, ran_er=ran_er)
        diter_val, dl_val, _ = getDataloader(load_path + 'bbox_val', n_classes=30, n_num=2)
        diter_train, dl_train, _ = getDataloader(load_path + 'bbox_tra', n_classes=30, n_num=2)
    else:
        diter, dl, ds_len = getDataloader(load_path + 'bounding_box_train', n_classes=n_classes, n_num=n_num, ran_er=ran_er)
        diter_train, dl_train, _ = getDataloader(load_path + 'bounding_box_train', n_classes=30, n_num=2)
        diter_val, dl_val, _ = getDataloader(load_path + 'bounding_box_test', n_classes=30, n_num=2)


    # Train
    logger.info('start training ...')
    loss_avg = []
    losses_val = []
    losses_train = []

    if bools[0][0] or bools[0][1]:
        arc_loss_avg = []
    if bools[1][0] or bools[1][1]:
        trip_loss_avg = []
    if bools[2][0] or bools[2][1]:
        id_loss_avg = []
    if bools[3][0] or bools[3][1]:
        center_loss_avg = []
    if bools[4][0] or bools[4][1]:
        centroid_loss_avg = []

    count = 0
    epochs = 0
    val_it = 20
    itPerEpoch = ds_len // batch_size
    lowest_loss = 1e10
    best_epoch = 0
    epochs_counter = []
    t_start = time.time()
    while True:
        try:
            imgs, lbs, _ = next(diter)
        except StopIteration:
            diter = iter(dl)
            imgs, lbs, _ = next(diter)
        imgs, lbs = imgs.cuda(), lbs.cuda()

        for m in model:
            m.train()

        embds = getEmbds(imgs, model)
     
        loss, _ = updateLoss(bools, loss_funcs, embds, lbs, selector, local, [model[1], model[2]])
        
        # Update model
        for m in model:
            m.zero_grad()

        for m in model_name:
            optimizer[m].zero_grad()

        loss.backward()

        for m in model_name:
            optimizer[m].step()

        loss_avg.append(loss.detach().cpu().numpy())
        if count % 20 == 0 and count != 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            t_end = time.time()
            time_interval = t_end - t_start
            logger.info('iter: {}, loss: {:4f}, time: {:3f}'.format(count, loss_avg, time_interval))
            loss_avg = []
            t_start = t_end

        count += 1
        if count % itPerEpoch == 0:
            epochs += 1
            if epochs % 1 == 0:
                for m in model:
                    m.eval()
                test_start = time.time()

                loss_val_avg, loss_train_avg = 0, 0

                if bools[0][0] or bools[0][1]:
                    arc_avg = 0
                if bools[1][0] or bools[1][1]:
                    trip_avg = 0
                if bools[2][0] or bools[2][1]:
                    id_avg = 0
                if bools[3][0] or bools[3][1]:
                    center_avg = 0
                if bools[4][0] or bools[4][1]:
                    centroid_avg = 0

                for _ in range(val_it):
                    # FOR TRAIN
                    try:
                        imgs, lbs, _ = next(diter_train)
                    except StopIteration:
                        diter_train = iter(dl_train)
                        imgs, lbs, _ = next(diter_train)
                    imgs, lbs = imgs.cuda(), lbs.cuda()
                    embds = getEmbds(imgs, model)
                    loss_train, losses = updateLoss(bools, loss_funcs, embds, lbs, selector, local, [model[1], model[2]])
                    loss_train = loss_train.detach().cpu().numpy()
                    loss_train_avg += loss_train/val_it


                    if bools[0][0] or bools[0][1]:
                        arc_avg += losses["arc"].detach().cpu().numpy()/val_it
                    if bools[1][0] or bools[1][1]:
                        trip_avg += losses["trip"].detach().cpu().numpy()/val_it
                    if bools[2][0] or bools[2][1]:
                        id_avg += losses["id"].detach().cpu().numpy()/val_it
                    if bools[3][0] or bools[3][1]:
                        center_avg += losses["center"].detach().cpu().numpy()/val_it
                    if bools[4][0] or bools[4][1]:
                        centroid_avg += losses["centroid"].detach().cpu().numpy()/val_it

                    # FOR VAL
                    try:
                        imgs, lbs, _ = next(diter_val)
                    except StopIteration:
                        diter_val = iter(dl_val)
                        imgs, lbs, _ = next(diter_val)
                    imgs, lbs = imgs.cuda(), lbs.cuda()
                    embds = getEmbds(imgs, model)
                    loss_val, _ = updateLoss(bools, loss_funcs, embds, lbs, selector, local, [model[1], model[2]])
                    loss_val = loss_val.detach().cpu().numpy()
                    loss_val_avg += loss_val/val_it

                test_end = time.time()
                time_interval = test_end - test_start
                logger.info('loss_tra: {}, loss_val: {}, time: {}'.format(loss_train_avg, loss_val_avg, time_interval))

                losses_val.append(loss_val_avg)
                losses_train.append(loss_train_avg)
                epochs_counter.append(epochs)

                if bools[0][0] or bools[0][1]:
                    arc_loss_avg.append(arc_avg)
                if bools[1][0] or bools[1][1]:
                    trip_loss_avg.append(trip_avg)
                if bools[2][0] or bools[2][1]:
                    id_loss_avg.append(id_avg)
                if bools[3][0] or bools[3][1]:
                    center_loss_avg.append(center_avg)
                if bools[4][0] or bools[4][1]:
                    centroid_loss_avg.append(centroid_avg)

                if loss_val_avg < lowest_loss:
                    state = {
                        "epoch": epochs,
                        "net": model[0].module.state_dict(),
                        "classifier1": model[1].module.state_dict(),
                        "classifier2": model[2].module.state_dict()
                    }

                    save_model(state, 'res2/', name + '_lowestloss.pkl')
                    lowest_loss = loss_val_avg
                    best_epoch = epochs           

        if epochs == num_ep:
            break

    losses = [None] * 5
    if bools[0][0] or bools[0][1]:
        losses[0] = arc_loss_avg
    if bools[1][0] or bools[1][1]:
        losses[1] = trip_loss_avg
    if bools[2][0] or bools[2][1]:
        losses[2] = id_loss_avg
    if bools[3][0] or bools[3][1]:
        losses[3] = center_loss_avg
    if bools[4][0] or bools[4][1]:
        losses[4] = centroid_loss_avg
    logger.info('Lowest loss model achieved after {} epochs with loss {}'.format(best_epoch, lowest_loss))
    # dump model
    logger.info('saving trained model')
    state = {
        "epoch": epochs_counter,
        "losses_val": losses_val,
        "losses_train": losses_train,
        "losses": losses,
        "net": model[0].module.state_dict(),
        "classifier1": model[1].module.state_dict(),
        "classifier2": model[2].module.state_dict()
    }
    save_model(state, 'res2/', name + '_full.pkl')

    logger.info('everything finished')


    return epochs_counter, losses_val, losses_train, losses

