import torch
import os
import time
from triplet_selector import BatchHardTripletSelector
from logger import logger
from utils_reid import loadNetwork, save_model, createOptimizer, getDataloader, getEmbds, updateLoss, defineLoss

def trainFull(name, load_path, bools, net_init, num_ep, ran_er):
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
    model_name = ["net", "net_DSAG", "classifier1", "classifier2", "classifier3", "classifier4"]
    optimizer = createOptimizer(model)


    # Train dataloader
    logger.info("creating dataloaders")
    selector = BatchHardTripletSelector()
    n_classes, n_num = 18, 4
    batch_size = n_classes*n_num
    if bools[0][0] or bools[0][1] or bools[2][0] or bools[2][1] or bools[3][0] or bools[3][1]:
        diter, dl, ds_len = getDataloader(load_path + 'bbox_tra',load_path+'uv_tra', n_classes=n_classes, n_num=n_num, ran_er=ran_er)
        diter_val, dl_val, _ = getDataloader(load_path + 'bbox_val',load_path+'uv_val', n_classes=30, n_num=2)
        diter_train, dl_train, _ = getDataloader(load_path + 'bbox_tra',load_path+'uv_tra', n_classes=30, n_num=2)
    else:
        diter, dl, ds_len = getDataloader(load_path + 'bounding_box_train',load_path+'uv_maps_train', n_classes=n_classes, n_num=n_num, ran_er=ran_er)
        diter_train, dl_train, _ = getDataloader(load_path + 'bounding_box_train', load_path+'uv_maps_train', n_classes=30, n_num=2)
        diter_val, dl_val, _ = getDataloader(load_path + 'bounding_box_test', load_path+'uv_maps_test', n_classes=30, n_num=2)

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
            imgs, imgs_dense, lbs, _ = next(diter)
        except StopIteration:
            diter = iter(dl)
            imgs, imgs_dense, lbs, _ = next(diter)
        imgs, imgs_dense, lbs = imgs.cuda(), imgs_dense.cuda(), lbs.cuda()

        for m in model:
            m.train()


        embds = getEmbds(imgs, imgs_dense, model)

        loss, _ = updateLoss(bools, loss_funcs, embds, lbs, selector,[model[2], model[3],model[4], model[5]])

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
                        imgs, imgs_dense, lbs, _ = next(diter_train)
                    except StopIteration:
                        diter_train = iter(dl_train)
                        imgs, imgs_dense, lbs, _ = next(diter_train)
                    imgs, imgs_dense, lbs = imgs.cuda(), imgs_dense.cuda(), lbs.cuda()
                    embds = getEmbds(imgs, imgs_dense, model)
                    loss_train, losses = updateLoss(bools, loss_funcs, embds, lbs, selector,[model[2], model[3],model[4], model[5]])
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
                        imgs, imgs_dense, lbs, _ = next(diter_val)
                    except StopIteration:
                        diter_val = iter(dl_val)
                        imgs, imgs_dense, lbs, _ = next(diter_val)
                    imgs, imgs_dense, lbs = imgs.cuda(), imgs_dense.cuda(), lbs.cuda()
                    embds = getEmbds(imgs, imgs_dense, model)
                    loss_val, _ = updateLoss(bools, loss_funcs, embds, lbs, selector, [model[2], model[3],model[4], model[5]])
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
                        "net_DSAG": model[1].module.state_dict(),
                        "classifier1": model[2].module.state_dict(),
                        "classifier2": model[3].module.state_dict(),
                        "classifier3": model[4].module.state_dict(),
                        "classifier4": model[5].module.state_dict()
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
        "net_DSAG": model[1].module.state_dict(),
        "classifier1": model[2].module.state_dict(),
        "classifier2": model[3].module.state_dict(),
        "classifier3": model[4].module.state_dict(),
        "classifier4": model[5].module.state_dict()
    }
    save_model(state, 'res2/', name + '_full.pkl')

    logger.info('everything finished')


    return epochs_counter, losses_val, losses_train, losses
