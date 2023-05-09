import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import sys
import logging

from net import EmbedNetwork, EmbedNetwork_DSAG
from dataWrapper import Wrapper

from utils_reid import pdist_np as pdist
from utils_reid import save_model, pdist_arc


torch.multiprocessing.set_sharing_strategy('file_system')

def embed(name, load_path, type = None, load_path_dense = None):

    # logging
    FORMAT = '%(levelname)s %(filename)s:%(lineno)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    # restore model   


    logger.info('restoring model')

    model = torch.load('res2/' + name + '.pkl')

    net = EmbedNetwork().cuda()
    net.load_state_dict(model["net"])
    net = nn.DataParallel(net)

    if not load_path_dense == None:
        net_DSAG = EmbedNetwork_DSAG().cuda()
        net_DSAG.load_state_dict(model["net_DSAG"])
        net_DSAG = nn.DataParallel(net_DSAG)
        net_DSAG.eval()

    net.eval()

    # load gallery dataset
    batchsize = 32
    if load_path_dense == None:
        ds = Wrapper(load_path, is_train = False)
        dl = DataLoader(ds, batch_size = batchsize, drop_last = False, num_workers = 4)
    else:
        ds = Wrapper(load_path, data_path_dense = load_path_dense, is_train = False)
        dl = DataLoader(ds, batch_size = batchsize, drop_last = False, num_workers = 4)

    # embedding samples
    logger.info('start embedding')
    all_iter_nums = len(ds) // batchsize + 1
    embeddingsGlobal = []
    embeddingsLocal = []
    label_ids = []
    label_cams = []

    if load_path_dense == None:
        for it, (img, lb_id, lb_cam) in enumerate(dl):
            print('\r=======>  processing iter {} / {}'.format(it, all_iter_nums),
                    end = '', flush = True)
            label_ids.append(lb_id)
            label_cams.append(lb_cam)
            embdsLocal = []
            embdsGlobal = []
            for im in img: 
                im = im.cuda()
                G, L = net(im) 
                G = G.detach().cpu().numpy()
                L = L.detach().cpu().numpy()

                embdsGlobal.append(G)
                embdsLocal.append(L)
            
            embedGlobal = sum(embdsGlobal) / len(embdsGlobal)
            embedLocal = sum(embdsLocal) / len(embdsLocal)
            embeddingsGlobal.append(embedGlobal)
            embeddingsLocal.append(embedLocal)
    else:
        for it, (img, img_dense, lb_id, lb_cam) in enumerate(dl):
            print('\r=======>  processing iter {} / {}'.format(it, all_iter_nums),
                    end = '', flush = True)
            label_ids.append(lb_id)
            label_cams.append(lb_cam)
            embdsLocal = []
            embdsGlobal = []
            for i in range(len(img)):
                im = img[i]
                im_dense = img_dense[i]
                im = im.cuda()
                im_dense = im_dense.cuda()
                G, L = net(im) 
                G = G.detach().cpu().numpy()
                L = L.detach().cpu().numpy()

                im_dense = im_dense.cuda()
                
                G_dense, L_dense = net_DSAG(im_dense)
                G_dense = G_dense.detach().cpu().numpy()
                L_dense = L_dense.detach().cpu().numpy()

                G = G+G_dense
                L = L+L_dense

                embdsGlobal.append(G)
                embdsLocal.append(L)
            
            embedGlobal = sum(embdsGlobal) / len(embdsGlobal)
            embedLocal = sum(embdsLocal) / len(embdsLocal)
            embeddingsGlobal.append(embedGlobal)
            embeddingsLocal.append(embedLocal)

    print('  ...   completed')

    embeddingsGlobal = np.vstack(embeddingsGlobal)
    embeddingsLocal = np.vstack(embeddingsLocal)
    label_ids = np.hstack(label_ids)
    label_cams = np.hstack(label_cams)

    if type == "query":
        label_cams = [1]*100
    
    # dump results
    logger.info('dump embeddings')
    embd_res = {'embeddingsGlobal': embeddingsGlobal, 'embeddingsLocal': embeddingsLocal, 'label_ids': label_ids, 'label_cams': label_cams}
    model["embd_" + type] = embd_res
    save_model(model, "res2/", name + ".pkl")

    logger.info('embedding finished')
    return torch.tensor(embeddingsGlobal), torch.tensor(embeddingsLocal), torch.tensor(label_ids)


def evaluate(name, local, arc):
    cmc_rank = 1

    # logging
    FORMAT = '%(levelname)s %(filename)s:%(lineno)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    # load embeddings

    model = torch.load('res2/' + name + '.pkl')

    logger.info('loading gallery embeddings')
    gallery_dict = model["embd_gallery"]
    embGlobal, embLocal, lb_ids, lb_cams = gallery_dict['embeddingsGlobal'], gallery_dict['embeddingsLocal'], gallery_dict['label_ids'], gallery_dict['label_cams']
    
    logger.info('loading query embeddings')
    query_dict = model["embd_query"]
    embGlobal_query, embLocal_query, lb_ids_query, lb_cams_query = query_dict['embeddingsGlobal'], query_dict['embeddingsLocal'], query_dict['label_ids'], query_dict['label_cams']

    # compute and clean distance matrix
    if local:
        embGallery = np.concatenate((embGlobal,embLocal),1)
        embQuery = np.concatenate((embGlobal_query, embLocal_query),1)
    else:
        embGallery = embGlobal
        embQuery = embGlobal_query

    if arc:
        dist_mtx = pdist_arc(embQuery, embGallery)
    else:
        dist_mtx = pdist(embQuery, embGallery) 
    n_q, n_g = dist_mtx.shape
    indices = np.argsort(dist_mtx, axis = 1)
    matches = lb_ids[indices] == lb_ids_query[:, np.newaxis]
    matches = matches.astype(np.int32)
    all_aps = []
    all_cmcs = []
    logger.info('starting evaluating ...')
    for qidx in range(n_q): #tqdm(range(n_q)):
        qpid = lb_ids_query[qidx]
        qcam = lb_cams_query[qidx]

        order = indices[qidx]
        pid_diff = lb_ids[order] != qpid
        cam_diff = lb_cams[order] != qcam
        useful = lb_ids[order] != -1
        keep = np.logical_or(pid_diff, cam_diff)
        keep = np.logical_and(keep, useful)
        match = matches[qidx][keep]

        if not np.any(match): continue
        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        all_cmcs.append(cmc[:cmc_rank])

        num_real = match.sum()
        match_cum = match.cumsum()
        match_cum = [el / (1.0 + i) for i, el in enumerate(match_cum)]
        match_cum = np.array(match_cum) * match
        ap = match_cum.sum() / num_real
        all_aps.append(ap)

    assert len(all_aps) > 0, "NO QUERY MATCHED"
    mAP = sum(all_aps) / len(all_aps)
    all_cmcs = np.array(all_cmcs, dtype = np.float32)
    cmc = np.mean(all_cmcs, axis = 0)

    logger.info('mAP is: {}, cmc is: {}'.format(mAP, cmc))

    model["mAP"] = mAP
    model["cmc"] = cmc 

    logger.info("Saving model with mAP and cmc...")
    save_model(model, "res2/", name + ".pkl")
    logger.info("Saving successfull!")
    return mAP, cmc


if __name__ == '__main__':
    load_path = '/mnt/analyticsvideo/DensePoseData/market1501/bounding_box_test_no0'
    load_path_dense = '/mnt/analyticsvideo/DensePoseData/market1501/uv_maps_test'
    load_path2 = '/mnt/analyticsvideo/DensePoseData/market1501/query'
    load_path2_dense = '/mnt/analyticsvideo/DensePoseData/market1501/uv_maps_query'

    #load_path = '/mnt/analyticsvideo/DensePoseData/market1501/bbox_train_gallery'
    #load_path2 = '/mnt/analyticsvideo/DensePoseData/market1501/bbox_train_query'

    # name = 'Triceratops3'
    # for mod in ['full']:
    #     embed(name + '_' + mod,load_path, type='gallery')
    #     embed(name + '_' + mod, load_path2, type='query')
    #     mAP, cmc = evaluate(name + '_' + mod, True, False)
    #     print('For {} mAP is: {}, cmc is: {}'.format(name, mAP, cmc))

    # name = 'Triceratops5'
    # for mod in ['full']:
    #     embed(name + '_' + mod,load_path, type='gallery')
    #     embed(name + '_' + mod, load_path2, type='query')
    #     mAP, cmc = evaluate(name + '_' + mod, True, False)
    #     print('For {} mAP is: {}, cmc is: {}'.format(name, mAP, cmc))
    

    names = ['Triceratops3_full']#['rn50_trip_baseline2_full', 'MF_trip_full', 'MFDSAG_trip_full', 'triceratops_full', 'rn50_trip_NORA_lowestloss', 'MF_trip_NORA_full', 'MFDSAG_trip_NORA_full', 'Triceratops_NORE_full']
    paths = [['/mnt/analyticsvideo/DensePoseData/chaxis/chaxis100_test', '/mnt/analyticsvideo/DensePoseData/chaxis/chaxis100re_query'], ['/mnt/analyticsvideo/DensePoseData/chaxis/chaxis100re_test', '/mnt/analyticsvideo/DensePoseData/chaxis/chaxis100re_query'], ['/mnt/analyticsvideo/DensePoseData/market1501/market100_test', '/mnt/analyticsvideo/DensePoseData/market1501/market100_query'], ['/mnt/analyticsvideo/DensePoseData/market1501/market100_test', '/mnt/analyticsvideo/DensePoseData/market1501/market100re_query']]
    for name in names:
        print("")
        for path in paths:
            embed(name, path[0], type = 'gallery')
            embed(name, path[1], type = 'query')
            if name == names[0] or name == names[4]:
                mAP, cmc = evaluate(name, False, False)
            else:
                mAP, cmc = evaluate(name, True, False)

            if path[0] == paths[0][0]:
                print("{} gets mAP {} on non-occluded chaxis".format(name, mAP))
            elif path[0] == paths[1][0]:
                print("{} gets mAP {} on occluded chaxis".format(name, mAP))
            elif path[1] == paths[2][1]:
                print("{} gets mAP {} on non-occluded market".format(name, mAP))
            else:
                print("{} gets mAP {} on occluded market".format(name, mAP))