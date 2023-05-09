import torch
import numpy as np
import random
from utils_reid import pdist_torch as pdist


class BatchHardTripletSelector(object):
    '''
    a selector to generate hard batch embeddings from the embedded batch
    '''
    def __init__(self, *args, **kwargs):
        super(BatchHardTripletSelector, self).__init__()

    def __call__(self, embeds, labels, rand = False):
        if rand:
            num = 72
            pos_idxs = []
            neg_idxs = []
            for i in range(embeds.shape[0]):
                lbl = labels[i]
                boo = True
                while boo:
                    j = random.randint(0,71)
                    if lbl != labels[j]:
                        neg_idxs.append(j)
                        boo = False
                boo = True
                while boo:
                    j = random.randint(0,71)
                    if lbl == labels[j] and i != j:
                        pos_idxs.append(j)
                        boo = False
        else:
            dist_mtx = pdist(embeds, embeds).detach().cpu().numpy()
            labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
            num = labels.shape[0]
            dia_inds = np.diag_indices(num)
            lb_eqs = labels == labels.T
            lb_eqs[dia_inds] = False
            dist_same = dist_mtx.copy()
            dist_same[lb_eqs == False] = -np.inf
            pos_idxs = np.argmax(dist_same, axis = 1)
            dist_diff = dist_mtx.copy()
            lb_eqs[dia_inds] = True
            dist_diff[lb_eqs == True] = np.inf
            neg_idxs = np.argmin(dist_diff, axis = 1)

        pos = embeds[pos_idxs].contiguous().view(num, -1)
        neg = embeds[neg_idxs].contiguous().view(num, -1)
        return embeds, pos, neg



if __name__ == '__main__':
    embds = torch.randn(10, 128)
    labels = torch.tensor([0,1,2,2,0,1,2,1,1,0])
    selector = BatchHardTripletSelector()
    anchor, pos, neg = selector(embds, labels)
    print(anchor.shape)