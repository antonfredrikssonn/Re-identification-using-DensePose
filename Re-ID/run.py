import logging
import os
from pathlib import Path
from embed_evaluate import embed, evaluate
from trainFull import trainFull
from trainMain import trainMain
import matplotlib.pyplot as plt


def run(name, load_path, load_path_query, bools, ran_er = True, net_init = None, fused = True, local = True, num_it = 30000):
    path_log = Path("log2/" + name + ".log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    try:
        if path_log.exists():
            raise FileExistsError
    except FileExistsError:
        print("\nAlready exist log file: {}".format(path_log))
        print("Do you really want to delete and overwrite \ {} \" ? (y/n) ".format(path_log))
        ch = input()
        if ch=='y':
            try:
                os.remove(path_log)
                print("\nThe File, \ {} \" deleted successfully!".format(path_log))
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                    datefmt="%a, %d %b %Y %H:%M:%S",
                    filename=path_log.__str__(),
                    filemode="w",
                )
            except IOError:
                print("\nThe file \ {} \" is not available in the directory!".format(path_log))
        else:
            print("\nExiting...")
            raise
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=path_log.__str__(),
            filemode="w",
        )
        print("Create log file: {}".format(path_log))

    if fused:
        epochs, losses_val, losses_train, losses = trainFull(name, load_path, bools, net_init, num_it, ran_er)
    else:
        epochs, losses_val, losses_train, losses = trainMain(name, load_path, bools, net_init, local, num_it, ran_er)

    plt.plot(epochs, losses_val, 'g', label="Validation loss")
    plt.plot(epochs, losses_train, 'b', label="Training Loss")
    plt.title('Loss plots for model {}'.format(name))
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('plots2/{}_loss'.format(name))
    plt.clf()

    if bools[0][0] or bools[0][1]:
        plt.plot(epochs, losses[0], 'g', label="ArcFace loss")
    if bools[1][0] or bools[1][1]:
        plt.plot(epochs, losses[1], 'b', label="Triplet loss")
    if bools[2][0] or bools[2][1]:
        plt.plot(epochs, losses[2], 'r', label="ID loss")
    if bools[3][0] or bools[3][1]:
        plt.plot(epochs, losses[3], 'm', label="Center loss")
    if bools[4][0] or bools[4][1]:
        plt.plot(epochs, losses[4], 'k', label="Centroid loss")

    plt.title('Different losses for model {}'.format(name))
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('plots2/{}_losses'.format(name))
    plt.clf()

    for mod in  ['_full', '_lowestloss']:
        embed(name+mod, load_path + 'bounding_box_test_no0', type = 'gallery')
        embed(name+mod, load_path_query, type = 'query')
        evaluate(name+mod, local, bools[0][0] or bools[0][1])
if __name__ == '__main__':
    load_path = '/mnt/analyticsvideo/DensePoseData/market1501/'
    load_path_query = '/mnt/analyticsvideo/DensePoseData/market1501/query'


    ran_er = False
    net_init = None
    fused = True
    local = True
    num_ep = 170
    arc = [False, False, 0.4, 0.4] # [main, fused, weight_main, weight_fused]
    trip = [False, True, 1.5, 1.5]
    id = [True, True, 0.2, 0.5]
    center = [True, True, 2.5e-3, 5e-3]
    centroid = [False, False, 4, 8]
    bools = [arc, trip, id, center, centroid]
    names = ['test']

    for name in names:
        run(name, load_path, load_path_query, bools, ran_er=ran_er, net_init=net_init, fused=fused, local=local,num_it=num_ep)
