import os
import numpy as np
from argparse import ArgumentParser


def make_datalist(data_fd, data_list):
    filename_all = os.listdir(data_fd)
    filename_all = [data_fd+'/'+filename+'\n' for filename in filename_all if filename.endswith('.tfrecords') or filename.endswith('.npy')]

    np.random.shuffle(filename_all)
    np.random.shuffle(filename_all)
    with open(data_list, 'a') as fp:
        fp.writelines(filename_all)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-fd", "--data_fd", dest="data_fd")
    parser.add_argument("-list", "--data_list", dest="data_list")

    args = parser.parse_args()

    make_datalist(args.data_fd, args.data_list)