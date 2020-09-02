# -*- coding: utf-8 -*-


class Config(object):
    DATASET_DIR = '/content/drive/My Drive/DATN/dataset/fchd_datas/brainwash'
    TRAIN_ANNOTS_FILE = 'brainwash_train.idl'
    VAL_ANNOTS_FILE = 'brainwash_val.idl'
    # New dataset
    BKDATASET_DIR= '/content/drive/My Drive/DATN/dataset/BK_datas'
    BKTRAIN_ANNOTS_FILE='train_d3bkhn.idl'
    BKVAL_ANNOTS_FILE = 'val_d3bkhn.idl'
    BK_PLUS_BRAINWASH_TRAIN_ANNOTS_FILE='train_d3bkhn_plus_brainwash.idl'
    #############
    CAFFE_PRETRAIN = True
    CAFFE_PRETRAIN_PATH = './checkpoints/pre_trained/vgg16_caffe.pth'
    # CAFFE_PRETRAIN_PATH = './checkpoints/checkpoint_06171105_1.000.pth'
    RPN_NMS_THRESH = 0.7
    RPN_TRAIN_PRE_NMS_TOP_N = 12000
    RPN_TRAIN_POST_NMS_TOP_N = 300
    RPN_TEST_PRE_NMS_TOP_N = 6000
    RPN_TEST_POST_NMS_TOP_N = 300
    RPN_MIN_SIZE = 16

    ANCHOR_BASE_SIZE = 16
    ANCHOR_RATIOS = [1]
    ANCHOR_SCALES = [2, 4]

    EPOCHS = 18
    PRINT_LOG = True
    PLOT_INTERVAL = 2
    VISDOM_ENV = 'fchd'

    MODEL_DIR = './checkpoints'
    BEST_MODEL_PATH = './checkpoints/checkpoint_best.pth'
    NEW_MODEL_PATH='./checkpoints/checkpoint_06280811_0.945.pth'

cfg = Config()
