import os
import pickle
import csv
import numpy as np
from operator import itemgetter

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.keras.utils import to_categorical as one_hot
from tensorflow.keras import backend as K

from sklearn.metrics import f1_score as skl_f1_score
from sklearn.metrics import precision_score as skl_precision
from sklearn.metrics import recall_score as skl_recall
from sklearn.metrics import confusion_matrix as skl_confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix as mcm
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score as AP

from module import class_process as class_cm
from module import dataset as dataset_cm
from module import model as model_cm

import argparse
import atexit

# from sklearn.metrics import confusion_matrix

def main():
    ##############################################################################################
    # initial parameter
    ##############################################################################################
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0","/GPU:1"])

    parser = argparse.ArgumentParser(description='Train Mask R-CNN.')
    parser.add_argument('--load_hdf5_path', required=True, help="load path of hdf5 file")
    parser.add_argument('--load_model_path', required=True, help="save path of model")
    parser.add_argument('--with_image', required=True, metavar="True or False", help='whether the image exists or not')
    parser.add_argument('--Num_bbox', required=True, type=int, metavar="5", help='width of image')
    parser.add_argument('--Num_class', required=True, type=int, metavar="4", help='height of image')

    args = parser.parse_args()
    print("load_hdf5_path : ", args.load_hdf5_path)
    print("load_model_path : ", args.load_model_path)

    print("with_image : ", args.with_image)

    print("Num_bbox : ", args.Num_bbox)
    print("Num_class : ", args.Num_class)

    LOAD_HDF5_PATH = args.load_hdf5_path
    LOAD_MODEL_PATH = args.load_model_path
    WITH_IMAGE = args.with_image
    NUM_BBOX = args.Num_bbox
    NUM_CLASS = args.Num_class

    path_dict = {}
    model_config_dict = {}

    ####### model dict
    model_config_dict['Num_bbox'] = NUM_BBOX
    model_config_dict['Num_class'] = NUM_CLASS
    model_config_dict['gpus'] = mirrored_strategy

    ####### path dict
    path_dict['hdf5'] = LOAD_HDF5_PATH
    path_dict['load_model_path'] = LOAD_MODEL_PATH

    ##############################################################################################
    # execute
    ##############################################################################################
    ####### data load
    if WITH_IMAGE == 'False':
        dataset_dict = dataset_cm.load_HDF5(path_dict)
    elif WITH_IMAGE == 'True':
        dataset_dict = dataset_cm.load_HDF5_with_image(path_dict)
    else:
        print('error : with image : typing True or False')
        exit()
    
    ###### evaluation single model
    check_mAP(dataset_dict, LOAD_MODEL_PATH, model_config_dict)
    atexit.register(mirrored_strategy._extended._collective_ops._pool.close)

    

###################################################################################################################
# evaluation and predict for check performence of model
###################################################################################################################
def check_mAP(dataset_dict, load_model_path, model_config_dict):
    print("***********************************************************************")
    print("check_mAP")
    ### initial
    check_metric_list = []
    Num_bbox = model_config_dict['Num_bbox']
    mirrored_strategy = model_config_dict['gpus']

    ### read data
    test_bbox = dataset_dict['test']['bbox']
    test_bbox = test_bbox.reshape(test_bbox.shape[0], Num_bbox, 2, 2, 1)
    test_object_class = dataset_dict['test']['object_class']
    test_action_class = one_hot(dataset_dict['test']['action_class'])

    with mirrored_strategy.scope():
        model = load_model(load_model_path)
        print("***********************************************************************")
        print('load_model_path :', load_model_path)

        # model.summary()

        ####### complie = keras metrics
        model.compile(metrics=["acc",
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall')])

        ####### output = predict
        y_test = test_action_class
        y_test_max = np.argmax(y_test, axis=1)

        x_test = [test_bbox, test_object_class]
        y_pred = model.predict(x_test, batch_size=64, verbose=1)
        y_pred_max = np.argmax(y_pred, axis=1)

    ######## mAP
    print("***********************************************************************")
    print('sklearn.metrics.classification_report : \n', classification_report(y_test_max, y_pred_max), '\n')
    print('sklearn.metrics.average_precision_score : mAP : \n', AP(y_test, y_pred), '\n')
    print('sklearn.metrics.average_precision_score : each score : \n', AP(y_test, y_pred, average = None), '\n')


###################################################################################################################
# end
###################################################################################################################
if __name__=='__main__':
    main()