import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, LSTM, ConvLSTM2D, MaxPooling3D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical as one_hot

from module import class_process as class_cm
from module import dataset as dataset_cm
from module import model as model_cm

import argparse
import numpy as np

from tensorflow.python.keras.metrics import accuracy
from sklearn.metrics import precision_score as skl_precision
from sklearn.metrics import recall_score as skl_recall

from tensorflow.python.client import device_lib
import atexit

def main():
    ##############################################################################################
    # initial config
    ##############################################################################################
    gpus = device_lib.list_local_devices()
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0","/GPU:1"])

    parser = argparse.ArgumentParser(description='Train Mask R-CNN.')

    parser.add_argument('--load_hdf5_path', required=True, help="load path of hdf5 file")
    parser.add_argument('--save_model_path', required=True, help="save path of model")

    parser.add_argument('--with_image', required=True, metavar="True or False", help='whether the image exists or not')

    parser.add_argument('--Num_bbox', required=True, type=int, metavar="5", help='width of image')
    parser.add_argument('--Num_class', required=True, type=int, metavar="4", help='height of image')
    parser.add_argument('--crop_width', required=True, type=int, metavar="256", help='width of image')
    parser.add_argument('--crop_height', required=True, type=int, metavar="256", help='height of image')

    parser.add_argument('--optimizer', required=True, metavar="adam", help='select optimizer in tensorflow')
    parser.add_argument('--loss', required=True, metavar="categorical_crossentropy", help='select loss function in tensorflow')

    parser.add_argument('--epoch', required=True, type=int, metavar="250", help='num epoch')
    parser.add_argument('--batch_size', required=True, type=int, metavar="20", help='num batch size')
    parser.add_argument('--learning_rate', required=True, type=float, metavar="0.0001", help='num learning_rate')

    args = parser.parse_args()
    print("load_hdf5_path : ", args.load_hdf5_path)
    print("save_model_path : ", args.save_model_path)

    print("with_image : ", args.with_image)

    print("Num_bbox : ", args.Num_bbox)
    print("Num_class : ", args.Num_class)
    print("crop_width : ", args.crop_width)
    print("crop_height : ", args.crop_height)

    print("optimizer : ", args.optimizer)
    print("loss : ", args.loss)
    print("epoch : ", args.epoch)
    print("batch_size : ", args.batch_size)
    print("learning_rate : ", args.learning_rate)

    LOAD_HDF5_PATH = args.load_hdf5_path
    SAVE_MODEL_PATH = args.save_model_path
    WITH_IMAGE = args.with_image
    NUM_BBOX = args.Num_bbox
    NUM_CLASS = args.Num_class
    CROP_WIDTH = args.crop_width
    CROP_HEIGHT = args.crop_height
    OPTIMIZER = args.optimizer
    LOSS = args.loss
    EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate

    model_config_dict = {}
    path_dict = {}
    
    ####### model dict (just for deliver)
    model_config_dict['gpus'] = mirrored_strategy
    model_config_dict['bool_image'] = WITH_IMAGE
    
    model_config_dict['Num_bbox'] = NUM_BBOX
    model_config_dict['width'] = CROP_WIDTH
    model_config_dict['height'] = CROP_HEIGHT
    model_config_dict['channel'] = 3

    model_config_dict['Num_class'] = NUM_CLASS
    model_config_dict['optimizer'] = OPTIMIZER
    model_config_dict['loss'] = LOSS
    model_config_dict['epoch'] = EPOCH
    model_config_dict['batch_size'] = BATCH_SIZE
    model_config_dict['learning_rate'] = LEARNING_RATE

    ##############################################################################################
    # define path
    ##############################################################################################
    print('**********************************************************************************')
    print('main :', '\n')
    ####### path dict
    path_dict['hdf5'] = LOAD_HDF5_PATH
    path_dict['output'] = SAVE_MODEL_PATH
    
    ####### load data
    if WITH_IMAGE == 'False':
        dataset_dict = dataset_cm.load_HDF5(path_dict)
    elif WITH_IMAGE == 'True':
        dataset_dict = dataset_cm.load_HDF5_with_image(path_dict)
    else:
        print('error : with image : typing True or False')
        exit()
    
    ####### train and evalutation
    repeat_experiment(dataset_dict, path_dict, model_config_dict)

    atexit.register(mirrored_strategy._extended._collective_ops._pool.close)


###################################################################################################################
# train model
###################################################################################################################
def action_classifier(dataset_dict, model_config_dict, path_dict):
    ####### model conifg
    mirrored_strategy = model_config_dict['gpus']
    optimizer = model_config_dict['optimizer']
    loss = model_config_dict['loss']
    epoch = model_config_dict['epoch']
    batch_size = model_config_dict['batch_size']
    learning_rate = model_config_dict['learning_rate']

    Num_bbox = model_config_dict['Num_bbox']

    ####### load data
    print ('dataset_dict : keys :', dataset_dict.keys())

    train_bbox = dataset_dict['train']['bbox']
    val_bbox = dataset_dict['val']['bbox']
    test_bbox = dataset_dict['test']['bbox']

    train_bbox = train_bbox.reshape(train_bbox.shape[0], Num_bbox, 2, 2, 1)
    val_bbox = val_bbox.reshape(val_bbox.shape[0], Num_bbox, 2, 2, 1)
    test_bbox = test_bbox.reshape(test_bbox.shape[0], Num_bbox, 2, 2, 1)

    train_object_class = dataset_dict['train']['object_class']
    val_object_class = dataset_dict['val']['object_class']
    test_object_class = dataset_dict['test']['object_class']

    train_action_class = one_hot(dataset_dict['train']['action_class'])
    val_action_class = one_hot(dataset_dict['val']['action_class'])
    test_action_class = one_hot(dataset_dict['test']['action_class'])

    ####### model
    with mirrored_strategy.scope():
        model = model_cm.ConvLSTM_model(model_config_dict)

        model.summary()

        model.compile(optimizer=optimizer, loss = loss, metrics=['accuracy'])
        model.optimizer.lr = learning_rate

    model.fit(x = [train_bbox, train_object_class],
                y = train_action_class,
                epochs = epoch,
                batch_size=batch_size,
                validation_data=([val_bbox, val_object_class], val_action_class))

    ###### test
    # _loss, _accuracy = model.evaluate(x = [test_bbox, test_object_class], y = test_action_class)

    save_model_path = path_dict['output']
    model.save(save_model_path)


def action_classifier_with_image(dataset_dict, model_config_dict, path_dict):
    ####### model conifg
    optimizer = model_config_dict['optimizer']
    loss = model_config_dict['loss']
    epoch = model_config_dict['epoch']
    batch_size = model_config_dict['batch_size']
    learning_rate = model_config_dict['learning_rate']

    Num_bbox = model_config_dict['Num_bbox']

    ####### load data
    train_bbox = dataset_dict['train']['bbox']
    val_bbox = dataset_dict['val']['bbox']
    test_bbox = dataset_dict['test']['bbox']

    train_bbox_image = dataset_dict['train']['bbox_image']
    val_bbox_image = dataset_dict['val']['bbox_image']
    test_bbox_image = dataset_dict['test']['bbox_image']

    train_object_class = dataset_dict['train']['object_class']
    val_object_class = dataset_dict['val']['object_class']
    test_object_class = dataset_dict['test']['object_class']

    train_action_class = one_hot(dataset_dict['train']['action_class'])
    val_action_class = one_hot(dataset_dict['val']['action_class'])
    test_action_class = one_hot(dataset_dict['test']['action_class'])


    ####### model
    model = model_cm.ConvLSTM_model_with_image(model_config_dict)

    model.summary()

    model.compile(optimizer=optimizer, loss = loss, metrics=['accuracy'])
    model.optimizer.lr = learning_rate

    ###### model 01
    model.fit(x = [train_bbox, train_bbox_image, train_object_class],
                y = train_action_class,
                epochs = epoch,
                batch_size=batch_size,
                validation_data=([val_bbox, val_bbox_image, val_object_class], val_action_class))

    
    _loss, _accuracy = model.evaluate(x = [test_bbox, test_bbox_image, test_object_class], y = test_action_class)

    save_model_path = path_dict['output']
    model.save(save_model_path)
    

###################################################################################################################
# repeat
###################################################################################################################
def repeat_experiment(dataset_dict, path_dict, model_config_list_dict):
    ####### initial
    bool_image = model_config_list_dict['bool_image']

    idx_result = 1
    model_config_dict = {}
    
    ####### config
    model_config_dict['gpus'] = model_config_list_dict['gpus']

    model_config_dict['Num_bbox'] = model_config_list_dict['Num_bbox']
    model_config_dict['width'] = model_config_list_dict['width']
    model_config_dict['height'] = model_config_list_dict['height']
    model_config_dict['channel'] = model_config_list_dict['channel']
    
    model_config_dict['Num_bbox'] = model_config_list_dict['Num_bbox']
    model_config_dict['Num_class'] = model_config_list_dict['Num_class']

    model_config_dict['optimizer'] = model_config_list_dict['optimizer']
    model_config_dict['loss'] = model_config_list_dict['loss']
    model_config_dict['epoch'] = model_config_list_dict['epoch']
    model_config_dict['batch_size'] = model_config_list_dict['batch_size']
    model_config_dict['learning_rate'] = model_config_list_dict['learning_rate']

    if bool_image == 'True':
        print('traning with image')
        action_classifier_with_image(dataset_dict, model_config_dict, path_dict)
    elif bool_image == 'False':
        print('traning')
        action_classifier(dataset_dict, model_config_dict, path_dict)

###################################################################################################################
# end
###################################################################################################################
if __name__=='__main__':
    main()