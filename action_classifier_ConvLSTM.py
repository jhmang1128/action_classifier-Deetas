import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, LSTM, ConvLSTM2D, MaxPooling3D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical as one_hot

from module import class_process as class_cm
from module import dataset as dataset_cm
from module import model as model_cm

import os
import csv
import numpy as np

from tensorflow.python.keras.metrics import accuracy
from sklearn.metrics import precision_score as skl_precision
from sklearn.metrics import recall_score as skl_recall


def main():
    ##############################################################################################
    # initial config
    ##############################################################################################
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0","/GPU:1"])

    model_config_dict = {}
    path_dict = {}
    result_csv_file_name = 'results_accuracy.csv'
    
    ####### date and experiment
    date_str = 'data_22_01_14'
    num_experiment = 'ex_04_ConvLSTM_mini'
    # num_experiment = 'ex_02_ConvLSTM_with_image'

    # including_image = True
    including_image = False

    ####### input config
    Num_bbox = 5
    image_width = 256
    image_height = 256

    Num_class = 4
    Num_repeat = 20
    
    ####### train config
    optimizer_list = ['adam']
    loss_list = ['categorical_crossentropy']
    epoch_list = [250]
    batch_size_list = [30]
    learning_rate_list = [0.00001]

    ####### model dict (just for deliver)
    model_config_dict['gpus'] = mirrored_strategy
    model_config_dict['bool_image'] = including_image
    
    
    model_config_dict['Num_bbox'] = Num_bbox
    model_config_dict['width'] = image_width
    model_config_dict['height'] = image_height
    model_config_dict['channel'] = 3

    model_config_dict['Num_class'] = Num_class
    model_config_dict['Num_repeat'] = Num_repeat

    model_config_dict['optimizer'] = optimizer_list
    model_config_dict['loss'] = loss_list
    model_config_dict['epoch'] = epoch_list
    model_config_dict['batch_size'] = batch_size_list
    model_config_dict['learning_rate'] = learning_rate_list

    ##############################################################################################
    # define path
    ##############################################################################################
    print('**********************************************************************************')
    print('main :', '\n')
    ####### input path
    # home_path = os.path.expanduser('~')
    # deetas_root_path = os.path.join(home_path, 'maeng_space/dataset_2021/Deetas')
    deetas_root_path = os.path.expanduser('../../../dataset_2021/Deetas')
    convlstm_root_path = os.path.join(deetas_root_path, date_str, 'json_ConvLSTM')
    hdf5_path = os.path.join(convlstm_root_path, 'dataset.hdf5')
    hdf5_path_with_image = os.path.join(convlstm_root_path, 'dataset_with_image.hdf5')
    
    ####### output path
    output_root_path = os.path.join(deetas_root_path, 'output_action_classifier')
    experiment_root_path = os.path.join(output_root_path, num_experiment)
    save_model_path = os.path.join(experiment_root_path, 'model')
    result_csv_path = os.path.join(experiment_root_path, result_csv_file_name)
    
    ####### path dict
    path_dict['ConvLSTM_path'] = convlstm_root_path
    path_dict['hdf5'] = hdf5_path
    path_dict['hdf5_with_image'] = hdf5_path_with_image

    path_dict['ex_path'] = experiment_root_path
    path_dict['save_model_path'] = save_model_path
    path_dict['csv_path'] = result_csv_path
    
    print('convlstm_root_path :', convlstm_root_path)
    print('save_model_path :', save_model_path)
    print('result_csv_path :', result_csv_path)
    
    ####### load data
    if not including_image:
        dataset_dict = dataset_cm.load_HDF5(path_dict)
    else:
        dataset_dict = dataset_cm.load_HDF5_with_image(path_dict)
    
    ####### train and evalutation
    repeat_experiment(dataset_dict, path_dict, model_config_dict)


###################################################################################################################
# train model
###################################################################################################################
def action_classifier(dataset_dict, model_config_dict, path_dict, csv_writer):
    ####### model conifg
    mirrored_strategy = model_config_dict['gpus']
    optimizer = model_config_dict['optimizer']
    loss = model_config_dict['loss']
    epoch = model_config_dict['epoch']
    batch_size = model_config_dict['batch_size']
    learning_rate = model_config_dict['learning_rate']
    idx_result = model_config_dict['idx_result']

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
    _loss, _accuracy = model.evaluate(x = [test_bbox, test_object_class], y = test_action_class)

    metric_list = [_loss, _accuracy]

    save_model_path = path_dict['save_model_path']
    model.save(save_model_path + '/' + str(idx_result) + '.h5')
    
    ####### result
    dataset_cm.write_results_accuracy(metric_list, model_config_dict, csv_writer)


def action_classifier_with_image(dataset_dict, model_config_dict, path_dict, csv_writer):
    ####### model conifg
    optimizer = model_config_dict['optimizer']
    loss = model_config_dict['loss']
    epoch = model_config_dict['epoch']
    batch_size = model_config_dict['batch_size']
    learning_rate = model_config_dict['learning_rate']
    idx_result = model_config_dict['idx_result']

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

    metric_list = [_loss, _accuracy]

    save_model_path = path_dict['save_model_path']
    model.save(save_model_path + '/' + str(idx_result) + '.h5')
    
    ####### result
    dataset_cm.write_results_accuracy(metric_list, model_config_dict, csv_writer)

###################################################################################################################
# repeat
###################################################################################################################
def repeat_experiment(dataset_dict, path_dict, model_config_list_dict):
    ####### initial
    bool_image = model_config_list_dict['bool_image']

    idx_result = 1
    model_config_dict = {}
    
    ####### config
    optimizer_list = model_config_list_dict['optimizer']
    loss_list = model_config_list_dict['loss']
    epoch_list = model_config_list_dict['epoch']
    batch_size_list = model_config_list_dict['batch_size']
    learning_rate_list = model_config_list_dict['learning_rate']

    model_config_dict['gpus'] = model_config_list_dict['gpus']

    model_config_dict['Num_bbox'] = model_config_list_dict['Num_bbox']
    model_config_dict['width'] = model_config_list_dict['width']
    model_config_dict['height'] = model_config_list_dict['height']
    model_config_dict['channel'] = model_config_list_dict['channel']
    
    model_config_dict['Num_bbox'] = model_config_list_dict['Num_bbox']
    model_config_dict['Num_class'] = model_config_list_dict['Num_class']

    Num_repeat = model_config_list_dict['Num_repeat']

    Num_experimnet = (len(optimizer_list) * len(loss_list) *
                        len(epoch_list) * len(batch_size_list) * len(learning_rate_list) *
                        Num_repeat)

    ####### path list
    result_csv_path = path_dict['csv_path']
    experiment_root_path = path_dict['ex_path']
    save_model_root_path = path_dict['save_model_path']

    ####### record result
    # fieldnames = ['optimizer', 'loss', 'epoch', 'batch_size', 'learning_rate',
    #                         'idx_repeat', 'Loss', 'Accuracy', 'Precision', 'Recall']
    fieldnames = ['optimizer', 'loss', 'epoch', 'batch_size', 'learning_rate',
                            'idx_repeat', 'Loss', 'Accuracy']

    
    if not os.path.exists(experiment_root_path):
        print("**************************************************************************")
        print('experiment_root_path :', experiment_root_path)
        os.makedirs(experiment_root_path)
        if not os.path.exists(save_model_root_path):
            print('save_model_root_path :', save_model_root_path)
            os.makedirs(save_model_root_path)
        print('\n')

    if os.path.isfile(result_csv_path):
        print("alredy exist result file : please change your num_experiment and reindex result or remove it")
        # dataset_cm.check_to_exist_file(5, result_csv_path)
        csvfile = open(result_csv_path, 'a', newline='')
    
    else:
        csvfile = open(result_csv_path, 'w', newline='')
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
    
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    ####### repeat
    check_path_model = path_dict['save_model_path']

    for optimizer in optimizer_list:
        for loss in loss_list:
            for epoch in epoch_list:
                for batch_size in batch_size_list:
                    for learning_rate in learning_rate_list:
                        for idx_repeat in range(Num_repeat):
                            model_config_dict['optimizer'] = optimizer
                            model_config_dict['loss'] = loss
                            model_config_dict['epoch'] = epoch
                            model_config_dict['batch_size'] = batch_size
                            model_config_dict['learning_rate'] = learning_rate
                            model_config_dict['idx_repeat'] = idx_repeat
                            print("**************************************************************************")
                            print('experiment : num ::', idx_result, "/", Num_experimnet, "\n")
                            if os.path.isfile(check_path_model + '/' + str(idx_result) + '.h5'):
                                idx_result += 1
                                continue

                            model_config_dict['idx_result'] = idx_result
                            if bool_image:
                                print('traning with image')
                                action_classifier_with_image(dataset_dict, model_config_dict, path_dict, csv_writer)
                            else:
                                print('traning')
                                action_classifier(dataset_dict, model_config_dict, path_dict, csv_writer)
                            idx_result += 1

###################################################################################################################
# end
###################################################################################################################
if __name__=='__main__':
    main()