import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate
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
    # initial parameter
    ##############################################################################################
    model_config_dict = {}
    path_dict = {}
    
    ####### date and experiment
    date_str = 'data_21_12_02'
    result_csv_file_name = 'results_accuracy.csv'
    num_experiment = 'ex_LSTM_01'

    ####### input config
    Num_bbox = 5
    Num_class = 4
    Num_repeat = 3

    ####### train config
    optimizer_list = ['adam', 'sgd']
    loss_list = ['categorical_crossentropy', 'binary_crossentropy']
    epoch_list = [10, 50, 200]
    batch_size_list = [32, 128]
    learning_rate_list = [0.001, 0.0001, 0.00001]
    model_list = ['model_01', 'model_02', 'model_03']
    
    ####### model dict (just for delivser)
    model_config_dict['Num_bbox'] = Num_bbox
    model_config_dict['Num_class'] = Num_class
    model_config_dict['Num_repeat'] = Num_repeat

    model_config_dict['model_list'] = model_list
    model_config_dict['optimizer'] = optimizer_list
    model_config_dict['loss'] = loss_list
    model_config_dict['epoch'] = epoch_list
    model_config_dict['batch_size'] = batch_size_list
    model_config_dict['learning_rate'] = learning_rate_list

    ##############################################################################################
    # define path and file config
    ##############################################################################################
    print('**********************************************************************************')
    print('main :', '\n')
    ####### root path
    home_path = os.path.expanduser('~')
    work_space_path = os.path.join(home_path, 'maeng_space')

    input_root_path = os.path.join(work_space_path, 'output_submodule/deetas')
    object_class_root_path = os.path.join(input_root_path, date_str, 'json_obj')
    act_class_root_path = os.path.join(input_root_path, date_str, 'json_act')

    output_root_path = os.path.join(work_space_path, 'output_submodule/multi_classifier/action_classifier-Deetas')
    experiment_root_path = os.path.join(output_root_path, num_experiment)
    
    ####### load path
    load_json_train_path = os.path.join(act_class_root_path, 'train.pickle')
    load_json_val_path = os.path.join(act_class_root_path, 'val.pickle')
    load_json_test_path = os.path.join(act_class_root_path, 'test.pickle')
    input_path_list = [load_json_train_path, load_json_val_path, load_json_test_path]

    ####### output path
    print('compare : output_path : /home/dblab/maeng_space/git_repository/preprocess_data/preprocess_Deetas_dataset/output_maeng')
    save_model_path = os.path.join(experiment_root_path, 'model')
    path_dict['save_model_path'] = save_model_path
    print('save_model_path :', save_model_path)

    result_csv_path = os.path.join(experiment_root_path, result_csv_file_name)
    path_dict['csv_path'] = result_csv_path
    print('result_csv_path :', result_csv_path)
    
    ####### load data
    # generate_random_data(100, Num_bbox)
    
    ####### define main algorithm
    dataset_dict = dataset_cm.load_all_bbox_only(Num_bbox, input_path_list)
    # action_classifier(dataset_dict, model_conifg_dict, mode)
    repeat_train_evaluate(dataset_dict, path_dict, model_config_dict)

###################################################################################################################
# train model
###################################################################################################################
def repeat_train_evaluate(dataset_dict, path_dict, model_config_list_dict):
    ####### initial
    idx_result = 1
    model_conifg_dict = {}
    
    ####### config
    optimizer_list = model_config_list_dict['optimizer']
    loss_list = model_config_list_dict['loss']
    epoch_list = model_config_list_dict['epoch']
    batch_size_list = model_config_list_dict['batch_size']
    learning_rate_list = model_config_list_dict['learning_rate']
    model_list = model_config_list_dict['model_list']

    model_conifg_dict['Num_bbox'] = model_config_list_dict['Num_bbox']
    model_conifg_dict['Num_class'] = model_config_list_dict['Num_class']

    Num_repeat = model_config_list_dict['Num_repeat']

    Num_experimnet = len(model_list) * len(optimizer_list) * len(loss_list) * len(epoch_list) * len(batch_size_list) * len(learning_rate_list) * Num_repeat

    ####### path list
    result_csv_path = path_dict['csv_path']

    ####### record result
    fieldnames = ['model_name', 'optimizer', 'loss', 'epoch', 'batch_size', 'learning_rate',
                            'idx_repeat', 'test_loss', 'test_accuracy']
    
    if os.path.isfile(result_csv_path):
        print("alredy exist result file : please change your num_experiment and reindex result or remove it")
    
    else:
        csvfile = open(result_csv_path, 'w', newline='')
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        csvfile.close()

    csvfile = open(result_csv_path, 'a', newline='')
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    ####### repeat
    check_path_model = path_dict['save_model_path']

    for model_name in model_list:
        for optimizer in optimizer_list:
            for loss in loss_list:
                for epoch in epoch_list:
                    for batch_size in batch_size_list:
                        for learning_rate in learning_rate_list:
                            for idx_repeat in range(Num_repeat):
                                model_conifg_dict['model_name'] = model_name
                                model_conifg_dict['optimizer'] = optimizer
                                model_conifg_dict['loss'] = loss
                                model_conifg_dict['epoch'] = epoch
                                model_conifg_dict['batch_size'] = batch_size
                                model_conifg_dict['learning_rate'] = learning_rate
                                model_conifg_dict['idx_repeat'] = idx_repeat
                                model_conifg_dict['idx_result'] = idx_result

                                print("**************************************************************************")
                                print(idx_result, "/", Num_experimnet, "\n")
                                if os.path.isfile(check_path_model + '/' + str(idx_result) + '.h5'):
                                    idx_result += 1
                                    continue
                                action_classifier(dataset_dict, model_conifg_dict, path_dict, csv_writer)
                                
                                idx_result += 1

                                # if idx_result == 4 :
                                #     exit()

def action_classifier(dataset_dict, model_conifg_dict, path_dict, csv_writer):
    ### load data
    train_bbox = dataset_dict['train_bbox']
    train_object_class = dataset_dict['train_object_class']
    train_action_class = dataset_dict['train_action_class']
    test_bbox = dataset_dict['test_bbox']
    test_object_class = dataset_dict['test_object_class']
    test_action_class = dataset_dict['test_action_class']

    ####### model conifg
    Num_bbox = model_conifg_dict['Num_bbox']
    Num_class = model_conifg_dict['Num_class']

    model_name = model_conifg_dict['model_name']
    optimizer = model_conifg_dict['optimizer']
    loss = model_conifg_dict['loss']
    epoch = model_conifg_dict['epoch']
    batch_size = model_conifg_dict['batch_size']
    learning_rate = model_conifg_dict['learning_rate']
    idx_result = model_conifg_dict['idx_result']

    ####### model
    if model_name == 'model_01':
        model = model_01(Num_bbox, Num_class)
    elif model_name == 'model_02':
        model = model_02(Num_bbox, Num_class)
    elif model_name == 'model_03':
        model = model_03(Num_bbox, Num_class)
    else:
        print('not exist model')
        exit()

    # model.summary()

    model.compile(optimizer=optimizer, loss = loss, metrics=['accuracy'])
    model.optimizer.lr = learning_rate

    model.fit(x = [train_bbox, train_object_class],
                y = train_action_class,
                epochs = epoch,
                batch_size=batch_size,
                validation_split=1/9,)

    # test_loss, test_accuracy, test_precision, test_recall = model.evaluate(x = [test_bbox, test_object_class], y = test_action_class)
    test_loss, test_accuracy = model.evaluate(x = [test_bbox, test_object_class], y = test_action_class)
    metric_list = [test_loss, test_accuracy]

    save_model_path = path_dict['save_model_path']
    model.save(save_model_path + '/' + str(idx_result) + '.h5')
    
    ####### result
    print(test_loss)
    print(test_accuracy)
    dataset_cm.write_results_accuracy(metric_list, model_conifg_dict, csv_writer)


###################################################################################################################
# model
###################################################################################################################
def model_01(Num_bbox, Num_class):
    bounding_box = Input(shape = (Num_bbox, 4))
    object_class = Input(shape=(1))

    bbox_FM = LSTM(16, activation = 'relu')(bounding_box)

    concat_FM = concatenate([bbox_FM, object_class])

    output_FM = Dense(32, activation='relu')(concat_FM)
    output_FM = Dense(Num_class, activation='sigmoid')(output_FM)

    model = Model(inputs=[bounding_box, object_class], outputs=output_FM)

    return model

def model_02(Num_bbox, Num_class):
    bounding_box = Input(shape = (Num_bbox, 4))
    object_class = Input(shape=(1))

    class_FM = Dense(32, activation = 'relu')(object_class)

    bbox_FM = LSTM(32, activation = 'relu')(bounding_box)

    concat_FM = concatenate([bbox_FM, class_FM])

    output_FM = Dense(64, activation='relu')(concat_FM)
    output_FM = Dense(Num_class, activation='sigmoid')(output_FM)

    model = Model(inputs=[bounding_box, object_class], outputs=output_FM)

    return model


def model_03(Num_bbox, Num_class):
    bounding_box = Input(shape = (Num_bbox, 4))
    object_class = Input(shape=(1))

    class_FM = Dense(16, activation = 'relu')(object_class)
    class_FM = Dense(32, activation = 'relu')(class_FM)
    
    bbox_FM = LSTM(32, activation = 'relu', return_sequences=True)(bounding_box)
    bbox_FM = LSTM(64, activation = 'relu')(bbox_FM)

    concat_FM = concatenate([bbox_FM, class_FM])

    output_FM = Dense(128, activation='relu')(concat_FM)
    output_FM = Dense(Num_class, activation='sigmoid')(output_FM)

    model = Model(inputs=[bounding_box, object_class], outputs=output_FM)

    return model



###################################################################################################################
# end
###################################################################################################################
if __name__=='__main__':
    main()