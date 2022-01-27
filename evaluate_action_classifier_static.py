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
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score as AP

from module import class_process as class_cm
from module import dataset as dataset_cm
from module import model as model_cm

# from sklearn.metrics import confusion_matrix

def main():
    ##############################################################################################
    # initial parameter
    ##############################################################################################
    path_dict = {}
    model_config_dict = {}
    result_csv_file_name = 'results_accuracy.csv'

    date_str = 'data_22_01_14'
    num_experiment = 'ex_04_ConvLSTM_mini'
    including_image = False

    num_model_list = ['1', '2', '3', '4', '5', '6', '7']

    Num_bbox = 5

    ####### model dict
    model_config_dict['Num_bbox'] = Num_bbox

    ##############################################################################################
    # define path and file config
    ##############################################################################################
    print('**********************************************************************************')
    print('main :', '\n')
    ####### input path
    deetas_root_path = os.path.expanduser('../../../dataset_2021/Deetas')
    ConvLSTM_root_path = os.path.join(deetas_root_path, date_str, 'json_ConvLSTM')
    hdf5_path = os.path.join(ConvLSTM_root_path, 'dataset.hdf5')
    hdf5_path_with_image = os.path.join(ConvLSTM_root_path, 'dataset_with_image.hdf5')

    ####### output path
    output_root_path = os.path.join(deetas_root_path, 'output_action_classifier')
    model_root_path = os.path.join(output_root_path, num_experiment, 'model')
    result_csv_path = os.path.join(output_root_path, num_experiment, result_csv_file_name)

    ####### path dict
    path_dict['ConvLSTM_path'] = ConvLSTM_root_path
    path_dict['hdf5'] = hdf5_path
    path_dict['hdf5_with_image'] = hdf5_path_with_image

    path_dict['save_model_path'] = model_root_path
    path_dict['csv_path'] = result_csv_path

    ##############################################################################################
    # execute
    ##############################################################################################
    ####### check result file (top 10)
    check_result_file(result_csv_path)

    ####### data load
    if not including_image:
        dataset_dict = dataset_cm.load_HDF5(path_dict)
    else:
        dataset_dict = dataset_cm.load_HDF5_with_image(path_dict)
    
    ####### eresult_csv_pathaluation all model (top10)
    # print('**********************************************************************************')
    # print("raw path :", "/home/dblab/maeng_space/git_repository/action_classifier/LSTM_action_classifier/output_maeng/ex_000/model/")
    # print("compare : model_path :", model_root_path)
    # model_name_list = os.listdir(model_root_path)
    # check_metric_list = []
    # for model_file_name in model_name_list:
    #     load_model_path = os.path.join(model_root_path, model_file_name)
    #     check_metric_list = check_mAP(dataset_dict, check_metric_list, load_model_path)
    
    # check_metric_list.sort(key = itemgetter(2), reverse=True)
    # top_10_result = check_metric_list[0:10]
    # for idx_top in top_10_result:
    #     print(idx_top)
    
    ###### evaluation single model
    check_metric_list = []
    for num_model in num_model_list:
        model_file_name = num_model + ".h5"
        load_model_path = os.path.join(model_root_path, model_file_name)
        check_metric_list = check_mAP(dataset_dict, load_model_path, model_config_dict, check_metric_list)

    

###################################################################################################################
# evaluation and predict for check performence of model
###################################################################################################################
def check_mAP(dataset_dict, load_model_path, model_config_dict, check_metric_total_list):
    print("***********************************************************************")
    print("check_mAP")
    ### initial
    check_metric_list = []
    Num_bbox = model_config_dict['Num_bbox']

    ### read data
    test_bbox = dataset_dict['test']['bbox']
    test_bbox = test_bbox.reshape(test_bbox.shape[0], Num_bbox, 2, 2, 1)
    test_object_class = dataset_dict['test']['object_class']
    test_action_class = one_hot(dataset_dict['test']['action_class'])

    model = load_model(load_model_path)
    print("***********************************************************************")
    print('load_model_path :', load_model_path)

    # model.summary()

    ####### complie 01 = keras metrics
    model.compile(metrics=["acc",
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall')])

    ####### complie 02 = keras metrics
    # model.compile(metrics=['accuracy', precision, recall, f1score])

    ####### output 01 = evaluation
    # _loss, _acc, _precision, _recall = model.evaluate(x = [test_bbox, test_object_class], y = test_action_class)

    # print("loss :", _loss,
    #         "\n" + "acc :", _acc,
    #         "\n" + "precision :", _precision,
    #         "\n" + "recall :", _recall)

    # check_metric_list.append(load_model_path)
    # check_metric_list.append(_loss)
    # check_metric_list.append(_acc)
    # check_metric_list.append(_precision)
    # check_metric_list.append(_recall)

    # check_metric_total_list.append(check_metric_list)
    

    ####### output 02 = predict
    y_test = test_action_class
    y_test_max = np.argmax(y_test, axis=1)

    x_test = [test_bbox, test_object_class]
    y_pred = model.predict(x_test, batch_size=64, verbose=1)
    y_pred_max = np.argmax(y_pred, axis=1)


    ######## mAP
    check_result = classification_report(y_test_max, y_pred_max)
    print('sklearn.metrics.classification_report : \n', check_result)

    print("***********************************************************************")
    # print('y_pred : before :', y_pred.shape)

    check_result = AP(y_test, y_pred)
    
    print('sklearn.metrics.average_precision_score : mAP : \n', check_result)
    
    check_result = AP(y_test, y_pred, average = None)
    
    print('sklearn.metrics.average_precision_score : each score : \n', check_result)
    
    return check_metric_total_list


###################################################################################################################
# result file (sort, top n)
###################################################################################################################
def check_result_file(load_csv_path):
    print("***********************************************************************")
    print("check_result_file : file_path :", '\n', load_csv_path)
    
    csv_file = open(load_csv_path, 'r')
    csv_reader = csv.reader(csv_file)

    test_accuracy_list = []
    for idx_line, result in enumerate(csv_reader):
        if idx_line == 0:
            continue
        result.append(idx_line)
        test_accuracy_list.append(result)
        if result[0] =='model_name':
            print(result[0])
            test_accuracy_list = []

    test_accuracy_list.sort(key = itemgetter(7), reverse=True)
    top_10_result = test_accuracy_list[0:10]
    for idx_top in top_10_result:
        print(idx_top)


###################################################################################################################
# handling action class
###################################################################################################################
def check_act_class(action_class):
    if action_class == 'Going':
        return False
    elif action_class == 'Coming':
        return False
    elif action_class == 'Crossing':
        return False
    elif action_class == 'Stopping':
        return False
    elif action_class == 'Moving':
        return False
    elif action_class == 'Avoiding':
        return False
    elif action_class == 'Opening':
        return True
    elif action_class == 'Closing':
        return True
    elif action_class == 'On':
        return True
    elif action_class == 'Off':
        return True


def convert_action_class (action_class):
    ### output = 0
    if action_class == 'Going':
        output = 0
    elif action_class == 'Coming':
        output = 1
    elif action_class == 'Crossing':
        output = 2
    elif action_class == 'Stopping':
        output = 3
    elif action_class == 'Moving':
        output = 4
    elif action_class == 'Avoiding':
        output = 5
    elif action_class == 'Opening': # not use
        output = 6
    elif action_class == 'Closing': # not use
        output = 7
    elif action_class == 'On': # not use
        output = 8
    elif action_class == 'Off': # not use
        output = 9

    return output


###################################################################################################################
# load and write
###################################################################################################################
def load_splited_data(Num_bbox, load_file_path):
    print('**********************************************************************************')
    print('compare : act path :', '/home/dblab/maeng_space/dataset/deetas/data_*/json_act', '\n',
            load_file_path, '\n',)

    ### define list to return
    bbox_list = []
    object_class_list = []
    action_class_list = []
    output_data_dict = {}

    ### load pickle data
    pickle_file = open(load_file_path, "rb")
    tracks_list = pickle.load(pickle_file)

    for idx_anno, annotations_list in enumerate(tracks_list):
        num_annotation = len(annotations_list) - 1
        # if num_annotation < Num_bbox:
        #     continue

        ### generate batch
        batch_list = []
        if num_annotation < Num_bbox:
            batch_data = []
            object_class = annotations_list[0]
            batch_data.append(object_class)
            batch_data.extend(annotations_list[1:num_annotation+1])
            for idx_padding in range(Num_bbox-num_annotation):
                batch_data.append(annotations_list[-1])
            batch_list.append(batch_data)

        else:
            for idx_anno in range(num_annotation - Num_bbox + 1):
                if idx_anno == 0:
                    object_class = annotations_list[0]
                    continue
                batch_data = []
                batch_data.append(object_class)
                batch_data.extend(annotations_list[idx_anno:idx_anno+5])
                batch_list.append(batch_data)

        ### batch to splited dataset
        for idx_batch, batch_data in enumerate(batch_list):
            temp_act = []
            temp_bbox = []
            for annotation in batch_data[1:]:
                action_class = annotation[0]
                image_id = annotation[1]
                bboxes = annotation[2]

                temp_act.append(action_class)
                temp_bbox.append(bboxes)

            temp_act = list(set(temp_act))
            if len(temp_act) > 1:
                print(temp_act)
                print("error : action_class")
                exit()

            object_class = batch_data[0]
            action_class = temp_act[0]
            if check_act_class(action_class):
                continue
            bboxes = temp_bbox

            object_class_list.append(object_class)
            action_class_list.append(convert_action_class(action_class))
            bbox_list.append(bboxes)

    object_class_np = np.array(object_class_list)
    bbox_np = np.array(bbox_list)
    action_class_np = np.array(action_class_list)

    output_data_dict['object_class'] = object_class_np
    output_data_dict['bbox'] = bbox_np
    output_data_dict['action_class'] = action_class_np

    # check_list = set(action_class_list)
    # print(check_list)

    return output_data_dict


def load_whole_data(Num_bbox, json_path_list):
    ####### initial
    dataset_dict = {}
    
    ####### load train, validation, test
    train_dict = load_splited_data(Num_bbox, json_path_list[0])
    val_dict = load_splited_data(Num_bbox, json_path_list[1])
    test_dict = load_splited_data(Num_bbox, json_path_list[2])

    train_bbox = train_dict['bbox']
    train_object_class = train_dict['object_class']
    train_action_class = train_dict['action_class']

    val_bbox = val_dict['bbox']
    val_object_class = val_dict['object_class']
    val_action_class = val_dict['action_class']
    
    train_bbox = np.concatenate((train_bbox, val_bbox), axis=0)
    train_object_class = np.concatenate((train_object_class, val_object_class), axis=0)
    train_action_class = np.concatenate((train_action_class, val_action_class), axis=0)
    train_action_class = one_hot(train_action_class)

    test_bbox = test_dict['bbox']
    test_object_class = test_dict['object_class']
    test_action_class = test_dict['action_class']
    test_action_class = one_hot(test_action_class)

    print('train_bbox :', train_bbox.shape)
    print('train_object_class :', train_object_class.shape)
    print('train_action_class :', train_action_class.shape)
    
    print('test_bbox :', test_bbox.shape)
    print('test_object_class :', test_object_class.shape)
    print('test_action_class :', test_action_class.shape)

    dataset_dict['train_bbox'] = train_bbox
    dataset_dict['train_object_class'] = train_object_class
    dataset_dict['train_action_class'] = train_action_class
    dataset_dict['test_bbox'] = test_bbox
    dataset_dict['test_object_class'] = test_object_class
    dataset_dict['test_action_class'] = test_action_class

    return dataset_dict


def write_result(input_dict):
    input_dict = 0

###################################################################################################################
# end
###################################################################################################################
if __name__=='__main__':
    main()