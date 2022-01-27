from tensorflow.keras.utils import to_categorical as one_hot

from module import class_process as class_cm

import h5py

import os
import random
import time

import pickle
import cv2
import json
import numpy as np

###################################################################################################################
# load act class, bbox
###################################################################################################################
def load_HDF5(path_dict):
    hdf5_path = path_dict['hdf5']
    print('load_HDF5 :', hdf5_path)
    HDF5_path = os.path.join(hdf5_path)
    data_dict = {}
    data_dict['train'] = {}
    data_dict['val'] = {}
    data_dict['test'] = {}
    
    hf = h5py.File(HDF5_path, 'r')

    data_dict['train']['bbox'] = hf['train_bbox'][:]
    data_dict['train']['object_class'] = hf['train_object_class'][:]
    data_dict['train']['action_class'] = hf['train_action_class'][:]

    data_dict['val']['bbox'] = hf['val_bbox'][:]
    data_dict['val']['object_class'] = hf['val_object_class'][:]
    data_dict['val']['action_class'] = hf['val_action_class'][:]

    data_dict['test']['bbox'] = hf['test_bbox'][:]
    data_dict['test']['object_class'] = hf['test_object_class'][:]
    data_dict['test']['action_class'] = hf['test_action_class'][:]

    print('train_bbox :', data_dict['train']['bbox'].shape)
    print('train_object_class :', data_dict['train']['object_class'].shape)
    print('train_action_class :', data_dict['train']['action_class'].shape)

    print('val_bbox :', data_dict['val']['bbox'].shape)
    print('val_object_class :', data_dict['val']['object_class'].shape)
    print('val_action_class :', data_dict['val']['action_class'].shape)

    print('test_bbox :', data_dict['test']['bbox'].shape)
    print('test_object_class :', data_dict['test']['object_class'].shape)
    print('test_action_class :', data_dict['test']['action_class'].shape)

    return data_dict

def load_HDF5_with_image(path_dict):
    hdf5_path = path_dict['hdf5']
    print('hdf5_with_image :', hdf5_path)
    HDF5_path = os.path.join(hdf5_path)
    data_dict = {}
    data_dict['train'] = {}
    data_dict['val'] = {}
    data_dict['test'] = {}
    
    hf = h5py.File(HDF5_path, 'r')

    print(hf)

    data_dict['train']['bbox'] = hf['train_bbox'][:]
    data_dict['train']['bbox_image'] = hf['train_bbox_image'][:]
    data_dict['train']['object_class'] = hf['train_object_class'][:]
    data_dict['train']['action_class'] = hf['train_action_class'][:]

    data_dict['val']['bbox'] = hf['val_bbox'][:]
    data_dict['val']['bbox_image'] = hf['val_bbox_image'][:]
    data_dict['val']['object_class'] = hf['val_object_class'][:]
    data_dict['val']['action_class'] = hf['val_action_class'][:]

    data_dict['test']['bbox'] = hf['test_bbox'][:]
    data_dict['test']['bbox_image'] = hf['test_bbox_image'][:]
    data_dict['test']['object_class'] = hf['test_object_class'][:]
    data_dict['test']['action_class'] = hf['test_action_class'][:]

    print('train_bbox :', data_dict['train']['bbox'])
    print('train_bbox_image :', data_dict['train']['bbox_image'])
    print('train_object_class :', data_dict['train']['object_class'])
    print('train_action_class :', data_dict['train']['action_class'])

    print('val_bbox :', data_dict['val']['bbox'])
    print('val_bbox_image :', data_dict['val']['bbox_image'])
    print('val_object_class :', data_dict['val']['object_class'])
    print('val_action_class :', data_dict['val']['action_class'])

    print('test_bbox :', data_dict['test']['bbox'])
    print('test_bbox_image :', data_dict['test']['bbox_image'])
    print('test_object_class :', data_dict['test']['object_class'])
    print('test_action_class :', data_dict['test']['action_class'])

    return data_dict


def load_all_bbox_only(Num_bbox, json_path_list):
    ####### initial
    dataset_dict = {}
    
    ####### load train, validation, test
    train_dict = load_splited_bbox_only(Num_bbox, json_path_list[0])
    val_dict = load_splited_bbox_only(Num_bbox, json_path_list[1])
    test_dict = load_splited_bbox_only(Num_bbox, json_path_list[2])

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


def load_splited_bbox_only(Num_bbox, load_file_path):
    print('**********************************************************************************')
    print('compare : act path :', '/home/dblab/maeng_space/dataset/deetas/data_00_00_00/json_act', '\n')
    print(load_file_path)

    ### define list to return
    bboxes_list = []
    object_class_list = []
    action_class_list = []
    image_ids_list = []
    output_data_dict = {}

    ### load pickle data
    pickle_file = open(load_file_path, "rb")
    tracks_list = pickle.load(pickle_file)

    for idx_anno, annotations_list in enumerate(tracks_list):
        num_annotation = len(annotations_list) - 1
        # if num_annotation < Num_bbox:
        #     continue

        ####### convert to batch
        batchs_list = []
        if num_annotation < Num_bbox:
            batch_data = []
            try:
                object_class = annotations_list[0]
            except:
                print('error : load object class')
                print(annotations_list)
                exit()
                
            batch_data.append(object_class)
            batch_data.extend(annotations_list[1:num_annotation+1])
            for idx_padding in range(Num_bbox-num_annotation):
                batch_data.append(annotations_list[-1])
            batchs_list.append(batch_data)

        else:
            for idx_anno in range(num_annotation - Num_bbox + 1):
                if idx_anno == 0:
                    object_class = annotations_list[0]
                    continue
                batch_data = []
                batch_data.append(object_class)
                batch_data.extend(annotations_list[idx_anno:idx_anno+5])
                batchs_list.append(batch_data)

        ####### convert to splited dataset
        for idx_batch, batch_data in enumerate(batchs_list):
            temp_act = []
            temp_image_id = []
            temp_bbox = []
            for annotation in batch_data[1:]:
                action_class = annotation[0]
                image_id = annotation[1]
                bboxes = annotation[2]

                temp_act.append(action_class)
                temp_image_id.append(image_id)
                temp_bbox.append(bboxes)

            temp_act = list(set(temp_act))
            if len(temp_act) > 1:
                print(temp_act)
                print("error : action_class")
                exit()

            object_class = batch_data[0]
            action_class = temp_act[0]
            if class_cm.check_act_class(action_class):
                continue
            bboxes = temp_bbox
            image_ids = temp_image_id

            object_class_list.append(object_class)
            action_class_list.append(class_cm.convert_action_class(action_class))
            image_ids_list.append(image_ids)
            bboxes_list.append(bboxes)

    index_list = np.arange(len(object_class_list))
    random.seed(777)
    random.shuffle(index_list)

    bbox_np = np.array(bboxes_list)
    bbox_np = bbox_np[index_list]
    output_data_dict['bbox'] = bbox_np

    image_id_np = np.array(image_ids_list)
    image_id_np = image_id_np[index_list]
    output_data_dict['image_id'] = image_id_np

    object_class_np = np.array(object_class_list)
    object_class_np = object_class_np[index_list]
    output_data_dict['object_class'] = object_class_np

    action_class_np = np.array(action_class_list)
    action_class_np = action_class_np[index_list]
    output_data_dict['action_class'] = action_class_np

    output_data_dict['object_class'] = object_class_np
    output_data_dict['bbox'] = bbox_np
    output_data_dict['image_id'] = image_id_np
    output_data_dict['action_class'] = action_class_np

    # check_list = set(action_class_list)
    # print(check_list)

    return output_data_dict


###################################################################################################################
# load act_class, bbox, image(processed)
###################################################################################################################
def load_all_with_image(Num_bbox, path_dict, model_config_dict):
    ####### initial
    dataset_dict = {}
    image_root_path = path_dict['image_root_path']
    json_path_list = path_dict['input_path_list']
    
    ####### load train, validation, test
    train_dict = load_splited_with_image(Num_bbox, json_path_list[0], path_dict, model_config_dict)
    val_dict = load_splited_with_image(Num_bbox, json_path_list[1], path_dict, model_config_dict)
    test_dict = load_splited_with_image(Num_bbox, json_path_list[2], path_dict, model_config_dict)

    train_bbox = train_dict['bbox']
    train_bbox_image = train_dict['bbox_image']
    train_object_class = train_dict['object_class']
    train_action_class = train_dict['action_class']

    val_bbox = val_dict['bbox']
    val_bbox_image = val_dict['bbox_image']
    val_object_class = val_dict['object_class']
    val_action_class = val_dict['action_class']
    
    train_bbox = np.concatenate((train_bbox, val_bbox), axis=0)
    train_bbox_image = np.concatenate((train_bbox_image, val_bbox_image), axis=0)
    train_object_class = np.concatenate((train_object_class, val_object_class), axis=0)
    train_action_class = np.concatenate((train_action_class, val_action_class), axis=0)
    train_action_class = one_hot(train_action_class)

    test_bbox = test_dict['bbox']
    test_bbox_image = test_dict['bbox_image']
    test_object_class = test_dict['object_class']
    test_action_class = test_dict['action_class']
    test_action_class = one_hot(test_action_class)

    print('train_bbox :', train_bbox.shape)
    print('train_bbox_image :', train_bbox_image.shape)
    print('train_object_class :', train_object_class.shape)
    print('train_action_class :', train_action_class.shape)
    
    print('test_bbox :', test_bbox.shape)
    print('test_bbox_image :', test_bbox_image.shape)
    print('test_object_class :', test_object_class.shape)
    print('test_action_class :', test_action_class.shape)

    dataset_dict['train_bbox'] = train_bbox
    dataset_dict['train_bbox_image'] = train_bbox_image
    dataset_dict['train_object_class'] = train_object_class
    dataset_dict['train_action_class'] = train_action_class

    dataset_dict['test_bbox'] = test_bbox
    dataset_dict['test_bbox_image'] = test_bbox_image
    dataset_dict['test_object_class'] = test_object_class
    dataset_dict['test_action_class'] = test_action_class

    return dataset_dict


def load_dataset_pickle(model_config_dict, path_dict):
    print('**********************************************************************************')
    print('load_dataset_pickle :', '\n',)
    ####### initial
    act_class_root_path = path_dict['ConvLSTM_path']
    pickle_filenames = path_dict['pickle_filenames']
    Num_bbox = model_config_dict['Num_bbox']

    pickle_paths = []
    for pickle_filename in pickle_filenames:
        pickle_path = os.path.join(act_class_root_path, pickle_filename)
        pickle_paths.append(pickle_path)

    output_dict = {}
    check_num_image = []

    ####### load pickle
    load_json_train_path = os.path.join(pickle_paths[0])
    load_json_val_path = os.path.join(pickle_paths[1])
    load_json_test_path = os.path.join(pickle_paths[2])
    
    for pickle_path in pickle_paths:
        ####### load pickle data
        pickle_pointer = open(pickle_path, "rb")
        tracks_list = pickle.load(pickle_pointer)
        num_track_after = 0
        batchs_list = []
        
        print(pickle_path)
        print('previous : num track : ', len(tracks_list),)

        ####### convert to batch (per track)
        for annotations_dict in tracks_list:
            ####### parent attribute
            annotation_id = annotations_dict['id']
            object_class = annotations_dict['category_id']
            annotations_list = annotations_dict['annotations']

            batchs_in_track = []
            num_annotation = len(annotations_list)

            ####### generate batch
            if num_annotation < Num_bbox:
                continue

                # batch_data = []
                # try:
                #     object_class = annotations_list[0]
                # except:
                #     print(annotations_list)
                # batch_data.append(object_class)
                # batch_data.extend(annotations_list[1:num_annotation+1])
                # for idx_padding in range(Num_bbox-num_annotation):
                #     batch_data.append(annotations_list[-1])
                # batchs_list.append(batch_data)

            else:
                ####### per batch in track
                for idx_anno in range(num_annotation - Num_bbox + 1):
                    batch_dataset = []
                    batch_bbox = []
                    batch_action = []
                    window_sliding = annotations_list[idx_anno : idx_anno + 5]
                    for window in window_sliding:
                        # image_id = window['image_id']
                        # file_name = window['image_file_name']
                        bbox = window['bbox']
                        action_class = window['Status']

                        batch_bbox.append(bbox)
                        batch_action.append(action_class)

                    check_action_different = list(set(batch_action))
                    if len(check_action_different) > 1:
                        continue
                    else:
                        action_class = check_action_different[0]
                        batch_dataset.append(object_class)
                        if class_cm.check_act_class(action_class):
                            print('error : action class', action_class)
                            continue
                        action_class = class_cm.convert_action_class(action_class)
                        batch_dataset.append(action_class)
                        batch_dataset.append(batch_bbox)
                        batchs_in_track.append(batch_dataset)
            
                if batchs_in_track:
                    num_track_after += 1
                    batchs_list.extend(batchs_in_track)

                ####### check num image
                for annotation in annotations_list:
                    image_id = annotation['image_id']
                    check_num_image.append(image_id)

        print('after : num track : ', num_track_after, '\n',)

        ####### convert to dataset for model (per-batch)
        splited_name = pickle_path[pickle_path.rfind('/')+1:-7]
        output_dict[splited_name] = {}
        bbox_list = []
        object_class_list = []
        action_class_list = []
        for batch_data in batchs_list:
            object_class = batch_data[0]
            action_class = batch_data[1]
            bbox = batch_data[2]

            bbox_list.append(bbox)
            object_class_list.append(object_class)
            action_class_list.append(action_class)

        index_list = np.arange(len(bbox_list))
        random.seed(777)
        random.shuffle(index_list)

        ####### shuffle
        bbox_np = np.array(bbox_list)
        bbox_np = bbox_np[index_list]
        output_dict[splited_name]['bbox'] = bbox_np

        object_class_np = np.array(object_class_list)
        object_class_np = object_class_np[index_list]
        output_dict[splited_name]['object_class'] = object_class_np

        action_class_np = np.array(action_class_list)
        action_class_np = action_class_np[index_list]
        output_dict[splited_name]['action_class'] = action_class_np

    num_image_list = list(set(check_num_image))
    print('after : num image : ', len(num_image_list), '\n',)

    return output_dict


###################################################################################################################
# write result
###################################################################################################################
def write_results_accuracy(metric_list, model_conifg_dict, csv_writer):
    test_loss, test_accuracy = metric_list

    idx_repeat = model_conifg_dict['idx_repeat']

    csv_writer.writerow({'idx_repeat': idx_repeat, 'Loss': test_loss, 'Accuracy': test_accuracy})


###################################################################################################################
# random data
###################################################################################################################
def generate_random_data(num_sample, Num_bbox):
    ### define algorithm
    bbox = np.random.random((num_sample, Num_bbox, 4)).astype(np.float64)
    object_class = np.random.random((num_sample, 1)).astype(np.float64)
    action_class = np.random.random((num_sample, 1)).astype(np.float64)

    return bbox, object_class, action_class

###################################################################################################################
# other
###################################################################################################################
def check_to_exist_file(num_delay, train_path):
    if os.path.isfile(train_path):
        for count in range(num_delay):
            print('already exist file check your model to train : ', num_delay - count)
            time.sleep(1)