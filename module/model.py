import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, LSTM, ConvLSTM2D, MaxPooling3D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical as one_hot


###################################################################################################################
# model
###################################################################################################################
def LSTM_model(Num_bbox, Num_class):
    bounding_box = Input(shape = (Num_bbox, 4))
    object_class = Input(shape=(1))

    ####### bounding box route
    bbox_FM = LSTM(16, activation = 'relu', return_sequences=True)(bounding_box)
    bbox_FM = LSTM(32, activation = 'relu', return_sequences=True)(bbox_FM)
    bbox_FM = LSTM(64, activation = 'relu')(bbox_FM)

    ####### object class route
    class_FM = Dense(16, activation = 'relu')(object_class)
    class_FM = Dense(32, activation = 'relu')(class_FM)

    ####### concat
    concat_FM = concatenate([bbox_FM, class_FM])

    concat_FM = Dense(128, activation='relu')(concat_FM)
    concat_FM = Dense(64, activation='relu')(concat_FM)
    output_FM = Dense(Num_class, activation='sigmoid')(concat_FM)

    model = Model(inputs=[bounding_box, object_class], outputs=output_FM)

    return model


###################################################################################################################
# model
###################################################################################################################
def ConvLSTM_model(model_config_dict):
    print(model_config_dict.keys())
    
    Num_bbox = model_config_dict['Num_bbox']
    Num_class = model_config_dict['Num_class']
    
    image_width = model_config_dict['width']
    image_height = model_config_dict['height']
    image_channel = model_config_dict['channel']

    ####### bounding box route
    bounding_box = Input(shape = (Num_bbox, 2, 2, 1))

    bbox_FM = ConvLSTM2D(filters=8, kernel_size=(2, 2), padding="same", data_format='channels_last',
                            return_sequences=True, activation="relu")(bounding_box)
    bbox_FM = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(bbox_FM)

    bbox_FM = ConvLSTM2D(filters=16, kernel_size=(2, 2), padding="same", data_format='channels_last',
                            return_sequences=True, activation="relu")(bbox_FM)
    bbox_FM = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(bbox_FM)

    bbox_FM = ConvLSTM2D(filters=32, kernel_size=(2, 2), padding="same", data_format='channels_last',
                            return_sequences=True, activation="relu")(bbox_FM)

    bbox_FM = Flatten()(bbox_FM)

    ####### object class route
    object_class = Input(shape=(1, 1))
    class_FM = Dense(16, activation='relu')(object_class)
    class_FM = Dense(32, activation='relu')(class_FM)
    class_FM = Flatten()(class_FM)

    ####### concat
    concat_FM = concatenate([bbox_FM, class_FM])

    concat_FM = Dense(128, activation='relu')(concat_FM)
    concat_FM = Dense(64, activation='relu')(concat_FM)
    output_FM = Dense(Num_class, activation='sigmoid')(concat_FM)

    ####### output
    model = Model(inputs=[bounding_box, object_class], outputs=output_FM)

    return model


###################################################################################################################
# model
###################################################################################################################
def ConvLSTM_model_with_image(model_config_dict):
    print(model_config_dict.keys())
    
    Num_bbox = model_config_dict['Num_bbox']
    Num_class = model_config_dict['Num_class']
    
    image_width = model_config_dict['width']
    image_height = model_config_dict['height']
    image_channel = model_config_dict['channel']

    ####### bounding box route
    bounding_box = Input(shape = (Num_bbox, 4))
    bbox_FM = LSTM(8, activation = 'relu', return_sequences=True)(bounding_box)
    bbox_FM = LSTM(16, activation = 'relu', return_sequences=True)(bbox_FM)
    bbox_FM = LSTM(32, activation = 'relu', return_sequences=True)(bbox_FM)
    bbox_FM = Flatten()(bbox_FM)

    ####### cropped image route
    image_in_bbox = Input(shape = (Num_bbox, image_width, image_height, image_channel))
    bb_img_FM = ConvLSTM2D(filters=8, kernel_size=(3, 3), padding="same", data_format='channels_last',
                            return_sequences=True, activation="relu")(image_in_bbox)

    bb_img_FM = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(bb_img_FM)

    bb_img_FM = ConvLSTM2D(filters=16, kernel_size=(3, 3), padding="same", data_format='channels_last',
                            return_sequences=True, activation="relu")(bb_img_FM)

    bb_img_FM = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(bb_img_FM)

    bb_img_FM = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", data_format='channels_last',
                            return_sequences=True, activation="relu")(bb_img_FM)

    # mid_img_FM = Flatten()(bb_img_FM) # take

    bb_img_FM = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(bb_img_FM)

    bb_img_FM = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", data_format='channels_last',
                            return_sequences=True, activation="relu")(bb_img_FM)

    bb_img_FM = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(bb_img_FM)

    bb_img_FM = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding="same", data_format='channels_last',
                            return_sequences=True, activation="relu")(bb_img_FM)

    bb_img_FM = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(bb_img_FM)

    bb_img_FM = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding="same", data_format='channels_last',
                            return_sequences=False, activation="relu")(bb_img_FM)

    bb_img_FM = Flatten()(bb_img_FM)

    ####### object class route
    object_class = Input(shape=(1, 1))
    class_FM = Dense(8, activation='relu')(object_class)
    class_FM = Dense(16, activation='relu')(class_FM)
    class_FM = Dense(32, activation='relu')(class_FM)
    class_FM = Flatten()(class_FM)

    ####### concat
    # concat_FM = concatenate([bbox_FM, bb_img_FM, mid_img_FM, class_FM])
    concat_FM = concatenate([bbox_FM, bb_img_FM, class_FM])

    concat_FM = Dense(128, activation='relu')(concat_FM)
    concat_FM = Dense(64, activation='relu')(concat_FM)
    output_FM = Dense(Num_class, activation='sigmoid')(concat_FM)

    ####### output
    model = Model(inputs=[bounding_box, image_in_bbox, object_class], outputs=output_FM)

    return model