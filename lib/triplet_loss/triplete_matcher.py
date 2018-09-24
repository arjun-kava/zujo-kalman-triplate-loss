import keras
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, Lambda
from keras.models import Model
from keras import optimizers
import keras.backend as K
import numpy as np
from PIL import Image
import multiprocessing as mp
import math
from tqdm import tqdm
from annoy import AnnoyIndex
import json
import PIL
from functools import partial, update_wrapper
import os
import cv2


class TripletMatcher:

    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        self.model = self._get_model()
        self.image_width, self.image_height = (299, 299)  # input image size for network
        self.pool = mp.Pool(processes=8)
        self.proto_types = []

    def _get_model(self):
        no_top_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')

        # apply L2 normalization
        x = no_top_model.output
        x = Dense(512, activation='elu', name='fc1')(x)
        x = Dense(128, name='fc2')(x)
        x = Lambda(lambda x: K.l2_normalize(x, axis=1), name='l2_norm')(x)

        return Model(no_top_model.inputs, x)

    def preprocess_image_worker(self, image_mat):
        img = cv2.resize(image_mat, (self.image_width, self.image_height))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def batch_generator_predict(self, pool, batch_size, images):
        while True:
            result = [self.preprocess_image_worker(image) for image in images]
            X_batch = np.concatenate(result, axis=0)
            yield X_batch

    def set_proto_types(self, proto_types):
        # set proto types
        self.proto_types = proto_types

        # predict prototypes
        self.proto_pred = self.model.predict_generator(self.batch_generator_predict(self.pool, 32, self.proto_types),
                                                       math.ceil(len(self.proto_types) / 32), workers=1, max_q_size=1)

        print("len(self.proto_pred)", len(self.proto_pred))
        # add proto items into search index
        self.search_index = AnnoyIndex(128, metric='euclidean')
        for i in range(len(self.proto_pred)):
            self.search_index.add_item(i, self.proto_pred[i])

            # build indexing
        self.search_index.build(50)

    def is_match(self, captured_image):
        width, height, channel = captured_image.shape
        if height > 0 and width > 0:
            # predict captured images
            """
            captured_pred = self.model.predict(self.preprocess_image_worker(captured_image))
            """
            # predict prototypes

            captured_pred = self.model.predict_generator(
                self.batch_generator_predict(self.pool, 32, [captured_image]),
                math.ceil(len(captured_image) / 32), workers=1, max_q_size=1)
            """
            for index, proto_pred in enumerate(self.proto_pred):
               
                search_index = AnnoyIndex(1, metric='euclidean')
                search_index.add_item(0, proto_pred)
                search_index.add_item(1, captured_pred)

                annoy_distance = search_index.get_distance(0, 1)

                distance = np.linalg.norm(captured_pred - proto_pred)
                print("annoy_distance", annoy_distance)
                print("distance", distance)

                if distance < 0.65:
                    print("index", index)
                    print("distance", distance)

                    matching_proto = self.proto_types[index]

                    matching_proto = cv2.resize(matching_proto, (200, 300))
                    captured_image = cv2.resize(captured_image, (200, 300))
                    concat = np.concatenate((matching_proto, captured_image), axis=1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10, 500)
                    fontScale = 1
                    fontColor = (255, 255, 255)
                    lineType = 2

                    cv2.putText(concat, ('distance: ' + str(distance)),
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)
                    cv2.imshow("matched proto", concat)
                """
            # get distance between proto and captured predictions
            results, distances = self.search_index.get_nns_by_vector(captured_pred[0], 100, include_distances=True)
            print("distances", distances)

            # display matched proto types
            matching_proto = self.proto_types[results[0]]
            if matching_proto is not None:
                matching_proto = cv2.resize(matching_proto, (200, 300))
                captured_image = cv2.resize(captured_image, (200, 300))
                concat = np.concatenate((matching_proto, captured_image), axis=1)
                cv2.imshow("matched proto", concat)
