from lib.yolo.detector import *
from threading import Thread, Lock
import cv2
import os
import numpy as np
from lib.deep_sort import preprocessing
from lib.deep_sort import nn_matching
from lib.deep_sort.detection import Detection
from lib.deep_sort.tracker import Tracker
from lib.tools import generate_detections as gdet


class VideoStreamer:
    def __init__(self, proto_path="", config_path="", weight_path="",
                 meta_path="", thresh=.5, width=320, height=320, fps=0.0):

        # set all properties
        self.protoPath = proto_path

        # get capture from source
        src = self.get_src_video()
        self.stream = cv2.VideoCapture(src)

        # get properties of stream
        self.file_name, self.file_extension = os.path.splitext(src.split("/").pop())
        self.frame_count = self.stream.get(cv2.CAP_PROP_FRAME_COUNT)
        self.stream.set(int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)), width)
        self.stream.set(int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)), height)
        self.stream.set(int(self.stream.get(cv2.CAP_PROP_FPS)), fps)

        # read sample stream
        (self.grabbed, self.frame) = self.stream.read()

        # init other params
        self.started = False
        self.read_lock = Lock()
        self.counter = 1

        # init network params
        self.config_path = config_path
        self.weight_path = weight_path
        self.meta_path = meta_path
        self.thresh = thresh

        self.detected_frames = []
        # init detector with configuration
        self.detector = Detector(self.config_path, self.weight_path, self.meta_path, self.thresh)
        # Load targeted network
        self.detector.loadNet()

        # Tracking objects
        # Definition of the parameters
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 1.0

        # deep_sort
        self.deep_sort_model_file = 'cfg/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.deep_sort_model_file, batch_size=1)

        # initilize tracker
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(metric)

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            if self.counter < self.frame_count:
                self.counter += 1
                (grabbed, frame) = self.stream.read()

                # detect objects and frames

                self.read_lock.acquire()
                self.grabbed, self.frame = grabbed, frame
                self.read_lock.release()
            else:
                self.started = False
                self.stream.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()

        # detect objects
        self.detector.frame = frame
        detections = self.detector.detect()
        apparels = self.get_detections(frame, detections)
        #self.drawRect(frame, detections)

        self.detected_frames.append({
            "frame": frame,
            "number": self.counter,
            "detections": detections,
            "apparels": apparels
        })

        # convert detections to bbox
        index_boxs = 2
        boxs = [detection[index_boxs] for detection in detections];

        # encode features and track boxes
        features = self.encoder(frame, boxs)
        self.track(frame, boxs, features)

        # release locked frame
        self.read_lock.release()

        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def drawRect(self, frame, detections):
        for box in detections:
            # extract detection parameters
            class_name = box[0]
            prob = box[1]
            bounds = box[2];
            top, left, bottom, right = box_to_rec(bounds)

            # extract frame parameters
            detected_image = frame[left:right, top:bottom]
            height, width, channels = detected_image.shape

            is_detection_found = detected_image is not None and height > 0 and width > 0
            if is_detection_found:
                label = class_name + ": " + str(prob)
                cv2.rectangle(frame, (top, left), (bottom, right), (0, 255, 0), 3)
                self.put_text(frame, label, bottom, left)

    def get_detections(self, frame, detections):
        self.detected_apparels = []
        for box in detections:
            # extract detection parameters
            class_name = box[0]
            prob = box[1]
            bounds = box[2];
            top, left, bottom, right = box_to_rec(bounds)

            # extract frame parameters
            detected_image = frame[left:right, top:bottom]
            height, width, channels = detected_image.shape

            is_detection_found = detected_image is not None and height > 0 and width > 0
            if is_detection_found:
                # crop detected portion from actual image
                detected = frame[left:right, top:bottom]
                self.detected_apparels.append(detected)

    def put_text(self, mat, label, bottom, left):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (bottom, left)
        font_scale = 1
        font_color = (0, 255, 0)
        line_type = 2
        cv2.putText(mat, label,
                    bottom_left_corner_of_text,
                    font,
                    font_scale,
                    font_color,
                    line_type)

    def get_src_video(self):
        return [os.path.join(self.protoPath, file) for file in os.listdir(self.protoPath) if
                file.endswith(".mp4") or file.endswith(".avi")][0]

    def get_src_proto(self):
        return [os.path.join(self.protoPath, file) for file in os.listdir(self.protoPath) if
                file.endswith(".jpg") or file.endswith(".jpeg")]

    def track(self, frame, boxs, features):
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]


        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        for track in self.tracker.tracks:
            if track.is_confirmed() and track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            top, left, bottom, right = box_to_rec(bbox)
            cv2.rectangle(frame, (top, left), (bottom, right), (0, 255, 0), 3)
            self.put_text(frame, str(track.track_id), bottom, left)


    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()
