from lib.yolo.libdarknet_wrapper import *

import os
import cv2


class Detector:
    config_path = ""
    weight_path = ""
    meta_path = ""
    thresh = 0
    hier_thresh = 0
    nms = 0
    debug = False

    meta_main = None
    net_main = None
    alt_names = None

    frame = []
    detections = []

    def __init__(self, config_path="./cfg/yolov3.cfg", weight_path="yolov3.weights", meta_path="./data/coco.data",
                 thresh=.5, hier_thresh=.5, nms=.45):
        self.config_path = config_path
        self.weight_path = weight_path
        self.meta_path = meta_path
        self.thresh = thresh
        self.hier_thresh = hier_thresh
        self.nms = nms

    def loadNet(self):
        assert 0 < self.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
        if not os.path.exists(self.config_path):
            raise ValueError("Invalid config path `" + os.path.abspath(self.config_path) + "`")
        if not os.path.exists(self.weight_path):
            raise ValueError("Invalid weight path `" + os.path.abspath(self.weight_path) + "`")
        if not os.path.exists(self.meta_path):
            raise ValueError("Invalid data file path `" + os.path.abspath(self.meta_path) + "`")
        if self.net_main is None:
            self.net_main = load_net_custom(self.config_path.encode("ascii"), self.weight_path.encode("ascii"), 0,
                                            1)  # batch size = 1
        if self.meta_main is None:
            self.meta_main = load_meta(self.meta_path.encode("ascii"))
        if self.alt_names is None:
            # In Python 3, the metafile default access craps out on Windows (but not Linux)
            # Read the names file and create a list to feed to detect
            try:
                with open(self.meta_path) as meta_f_h:
                    meta_contents = meta_f_h.read()
                    import re
                    match = re.search("names *= *(.*)$", meta_contents, re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                names_list = namesFH.read().strip().split("\n")
                                self.alt_names = [x.strip() for x in names_list]
                    except TypeError:
                        pass
            except Exception:
                pass

            return self.net_main, self.meta_main, self.alt_names

    def detect(self):
        custom_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        custom_image = cv2.resize(custom_image, (lib.network_width(self.net_main), lib.network_height(self.net_main)),
                                  interpolation=cv2.INTER_LINEAR)

        im, arr = array_to_image(custom_image)
        if self.debug: print("Loaded image")
        num = c_int(0)
        if self.debug: print("Assigned num")
        pnum = pointer(num)
        if self.debug: print("Assigned pnum")
        predict_image(self.net_main, im)
        if self.debug: print("did prediction")
        dets = get_network_boxes(self.net_main, self.frame.shape[1], self.frame.shape[0], self.thresh, self.hier_thresh,
                                 None, 0,
                                 pnum, 0)
        if self.debug: print("Got dets")
        num = pnum[0]
        if self.debug: print("got zeroth index of pnum")
        if self.nms:
            do_nms_sort(dets, num, self.meta_main.classes, self.nms)
        if self.debug: print("did sort")
        res = []
        if self.debug: print("about to range")
        for j in range(num):
            if self.debug: print("Ranging on " + str(j) + " of " + str(num))
            if self.debug: print("Classes: " + str(self.meta_main), self.meta_main.classes, self.meta_main.names)
            for i in range(self.meta_main.classes):
                if self.debug: print(
                    "Class-ranging on " + str(i) + " of " + str(self.meta_main.classes) + "= " + str(dets[j].prob[i]))
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox

                    draw_bbox(im, b, 255, 255, 0)
                    if self.alt_names is None:
                        nameTag = self.meta_main.names[i]
                    else:
                        nameTag = self.alt_names[i]
                    if self.debug:
                        print("Got bbox", b)
                        print(nameTag)
                        print(dets[j].prob[i])
                        print((b.x, b.y, b.w, b.h))

                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        if self.debug: print("did range")
        res = sorted(res, key=lambda x: -x[1])
        if self.debug: print("did sort")
        # free_image(im)
        if self.debug: print("freed image")
        free_detections(dets, num)
        if self.debug: print("freed detections")
        # init detections
        self.detections = res
        return res

    def getMatches(self):
        detected_images = []
        for box in self.detections:
            # extract detection parameters
            class_name = box[0]
            prob = box[1]
            bounds = box[2];
            top, left, bottom, right = box_to_rec(bounds)

            # extract cropped image from actual frame
            detected_image = self.frame[left:right, top:bottom]
            detected_images.append(detected_image)

        return detected_images
