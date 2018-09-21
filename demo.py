from lib.video_streamer import *
import os

# get root execution path
cwd = os.getcwd()



# darknet yolo configuration settings
thresh = 0.6
config_path = os.path.join(cwd, "cfg/suit-tiny.test.cfg")
weight_path = os.path.join(cwd, "cfg/suit-tiny.weights")
meta_path = os.path.join(cwd, "cfg/suit-tiny.data")

# video configuration settings
proto_path = os.path.join(cwd, "assets/videos/neilman/")

# Init video constructor with suit config
video = VideoStreamer(proto_path, config_path, weight_path, meta_path, thresh)
video.start()
apparels = []
detections = []
while video.counter < video.frame_count:
    frame = video.read()

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.stop()
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
