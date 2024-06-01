# Description: This file is used to detect objects in a video file using the YOLOv3 model.
from imageai.Detection import VideoObjectDetection
import os

os.system("pip install tensorflow==2.16.1")
os.system("pip install keras==2.4.3 numpy==1.19.3 pillow==7.0.0 scipy==1.4.1 h5py==2.10.0 matplotlib==3.3.2 opencv-python keras-resnet==0.2.0")
os.system("pip install imageai --upgrade")
execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join( execution_path, "traffic-mini.mp4"),
                                             output_file_path=os.path.join(execution_path, "traffic_mini_detected_1")
                                             , frames_per_second=29, log_progress=True)
print(video_path)