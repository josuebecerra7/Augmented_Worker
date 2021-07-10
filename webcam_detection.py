# Josue Becerra Rico

# Import libraries
import cv2
import matplotlib.pyplot as plt

# Import coco trained model
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'


# Open labels file
classLabels = []
file_name = 'coco.names'
with open(file_name,'rt') as ftp:
    classLabels = ftp.read().rstrip('\n').split('\n')
    
# Configure model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Resize the images according to the configuration file (320x320)pixels
model.setInputSize(320,320)
model.setInputScale(1.0/127.5) # 255/2 = 127.5
model.setInputMean((127.5,127.5,127.5)) # mobilenet takes [-1,1]
model.setInputSwapRB(True)

cap = cv2.VideoCapture(0)

# Check if the video is opened
if not cap.isOpened():
    raise IOError("Cannot open video")
    
font_scale = 1
font = cv2.FORMATTER_FMT_PYTHON

while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.45)
    
 #   print(ClassIndex)
    if (len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd<=91):
                cv2.rectangle(frame, boxes, (255,0,0), 2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(15,200,10), thickness=3)
    cv2.imshow('Object Detection', frame)
    # Press q to quit
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()