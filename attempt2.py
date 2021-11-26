import cv2
import numpy as np
import matplotlib.pyplot as plt

yolo = cv2.dnn.readNet("./yolov3_training_last_1000.weights", "./yolov3_testing.cfg")
classes = ["garbage"]
img = glob.glob("C:/archit_3/yolo-trash/yolo_custom_detection/images/00a40ac0-1a14-4b3a-a1ce-d8c89d37d1cd_6afde6c9-bbbb-47c6-acf5-8b0392c59fba.jpg")
# img = cv2.dnn.readNetFromTensorflow("./files_for_pb/retrained_graph.pb", "./files_for_pb/retrained_labels.pbtxt")
blob = cv2.dnn.blobFromImage(img,1/255,(320,320),(0,0,0),swapRB=True,crop=False)

i= blob[0].reshape(320,320,3)
yolo.setInput(blob)
output_layers_name = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layers_name)


boxs = []
confidences = []
class_ids = []
for output in layeroutput:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.7:
            center_x = int(detection[0]*width)
            center_x = int(detection[0]*height)
            w = int(detection[0]*width)
            h = int(detection[0]*height)

            x = int(center_x - w/2)
            x = int(center_y - h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxs,confidences,0.5,0.4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size=(len(boxs),3))

if len(indexes) > 0:
    for i in indexes.flatten():
        x,y,w,h = boxs[i]

        label = str(classes[class_ids[i]])
        confi = str(round(confidences[i],2))
        color = colors[i]

        cv2.rectangle(img,(x,y),(x+w,y+h),color,1)
        cv2.putText(img,label+" "+confi,(x,y+20),font,2,(255,255,255),5)
cv2.imwrite("./image111.jpg",img)