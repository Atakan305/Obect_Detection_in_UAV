# %% 1. Bölüm
import imutils
import time
import cv2
import numpy as np
#from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse

frame_count = 0 
all_frames = []
start = time.time()


cap = cv2.VideoCapture("video.mp4")


while True:
    ret, frame = cap.read()  # eger frameler doğru okunursa, true döndür
    frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame,(720,576))
    
    
    frame_width = frame.shape[1]
    frame_height =frame.shape[0]
    
    cv2.line(frame, (355, 288), (365, 288), (0,255,255),2)
    cv2.line(frame, (360, 283), (360, 293), (0,255,255),2)
    cv2.rectangle(frame,(0,0),(720,576),(75,124,96),10) 
    cv2.rectangle(frame,(180,58),(540,518),(160,47,114),3) 
    #cv2.line(frame, (195,20),(185,210), (255,0,0),)
    # 640,480
    #cv2.line(frame, (320,240), 3, (0,0,255),-1)
    
    
    frame_blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    
    all_frames.append(frame)
    frame_count = frame_count + 1
    
    labels = ["hedef IHA"]

    colors = ["0,0,255"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors, (18, 1))

    model = cv2.dnn.readNetFromDarknet("custom.cfg","iha.weights")
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layers = model.getLayerNames()
    output_layer = [layers[layer[0] - 1] for layer in model.getUnconnectedOutLayers()]

    model.setInput(frame_blob)

    detection_layers = model.forward(output_layer)

    ############## NON-MAXIMUM SUPPRESSION - OPERATION 1 ###################

    ids_list = []  # en yüksek oranlı değerin id'si
    boxes_list = []  # kutuların koordinatları
    confidences_list = []  # en yüksek oranlı değer, lebel yüzdesi

    ############################ END OF OPERATION 1 ########################

    for detection_layer in detection_layers:
        for object_detection in detection_layer:

            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]

            if confidence > 0.70:
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

                start_x = int(box_center_x - (box_width / 2))
                start_y = int(box_center_y - (box_height / 2))
                                              
                cv2.line(frame, (box_center_x, box_center_y), (360,288), (255,255,255),2)
                cv2.circle(frame, (box_center_x,box_center_y), 3, (0,255,255),-1)
                                
                s = "x: {}, y: {}, width: {}, height: {}".format(np.round(start_x),np.round(start_y),np.round(box_width),np.round(box_height))
                print(s)
                
                font = cv2.FONT_HERSHEY_SIMPLEX 
                cv2.putText(frame,'TESPIT EDILEN NESNE: IHA', (100,22), font, 0.4, (255,255,255),2,cv2.LINE_AA)
                cv2.putText(frame,'UCUS MODU: OTONOM', (290,22), font, 0.4, (255,255,255),2,cv2.LINE_AA)
                cv2.putText(frame,'TAKIP EDILIYOR', (450,22), font, 0.4, (255,255,255),2,cv2.LINE_AA)
                
                if box_center_x>=180 and box_center_x<=360 and box_center_y>58 and box_center_y<288:
                    print("hedef 2.bolgede")
                    #cv2.rectangle(frame,(160,48),(480,432),(0,0,255),1) 
                    
                elif box_center_x>360 and box_center_x<540  and box_center_y>58 and box_center_y<288:
                    print("hedef 1.bolgede")
                    #cv2.rectangle(frame,(160,48),(480,432),(0,0,255),1) 
                
                elif box_center_x>180 and box_center_x<360 and box_center_y>288 and box_center_y<518:
                    print("hedef 3.bolgede")
                    #cv2.rectangle(frame,(160,48),(480,432),(0,0,255),1) 
                    
                elif box_center_x>360 and box_center_x<540 and box_center_y>288 and box_center_y<518:
                    print("hedef 4. bolgede")
                    #cv2.rectangle(frame,(160,48),(480,432),(0,0,255),1) 
                    
                    
                
                ############## NON-MAXIMUM SUPPRESSION - OPERATION 2 ###################

                ids_list.append(predicted_id)  # id_list'e, yüksek skorlu, seçilen id'yi ekle
                confidences_list.append(float(
                    confidence))  # yüksek skorlu id'yi, predicted_id'nin "değerini", confindences_list'e ekle (yüzde)
                boxes_list.append([start_x, start_y, int(box_width),
                                   int(box_height)])  # boxes_list'e başlangç ve genişlik, yükseklik değerlerini ekle

                ############################ END OF OPERATION 2 ########################

    ############## NON-MAXIMUM SUPPRESSION - OPERATION 3 ###################

    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
    # en yüksek id'li, tespit edilen nesnelerin boxes_list'teki boyutları, nesnenin yüzdelerini atıyoruz.

    for max_id in max_ids:  # max_id,

        max_class_id = max_id[0]  # max_id'nin sıfırıncı indexi,
        box = boxes_list[max_class_id]  # boundingBox'ların özellikleri
        # box'ın ilk 3 indexi, srasıyla x başlangıç, y başlangıç,  genişlik, yükseklik değerlerini verir
        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_id = ids_list[max_class_id]  # nesnenin
        label = labels[
            predicted_id]  # seçilen id'nin etiketini, belirttiğimiz label listesinden, predicted_id'nin etiketini alıyoruz
        confidence = confidences_list[
            max_class_id]  # yüksek yüzdeliler arasından, confidences_list'ten, max_class_id olanı seçiyoruz. yani en yüksek yüzdeli olanı

        ############################ END OF OPERATION 3 ########################

        end_x = start_x + box_width  # bitiş x koordinatı, başlangıçX ve genişlik değerlerinin toplamı
        end_y = start_y + box_height  # bitiş y koordinatı, başlangıçY ve yükseklik değerlerinin toplamı

        box_color = colors[predicted_id]  # kutunun rengini, colors listesinden, predicted_id'nin olduğu rengi atıyoruz
        box_color = [int(each) for each in box_color]  #

        label = "{}: {:.2f}%".format(label, confidence * 100)
        print("predicted object {}".format(label))
        if (box_center_x<180 or box_center_y<58 or box_center_x>540 or box_center_y>518):
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (57,239,249), 2)
        else:
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,0,255), 2)
            
        
        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    cv2.imshow("Detection Window", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
end = time.time()
print("-------------------------------------")
print("frame count: ",frame_count)
#print(len(all_frames))
print("ens-start: ",end - start)
fps = (frame_count / (end - start))
print("FPS: ",fps)



cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows