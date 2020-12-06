import numpy as np
import os
import cv2
import time


def prepare_yolo_dataset(img_path, annot_path, file_name):
    f = open(file_name + ".txt", "w")
    for filename in os.listdir(annot_path):
        img = cv2.imread(os.path.join(annot_path,filename))

        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(img, 254, 255)
            contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            boxes = ""
            h, w = gray.shape
            for c in contours:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                Xs = [i[0] for i in box]
                Ys = [i[1] for i in box]
                x1 = min(Xs)
                x2 = max(Xs)
                y1 = min(Ys)
                y2 = max(Ys)

                boxes += " " + str(int(x1)) + "," + str(int(y1)) + "," + str(int(x2)) + "," + str(int(y2)) + ",0"
            f.write(img_path + filename + boxes + "\n")
    f.close()
   

def main():
    train_folder_path = 'AWE\\train\\'
    train_folder_annot_path = 'AWE\\trainannot_rect'
    test_folder_path = '\AWE\\test\\'
    test_folder_annot_path = 'AWE\\testannot_rect'

    prepare_yolo_dataset(train_folder_path, train_folder_annot_path, "train_data")
    prepare_yolo_dataset(test_folder_path, test_folder_annot_path, "test_data")
    
    
    



    
if __name__ == "__main__":
    main()