
import os
from timeit import default_timer as timer

import numpy as np
#import tensorflow.compat.v1.keras.backend as K
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
import cv2
import tensorflow as tf

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

classes_path = 'classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
model_path = 'model_data/yolo.h5'
class_names = get_classes(classes_path)

num_classes = len(class_names)
anchors = get_anchors(anchors_path)
score = 0.3
iou = 0.2
input_image_shape = model_image_size = (416,416) # multiple of 32, hw

sess = K.get_session()

model_path = os.path.expanduser(model_path)
assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

# Load model, or construct model and load weights.
num_anchors = len(anchors)
num_classes = len(class_names)
is_tiny_version = num_anchors==6 # default setting
try:
    yolo_model = load_model(model_path, compile=False)
except:
    yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
        if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
    yolo_model.load_weights(model_path) # make sure model, anchors and classes match
else:
    assert yolo_model.layers[-1].output_shape[-1] == \
        num_anchors/len(yolo_model.output) * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes'

print('{} model, anchors, and classes loaded.'.format(model_path))

# Generate output tensor targets for filtered bounding boxes.
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = yolo_eval(yolo_model.output, anchors,
        len(class_names), input_image_shape,
        score_threshold=score, iou_threshold=iou)



def gt_boxes(annot_path, file_name):
    boxes = []
    #for filename in os.listdir(annot_path):
    img = cv2.imread(os.path.join(annot_path,file_name))
    
    if img is not None:
        img_boxes = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, 254, 255)
        contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
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
            img_boxes.append([y1,x1,y2,x2])
    
    return img_boxes

def eval(boxes1, boxes2):
    if (boxes1 == []):
        return []
    if (boxes2 == []):
        if (boxes1 != []):
            return np.zeros((1, len(boxes1)))
    IOUs = []
    for box1 in boxes1:
        bestIOU = 0
        for box2 in boxes2:
            ratio = iou_fun(box1, box2)
            if ratio > bestIOU:
                bestIOU = ratio
        IOUs.append(bestIOU)
    return IOUs

def iou_fun(box1, box2):
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])

    intersect = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    r_iou = intersect / float(box1Area + box2Area - intersect)
    return r_iou

def non_max_suppression(boxes, scores, threshold):
    boxes = np.array(boxes)
    scores = np.array(scores)
    if len(boxes) == 0:
        return [], []
    y1 = boxes[:,0]
    x1 = boxes[:,1]
    y2 = boxes[:,2]
    x2 = boxes[:,3]
    compared = np.arange(0,len(boxes))
    compared_scores = scores.copy()
    for i in range(0, len(boxes)):        
        if len(compared) > 0:  
            if (boxes[compared] == boxes[i]).any():   
            #if (boxes[compared] == boxes[i]).all(axis=1).any():
                temp = compared.copy()
                temp_scores = compared_scores.copy()

                for j in range(0, len(compared)):
                    if (boxes[i] != boxes[compared[j]]).all() and (boxes[temp] == boxes[compared[j]]).any():
                        ratio = iou_fun(boxes[i], boxes[compared[j]])
                        if ratio > threshold:
                            if scores[i] > compared_scores[j]:
                                idx = np.argwhere((boxes[temp]==boxes[compared[j]]))[0][0]
                                temp = np.delete(temp,idx)
                                temp_scores = np.delete(temp_scores, idx)
                            else:
                                idx = np.argwhere((boxes[temp]==boxes[i]))[0][0]
                                temp = np.delete(temp, idx)
                                temp_scores = np.delete(temp_scores, idx)
                                break
                            
                compared = temp
                compared_scores = temp_scores
    return boxes[compared], scores[compared]



def detect_image(image):
    start = timer()

    if model_image_size != (None, None):
        assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                            image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

    thickness = (image.size[0] + image.size[1]) // 300
    all_boxes = []
    all_scores = []
    # for i, c in reversed(list(enumerate(out_classes))):
    #     predicted_class = class_names[c]
    #     box = out_boxes[i]
    #     all_boxes.append(np.array(box))
    #     score = out_scores[i]
    #     all_scores.append(np.array(score))
    if(len(out_boxes)) == 0:
        best_boxes = []
        best_scores = []
    else:
        best_boxes = out_boxes
        best_scores = out_scores
    # Apply non-maxiumum suppression if IOU of bounding boxes is > 0.2
    #best_boxes, best_scores = non_max_suppression(all_boxes,all_scores,0.2)
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label)
        if (best_boxes == box).all(axis=1).any():
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i])
            del draw

    end = timer()
    print("Time:", end - start)
    return image, best_boxes, best_scores

def main():
    # testing on train set because of mixup
    test_path = 'D:\\Faks\\2.letnik\\SB\\Assignment_2\\AWE\\train\\'
    annot_path = 'AWE\\trainannot_rect\\'
    count = 1   
    IOUs = []
    best_scores = []
    gts = []
    
    for filename in os.listdir(test_path):
        img = Image.open(os.path.join(test_path,filename))
        if img.mode == 'L':
            img = np.stack((img,)*3, axis=-1)
            img = Image.fromarray(img)
        # detect ear in image
        img, boxes, scores = detect_image(img)
        img = np.array(img) 
        img = img[:, :, ::-1].copy()
        img_name = str(count).zfill(4)

        # get ground truth bounding boxes
        boxes2 = gt_boxes(annot_path, img_name + ".png")
        gts.append(len(boxes2))
        ratios = eval(boxes, boxes2)
        if len(ratios) > 1:
            prntratio = ""            
            for r in ratios:
                IOUs.append(r)
                prntratio += " " + str(r)
            for s in scores:
                best_scores.append(s)
            print("two ratios:", prntratio)
        else:
            if ratios == []:
                #false negative
                ratios = [0]
            else:
                IOUs.append(ratios[0])
                best_scores.append(scores[0])
                print("ratio: ", ratios[0])
        if len(ratios) < len(boxes2):
            num_zeros = len(boxes2) - len(ratios)
            #IOUs.append(0)
            #best_scores.append(0)
        # write image
        cv2.imwrite("results-images\\" + img_name + ".png", img)
        count += 1
        
    print(IOUs)
    with open('eval-results\\yolo-lr-es-iou.npy', 'wb') as f:
        np.save(f, IOUs)
    print(best_scores)
    with open('eval-results\\yolo--lr-es-scores.npy', 'wb') as f:
        np.save(f, IOUs)
    print("Average:", np.average(IOUs))
    print("Ground truth num:", np.sum(gts))
if __name__ == '__main__':
    main()
