import numpy as np
import matplotlib.pyplot as plt

def main():
    #number of ground truth bounding boxes in test dataset
    gt_num = 850

    #Intersections over unions and confidence scores for the three trained YOLOv3 models
    with open('eval-results\\yolo-lr-es-iou.npy', 'rb') as f:
        yolo_lr_es = np.load(f)
    with open('eval-results\\yolo-lr-es-scores.npy', 'rb') as f:
        yolo_lr_es_scores = np.load(f)

    with open('eval-results\\yolo-lr-iou.npy', 'rb') as f:
        yolo_lr = np.load(f)
    with open('eval-results\\yolo-lr-scores.npy', 'rb') as f:
        yolo_lr_scores = np.load(f)

    with open('eval-results\\yolo-iou.npy', 'rb') as f:
        yolo = np.load(f)
    with open('eval-results\\yolo-scores.npy', 'rb') as f:
        yolo_scores = np.load(f)


    yolo_versions = [yolo_lr_es, yolo_lr, yolo]
    yolo_versions_scores = [yolo_lr_es_scores, yolo_lr_scores, yolo_scores]
    yolo_titles = ['YOLOv3 with reducing learning rate and early stopping', 'YOLOv3 with reducing learning rate', 'YOLOv3']


    legend = []
    plt_count = 0
    for i in range(0, len(yolo_versions)):
        yolo = yolo_versions[i]
        precisions = []
        recalls = []
        thresh = 0.5
        TP = 0
        FP = 0
        total_detections = 0
        length = len(yolo)
        for iou in yolo:     
            if iou > thresh:
                TP += 1
            else:
                FP += 1
            total_detections += 1
            precisions.append(TP/total_detections)
            recalls.append(TP/gt_num)
        
        l, = plt.plot(recalls, precisions)
        legend.append(l)
        f_measure = 2 * ((recalls[-1] * precisions[-1]) / (precisions[-1] + recalls[-1]))
        print("F-measure:", f_measure)
        print("Precision:", precisions[-1])
        print("Recall:", recalls[-1])
        print("Avg IOU:", np.average(yolo))
        plt_count += 1
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.legend(legend, yolo_titles)
    
    
        
    plt.show()


if __name__ == '__main__':
    main()
