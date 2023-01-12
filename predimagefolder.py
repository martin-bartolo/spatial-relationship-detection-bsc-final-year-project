import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import json
import sys
import cv2
import copy
from absl import app, flags, logging
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tensorflow_yolov4_tflite.detect import detect_frame
import tensorflow_yolov4_tflite.core.utils as utils
from tensorflow.python.saved_model import tag_constants
from sVOC2k_lib_lang import englishObjectList
from sVOC2k_lib_util import Object
from sVOC2k_lib_feat import compute_geometrical_features
from sVOC2k_lib_lang import eng_preposition_list

np.set_printoptions(threshold=sys.maxsize)

flags.DEFINE_string('classifier', './models/multilabel_model_partfeatures.sav', 'model to classify relationships')
flags.DEFINE_string('scaler', './models/scalar_partfeatures.sav', 'model to normalise features')
flags.DEFINE_string('yoloweights', './tensorflow_yolov4_tflite/checkpoints/yolov4-416', 'path to weights file')

def boxes_intersect(box1, box2):
    box1_area = (box1[2]-box1[0]+1) * (box1[3]-box1[1]+1)
    box2_area = (box2[2]-box2[0]+1) * (box2[3]-box2[1]+1)
    intersection_area = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1)
    union_area = float(box1_area + box2_area - intersection_area)

    if intersection_area/union_area >= 0.95: return True
    else: return False


def pred_dictionary(prediction):
    pred_dict = { 'above' : prediction[0],
                    'across' : prediction[1],
                    'against' : prediction[2],
                    'along' : prediction[3],
                    'around' : prediction[4],
                    'at_the_level_of' : prediction[5],
                    'behind' : prediction[6],
                    'beyond' : prediction[7],
                    'below' : prediction[8],
                    'far_from' : prediction[9],
                    'in' : prediction[10],
                    'in_front_of' : prediction[11],
                    'near' : prediction[12],
                    'next_to' : prediction[13],
                    'none' : prediction[14],
                    'on' : prediction[15],
                    'opposite' : prediction[16],
                    'outside_o' : prediction[17],
                    'under' : prediction[18]
                }
    return pred_dict

#return string of predictions with confidence higher than 0.5
def get_best_predictions(prediction):
    prepositions = []
    for p in prediction:
        if p >= 0.5:
            prepositions.append(eng_preposition_list[prediction.index(p)])
    return prepositions


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    classifier_path = flags.FLAGS.classifier
    scaler_path = flags.FLAGS.scaler
    weights_path = flags.FLAGS.yoloweights
    classifier_path = flags.FLAGS.classifier

    #load models
    scaler = pickle.load(open(scaler_path, 'rb'))
    classifier = pickle.load(open(classifier_path, 'rb'))
    saved_model_loaded = tf.saved_model.load(weights_path, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    ##load glove word embeddings
    #file = open("./tensorflow_yolov4_tflite/data/dataset/20-vectors-glove.txt", mode="r", encoding="utf-8")
    #lines = file.readlines()
    #glove_dict = {}

    #for line in lines:
    #    split_line = line.strip("\n").split(' ')
    #    vec = [float(i) for i in split_line[1:]]
    #    glove_dict[split_line[0]] = vec

    #initialise dictionary to store information
    dict = {}

    #iterate through videos in the folder
    print ("Found ", len(os.listdir("./testing/images")), " images")
    c = 0
    for image_path in os.listdir("./testing/images"):

        #load image
        image = cv2.imread("./testing/images/"+image_path)

        #get bounding boxes using YOLO
        pred_bbox, boxes, scores, classes, valid_detections, imHeight, imWidth = detect_frame(image, infer)

        #we remove any object with a confidence score less than 0.5 (not useful for us to annotate since the machine is not sufficiently sure about it)
        low_confidence_detections = 0
        new_scores = []
        new_boxes = []
        new_classes = []
        for i in range(valid_detections):
            if(scores[i] < 0.5):
                low_confidence_detections += 1
            else:
                new_scores.append(scores[i])
                new_boxes.append(boxes[i])
                new_classes.append(classes[i])

        valid_detections -= low_confidence_detections

        #we check for boxes with intersect more than 95% and in these intstances remove the prediction with lowest confidence (removes double predictions)
        indices_toremove = []
        intersecting_detections = 0
        for i in range(valid_detections):
            for j in range(valid_detections):
                if i == j: continue
                if boxes_intersect(new_boxes[i], new_boxes[j]):
                    if new_scores[i] > new_scores[j]: index_toremove = j
                    else: index_toremove = i
                    if index_toremove not in indices_toremove:
                        indices_toremove.append(index_toremove)
                        intersecting_detections += 1

        new_scores = [x for x in new_scores if new_scores.index(x) not in indices_toremove]
        new_classes = [x for x in new_classes if new_classes.index(x) not in indices_toremove]
        new_boxes = [x for x in new_boxes if new_boxes.index(x) not in indices_toremove]
        valid_detections -= intersecting_detections

        #we assign the remaining values to the pred_bbox list
        pred_bbox = [[new_boxes], [new_scores], [new_classes], [valid_detections]]
        new_boxes_copy = copy.deepcopy(new_boxes)#we kep a copy to save (since drawing the bboxes changes the values)
        new_classes_copy = copy.deepcopy(new_classes)#we kep a copy to save (since annotating the bounding boxes with classes changes the values on some occasions)

        #draw bounding boxes
        result = utils.draw_bbox(image, pred_bbox)
        imagename = "./testing/test"+str(c)+".jpg"
        cv2.imwrite(imagename, result)

        #resize image if it is too big to fit on screen
        newheight = result.shape[0]
        newwidth = result.shape[1]
        if(result.shape[0] > 1080):
            height = result.shape[0]
            width = result.shape[1]
            newheight = 1080
            newwidth = int(round((1-(np.abs(height-1080)/height))*width))
            dim = (newwidth, newheight)
            result = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)

        #display frame with all detected objects
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", result)
        cv2.waitKey(1)

        #create a list of the detected objects
        objects = []
        objects_tosave = [] #bounding boxes will not be changed in this one (we must change them in the other to scale with the image)

        #create objects
        for i in range(valid_detections):
            objects.append(Object(englishObjectList[int(new_classes[i])], new_boxes[i][0], new_boxes[i][1], new_boxes[i][2], new_boxes[i][3], []))
            objects_tosave.append(Object(englishObjectList[int(new_classes_copy[i])], new_boxes_copy[i][0], new_boxes_copy[i][1], new_boxes_copy[i][2], new_boxes_copy[i][3], []))

        #initialise list of relationships that will be filled with dictionaries of each pair
        relationships = []

        #generate_features from bounding boxes
        for i in range(len(objects)):
            for j in range(len(objects)):

                #make sure that we do not compare an object to itself
                if(i == j): continue

                feats, extrafeats = compute_geometrical_features(imWidth, imHeight, objects[i], objects[j])

                #filter out only features that we want since the model only uses some of them
                feats_filtered =   np.array([[feats["AreaObj1_Norm_wt_Union"],
                                    feats["AreaObj2_Norm_wt_Union"],
                                    feats["objAreaRatioTrajLand"],
                                    feats["DistBtCentr_Norm_wt_UnionBB"],
                                    feats["AreaOverlap_Norm_wt_Total"],
                                    feats["AreaOverlap_Norm_wt_Min"],
                                    feats["InvFeatXminXmin"],
                                    feats["InvFeatXmaxXmin"],
                                    feats["InvFeatYminYmin"],
                                    feats["InvFeatYmaxYmin"],
                                    feats["AspectRatioObj_1"],
                                    feats["AspectRatioObj_2"],
                                    feats["EuclDist_Norm_wt_UnionBB"],
                                    feats["unitVecTrajLand_Norm_wt_UnionBB_x"],
                                    feats["unitVecTrajLand_Norm_wt_UnionBB_y"],
                                    ]])

                ##add glove language features
                #full_feats = np.array([feats_filtered + glove_dict[objects[i].label]+glove_dict[objects[j].label]])

                #scale/normalise features
                feats_scaled = scaler.transform(feats_filtered)

                #Make predictions on the image
                prediction = classifier.predict_proba(feats_scaled)
                print(objects_tosave[i].label, i, prediction, objects_tosave[j].label, j)
            
                # write pair to the dictionary
                pair = {    'object1_label': objects_tosave[i].label + str(i),
                            'object1_confidence': scores[i],
                            'object1_bbox' : [objects_tosave[i].xmin, objects_tosave[i].ymin, objects_tosave[i].xmax, objects_tosave[i].ymax],
                            'object2_label': objects_tosave[j].label + str(j),
                            'object2_confidence': scores[j],
                            'object2_bbox' : [objects_tosave[j].xmin, objects_tosave[j].ymin, objects_tosave[j].xmax, objects_tosave[j].ymax],
                            'features' : feats_scaled.tolist(),
                            'best_prepositions': get_best_predictions(prediction[0].tolist()),
                            'prep_prediction': pred_dictionary(prediction[0].tolist())
                        }
                relationships.append(pair)

        file = {'height' : newheight,
                'width' : newwidth,
                'num_relationships' : (len(objects)**2)-len(objects),
                'relationships' : relationships
                }

        filename = "image_"+str(c)
        dict[filename] = file

        c += 1

    #save dictionary to file
    file = open('./testing/test.json', 'w+')
    json.dump(dict, file, indent = 2)
    file.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
