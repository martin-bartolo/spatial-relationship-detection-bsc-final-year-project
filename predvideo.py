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

flags.DEFINE_string('video', './tensorflow_yolov4_tflite/data/videos', 'path of folder containing input videos')
flags.DEFINE_string('classifier', './models/multilabel_model_partfeatures_withlanguage_glove.sav', 'model to classify relationships')
flags.DEFINE_string('scaler', './models/scalar_partfeatures_withlanguage_glove.sav', 'model to normalise features')
flags.DEFINE_string('yoloweights', './tensorflow_yolov4_tflite/checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_bool('optimisation', False, 'removing adjacent frames which have the same relationships')


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
                  'outside_of' : prediction[17],
                  'under' : prediction[18]
                }
    return pred_dict

#delete frame from result folder
def delete_frame(dict, i):
    #first we delete the image
    if os.path.exists("./result/frames/image_"+str(i)+".jpg"): os.remove("./result/frames/image_"+str(i)+".jpg")
    if os.path.exists("./result/frameswithboxes/image_"+str(i)+".jpg"): os.remove("./result/frameswithboxes/image_"+str(i)+".jpg")

    #next we remove it from the dictionary
    del dict["image_"+str(i)+".jpg"]

        
#return string of predictions with confidence higher than 0.5
def get_best_predictions(prediction):
    prepositions = []
    for p in prediction:
        if p >= 0.5:
            prepositions.append(eng_preposition_list[prediction.index(p)])
    return prepositions


#return value of average confidence of all relations of a frame
def get_average_confidence(relations, confidences):
    confidence_total = 0
    num_prepositions = 0
    for i in range(len(relations)):
        for j in range(len(relations[i][2])):
            confidence_total += confidences[i][eng_preposition_list.index(relations[i][2][j])]
            num_prepositions += 1
    
    if num_prepositions == 0:
        return 0
    else:
        return confidence_total/num_prepositions


#return the index of the best frame in the set (highest confidence)
def get_best_frame(relations_set, confidences_set):
    confidence_averages = []
    for i in range(len(relations_set)):
        confidence_averages.append(get_average_confidence(relations_set[i], confidences_set[i]))
    return confidence_averages.index(max(confidence_averages))


def get_frame_info(dict, frame_path):
    #set of relation 3-tuples
    relations = []

    #set of confidences of each relation in the fram
    preposition_confidences = []

    for relation in dict[frame_path]['relationships']:
        relations.append((relation['object1_label'], relation['object2_label'], relation['best_prepositions']))
        preposition_confidences.append(list(relation['prep_prediction'].values()))

    return relations, preposition_confidences


def optimise_frames(dict):
    print("Removing redundant frames")

    #get number of images in the results folder
    num_images = len([img for img in os.listdir('./result/frames') if os.path.isfile(os.path.join("./result/frames", img))])
    
    #initialise deleted counter
    deleted_counter = 0

    #relation set. We want to add adjacent equivalent frames to the set and then keep the best from each set
    relations_set = []
    confidences_set = []

    #get variables for first frame of the set
    relations_first, confidences_first = get_frame_info(dict, "image_0.jpg")

    #add them to the set 
    relations_set.append(relations_first)
    confidences_set.append(confidences_first)

    #cycle through the rest of the frames, add them to sets and optimise each set
    i = 1
    while(i < num_images):
        #get next frame
        relations_i, confidences_i = get_frame_info(dict, "image_"+ str(i) +".jpg")
        
        #check if the relations match by comparing sets (order will be ignored)
        #if there are full matches then the difference between lists will be empty
        relations_difference = [i for i in relations_first + relations_i if i not in relations_first or i not in relations_i]
        if not relations_difference:
            relations_set.append(relations_i)
            confidences_set.append(confidences_i)
        #otherwise we optimise the set (move frames which are not the best from the set to the to_delete list) and start a new set
        else:
            #get the best frame in the set and delete the rest
            if(len(relations_set) > 1):
                best_frame = i-len(relations_set)+get_best_frame(relations_set, confidences_set)
                for j in range(i-len(relations_set), i):
                    if(j != best_frame): 
                        delete_frame(dict, j)
                        deleted_counter += 1

            #reset set lists and start a new set
            relations_first = copy.deepcopy(relations_i)
            confidences_first = copy.deepcopy(confidences_i)
            relations_set = [relations_first]
            confidences_set = [confidences_first]

        i += 1

    print("Deleted ", deleted_counter, " frames")

    return dict


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    video_folder = flags.FLAGS.video
    classifier_path = flags.FLAGS.classifier
    scaler_path = flags.FLAGS.scaler
    weights_path = flags.FLAGS.yoloweights
    optimisation_flag = flags.FLAGS.optimisation

    #load models
    scaler = pickle.load(open(scaler_path, 'rb'))
    classifier = pickle.load(open(classifier_path, 'rb'))
    saved_model_loaded = tf.saved_model.load(weights_path, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    #load glove word embeddings
    file = open("./tensorflow_yolov4_tflite/data/dataset/20-vectors-glove.txt", mode="r", encoding="utf-8")
    lines = file.readlines()
    glove_dict = {}

    for line in lines:
        split_line = line.strip("\n").split(' ')
        vec = [float(i) for i in split_line[1:]]
        glove_dict[split_line[0]] = vec

    #initialise dictionary to store information
    dict = {}

    #iterate through videos in the folder
    print ("Found ", len(os.listdir(video_folder)), " videos")
    c = 0
    for video_path in os.listdir(video_folder):

        #load video
        video = cv2.VideoCapture(video_folder+"/"+video_path)

        #get video info
        fps = round(video.get(cv2.CAP_PROP_FPS))
        frames = round(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print("Processing video: ", video_path)
        print("fps", fps)
        print("frames", frames)
        frame_rate = 10 #CHANGE ACCORDING TO HOW MANY FRAMES PER SECOND WE WANT TO READ
        frame_divisor = round(fps/frame_rate)

        #loop through each frame
        frame_num = 0
        accepted_frames = 0

        while(True):

            ret, frame = video.read()
            if(ret!=1):
                print("Reached end of video")
                break
        
            #we take frame_rate frames per second
            if(frame_num%frame_divisor==0):

                #set the filenames
                path = "./result/"
                filename = "image_" + str(c) +".jpg"

                #get bounding boxes using YOLO
                pred_bbox, boxes, scores, classes, valid_detections, imHeight, imWidth = detect_frame(frame, infer)

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

                #if there is less than 2 detected objects then we cannot detect any relationships from it so we skip it
                if(valid_detections < 2):
                    continue

                #we check for boxes with intersect more than 95% and in these instances remove the prediction with lowest confidence (removes double predictions)
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

                #if there is less than 2 detected objects then we cannot detect any relationships from it so we skip it
                if(valid_detections < 2):
                    continue

                #we assign the remaining values to the pred_bbox list
                pred_bbox = [[new_boxes], [new_scores], [new_classes], [valid_detections]]
                new_boxes_copy = copy.deepcopy(new_boxes)#we kep a copy to save (since drawing the bboxes changes the values)
                new_classes_copy = copy.deepcopy(new_classes)#we kep a copy to save (since annotating the bounding boxes with classes changes the values on some occasions)

                #draw bounding boxes
                cv2.imwrite(path+"frames/"+filename, frame)
                result = utils.draw_bbox(frame, pred_bbox)
                cv2.imwrite(path+"frameswithboxes/"+filename, result)

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
                        feats_filtered = [feats["AreaObj1_Norm_wt_Union"],
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
                                           ]

                        #add glove language features
                        full_feats = np.array([feats_filtered + glove_dict[objects[i].label]+glove_dict[objects[j].label]])

                        #scale/normalise features
                        fullfeats_scaled = scaler.transform(full_feats)

                        #Make predictions on the image
                        prediction = classifier.predict_proba(fullfeats_scaled)
                        print(objects_tosave[i].label, i, prediction, objects_tosave[j].label, j)

                        #if there are no prepositions with confidence over 0.5 then we do not write it to the file
                        if(max(prediction[0]) < 0.5):
                            continue

                        # write pair to the dictionary
                        pair = { 'object1_label': objects_tosave[i].label + str(i),
                                    'object1_confidence': scores[i],
                                    'object1_bbox' : [objects_tosave[i].xmin, objects_tosave[i].ymin, objects_tosave[i].xmax, objects_tosave[i].ymax],
                                    'object2_label': objects_tosave[j].label + str(j),
                                    'object2_confidence': scores[j],
                                    'object2_bbox' : [objects_tosave[j].xmin, objects_tosave[j].ymin, objects_tosave[j].xmax, objects_tosave[j].ymax],
                                    'best_prepositions': get_best_predictions(prediction[0].tolist()),
                                    'prep_prediction': pred_dictionary(prediction[0].tolist()) 
                                }
                        relationships.append(pair)

                file = { 'height' : newheight,
                         'width' : newwidth,
                         'num_relationships' : (len(objects)**2)-len(objects),
                         'relationships' : relationships
                       }
                dict[filename] = file
                c += 1
                accepted_frames += 1
            frame_num += 1

        print("Processed video: ", video_path, " and found ", accepted_frames, " frames")

    print("Finished processing all videos and found ", c, " frames")

    #we optimise the frames if the flag is set to True
    if(optimisation_flag): 
        print("Optimising frames")
        dict = optimise_frames(dict)
    
    #save dictionary to file
    file = open('./result/info.json', 'w+')
    json.dump(dict, file, indent = 2)
    file.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass