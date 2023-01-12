import json
import sys
import numpy as np
import statistics
import pickle
import math
from sVOC2k_lib_lang import eng_preposition_list
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

np.set_printoptions(threshold=sys.maxsize)

def get_scores(preds, tests):

    #make lists of indices where preds and tests have values of 1 or 0
    pred_pos_indices = []
    pred_neg_indices = []
    test_pos_indices = []
    test_neg_indices = []

    #make lists of indices where preds have values of 1 or 0
    for pred in preds:
        pos_indices = []
        neg_indices = []
        for i in range(len(pred)):
            if pred[i] == 1:
               pos_indices.append(i)
            elif pred[i] == 0:
               neg_indices.append(i)
        pred_pos_indices.append(pos_indices)
        pred_neg_indices.append(neg_indices)

    #make lists of indices where test have values of 1 or 0
    for test in tests:
        pos_indices = []
        neg_indices = []
        for i in range(len(test)):
            if test[i] == 1:
               pos_indices.append(i)
            elif test[i] == 0:
                neg_indices.append(i)
        test_pos_indices.append(pos_indices)
        test_neg_indices.append(neg_indices)

    #IOU accuracy
    intersections = []
    unions = []
    
    #make lists of unions and intersections
    for i in range(len(pred_pos_indices)):
        unions.append(list(set(pred_pos_indices[i]).union(set(test_pos_indices[i]))))
        intersections.append(list(set(pred_pos_indices[i]) & set(test_pos_indices[i])))

    #make list of intersection/union values
    probs = []
    for i in range(len(intersections)):
        probs.append(len(intersections[i])/len(unions[i]))

    accuracy = statistics.mean(probs)

    #precision
    prec = []
    for i in range(len(intersections)):
        if(len(pred_pos_indices[i]) != 0):
            prec.append(len(intersections[i])/len(pred_pos_indices[i]))
    precision = statistics.mean(prec)

    #recall
    rec = []
    for i in range(len(intersections)):
        if(len(test_pos_indices[i]) != 0):
            rec.append(len(intersections[i])/len(test_pos_indices[i]))
    recall = statistics.mean(rec)
    
    #f-score
    fscore = (2*precision*recall)/(precision+recall)

    #label-based scores
    true_positives = np.zeros(19)
    false_positives = np.zeros(19)
    true_negatives = np.zeros(19)
    false_negatives = np.zeros(19)

    for i in range(len(pred_pos_indices)):
        for pred in pred_pos_indices[i]:
            #update true positives
            if pred in test_pos_indices[i]:
                true_positives[pred] += 1
            #update false positives
            else:
                false_positives[pred] += 1
        #update false negatives
        for test in test_pos_indices[i]:
            if test not in pred_pos_indices[i]:
                false_negatives[test] += 1
        
        #update true negatives
        for test in test_neg_indices[i]:
            if test in pred_neg_indices[i]:
                true_negatives[test] += 1

    label_accuracies = []
    label_precisions = []
    label_recalls = []
    label_f1s = []

    #calculate accuracy, precision, recall and f1 for each preposition
    for i in range(19):
        
        label_precisions.append(true_positives[i]/(true_positives[i]+false_positives[i]))
        label_recalls.append(true_positives[i]/(true_positives[i]+false_negatives[i]))
        label_f1s.append((2*true_positives[i])/((2*true_positives[i])+false_negatives[i]+false_positives[i]))

    return(accuracy, precision, recall, fscore, label_precisions, label_recalls, label_f1s)

#feature input file (change as needed)
with open("features.json", mode="r", encoding="utf-8") as f:
    dict = json.load(f)

#generate features and relations numpy arrays
feats = []
rels = []
object1s = []
object2s = []

for d in dict.values():
    rels.append(d['relations'])
    feats.append([d['features']["AreaObj1_Norm_wt_Union"],
             d['features']["AreaObj2_Norm_wt_Union"],
             d['features']["objAreaRatioTrajLand"],
             d['features']["DistBtCentr_Norm_wt_UnionBB"],
             d['features']["AreaOverlap_Norm_wt_Total"],
             d['features']["AreaOverlap_Norm_wt_Min"],
             d['features']["InvFeatXminXmin"],
             d['features']["InvFeatXmaxXmin"],
             d['features']["InvFeatYminYmin"],
             d['features']["InvFeatYmaxYmin"],
             d['features']["AspectRatioObj_1"],
             d['features']["AspectRatioObj_2"],
             d['features']["EuclDist_norm_wt_ImageBB"],
             d['features']["unitVecTrajLand_Norm_wt_UnionBB_x"],
             d['features']["unitVecTrajLand_Norm_wt_UnionBB_y"],
             ])
    object1s.append(d['object_0'])
    object2s.append(d['object_1'])
feats = np.array(feats)
rels = np.array(rels)

#binarize relationships
mlb = MultiLabelBinarizer()
rels = mlb.fit_transform(rels)

#parse word2vec text file into a dictionary
file = open("./tensorflow_yolov4_tflite/data/dataset/20-vectors-glove.txt", mode="r", encoding="utf-8")
lines = file.readlines()
word2vec_dict = {}

for line in lines:
    split_line = line.strip("\n").split(' ')
    vec = [float(i) for i in split_line[1:]]
    word2vec_dict[split_line[0]] = vec

# compose geometrical and language features
fullfeats = []
num_rows, num_cols = feats.shape
for i in range(num_rows):
    row = np.append(feats[i], np.append(np.array(word2vec_dict[object1s[i]]), np.array(word2vec_dict[object2s[i]])))
    fullfeats.append(row)
fullfeats = np.array(fullfeats)

#split into training and testing
#iterative
np.random.seed(3)
x_train, y_train, x_test, y_test = iterative_train_test_split(fullfeats, rels, test_size=0.2)
#x_train_train, y_train_train, x_val, y_val = iterative_train_test_split(x_train, y_train, test_size=0.2)

#normal
#x_train, x_test, y_train, y_test = train_test_split(fullfeats, rels, test_size=0.2, random_state=0)
#x_train_train, x_val, y_train_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

#normalise geometrical features
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#save the scalar
filename = './models/scalar_partfeatures_withlanguage_glove_farfrom.sav'
pickle.dump(scaler, open(filename, 'wb'))

#create MLPClassifier instance
classifier = MLPClassifier(activation='relu', solver='adam', random_state=5, hidden_layer_sizes=(40,40,40), beta_1=0.86, beta_2=0.998)

##find best hyper parameters
#distributions = {'activation': ['relu'], 'solver': ['adam'], 'random_state': [0,1,2,3,4,5,6,7,8,9], 
#                 'beta_1': [0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95], 'beta_2': [0.995,0.996,0.997,0.998,0.999]}
#clf = RandomizedSearchCV(classifier, distributions, random_state=0)
#search = clf.fit(x_train, y_train)
#print(search.best_params_)
#print(search.best_score_)

#train model
classifier.fit(x_train, y_train)

#save the model
filename = './models/multilabel_model_partfeatures_withlanguage_glove_farfrom.sav'
pickle.dump(classifier, open(filename, 'wb'))

#Make predictions on entire test data
predictions = classifier.predict(x_test)

#Use score method to get accuracy of model
accuracy, precision, recall, f1score, label_precisions, label_recalls, label_f1s = get_scores(predictions, y_test)

#print overall scores
print("Overall Accuracy: ", accuracy)
print("Overall Precision: ", precision)
print("Overall Recall: ", recall)
print("Overall F1 Score: ", f1score)
print("\n\nLabel-Based Metrics:\n")

counts = y_test.sum(axis=0)

#print label based scores in a table
print("------------------------------------------------------------------------------------")
print("| {:20s} | {:17s} | {:17s} | {:17s} |".format("Preposition", "Precision", "Recall", "F1"))
for i in range(19):
    print("| {:20s} | {:17.15f} | {:17.15f} | {:17.15f} |".format(eng_preposition_list[i]+"("+str(counts[i])+")", label_precisions[i], label_recalls[i], label_f1s[i]))
print("------------------------------------------------------------------------------------")