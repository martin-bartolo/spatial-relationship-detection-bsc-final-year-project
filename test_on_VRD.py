import json
import numpy as np
import pickle
import statistics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sVOC2k_lib_lang import eng_preposition_list
from sVOC2k_lib_lang import englishObjectList

#function to get scores of prediction
def get_accuracy(preds, tests):

    #get index of correct class in test set
    test_indices = [test.index(1) for test in tests.tolist()]

    total = len(preds)
    correct = 0
    for i in range(len(preds)):
        #if the correct preposition is one of our predicted prepositions add one
        if preds[i][test_indices[i]] == 1:
            correct += 1
    return correct/total

#load dictionary
with open("./features_vrd_filteredobjects.json", mode="r", encoding="utf-8") as f:
    dict = json.load(f)

#generate features and relations numpy arrays
feats = []
rels = []
object1s = []
object2s = []

for d in dict.values():
    if not d:
        continue
    rels.append([d['predicate']])
    feats.append([d['feats']["AreaObj1_Norm_wt_Union"],
             d['feats']["AreaObj2_Norm_wt_Union"],
             d['feats']["objAreaRatioTrajLand"],
             d['feats']["DistBtCentr_Norm_wt_UnionBB"],
             d['feats']["AreaOverlap_Norm_wt_Total"],
             d['feats']["AreaOverlap_Norm_wt_Min"],
             d['feats']["InvFeatXminXmin"],
             d['feats']["InvFeatXmaxXmin"],
             d['feats']["InvFeatYminYmin"],
             d['feats']["InvFeatYmaxYmin"],
             d['feats']["AspectRatioObj_1"],
             d['feats']["AspectRatioObj_2"],
             d['feats']["EuclDist_Norm_wt_UnionBB"],
             d['feats']["unitVecTrajLand_Norm_wt_UnionBB_x"],
             d['feats']["unitVecTrajLand_Norm_wt_UnionBB_y"],
             ])
    object1s.append(d['object1'])
    object2s.append(d['object2'])
feats = np.array(feats)
rels = np.array(rels)

#binarize relationships
rels_bin = []
for rel in rels:
    rels_row = []
    for i in range(19):
        if(i == eng_preposition_list.index(rel)):
           rels_row.append(1)
        else:
            rels_row.append(0)
    rels_bin.append(rels_row)

rels_bin = np.array(rels_bin)

#compose geometrical and language features
fullfeats = []
fullfeats = []
num_rows, num_cols = feats.shape
for i in range(num_rows):
    row = np.append(feats[i], np.append(object1s[i], object2s[i]))
    fullfeats.append(row)
fullfeats = np.array(fullfeats)

print(fullfeats)

#normalise geometrical features
scaler = pickle.load(open("./models/scalar_partfeatures_withlanguage_glove.sav", 'rb'))
fullfeats_scaled = scaler.transform(fullfeats)

#load model
classifier = pickle.load(open("./models/multilabel_model_partfeatures_withlanguage_glove.sav", 'rb'))

#Make predictions on entire test data
predictions = classifier.predict(fullfeats_scaled)

#Use score method to get accuracy of model
accuracy = get_accuracy(predictions, rels_bin)

#print overall scores
print("Overall Accuracy: ", accuracy)