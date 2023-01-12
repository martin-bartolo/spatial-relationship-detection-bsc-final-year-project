import json
from sVOC2k_lib_lang import eng_preposition_list
from sVOC2k_lib_lang import englishObjectList
from sVOC2k_lib_feat import compute_geometrical_features
from sVOC2k_lib_util import Object

#load dictionary
with open("./tensorflow_yolov4_tflite/data/dataset/vrd/vrd_annotations_test.json", mode="r", encoding="utf-8") as f:
    dict = json.load(f)

#list of vrd objects
vrd_objects = ["person", "sky", "building", "truck", "bus", "table", "shirt", "chair", "car", "train", "glasses", 
               "tree", "boat", "hat", "trees", "grass", "pants", "road", "motorcycle", "jacket", "monitor", "wheel", 
               "umbrella", "plate", "bike", "clock", "bag", "shoe", "laptop", "desk", "cabinet", "counter", "bench", 
               "shoes", "tower", "bottle", "helmet", "stove", "lamp", "coat", "bed", "dog", "mountain", "horse", "plane", 
               "roof", "skateboard", "traffic light", "bush", "phone", "airplane", "sofa", "cup", "sink", "shelf", 
               "box", "van", "hand", "shorts", "post", "jeans", "cat", "sunglasses", "bowl", "computer", "pillow", 
               "pizza", "basket", "elephant", "kite", "sand", "keyboard", "plant", "can", "vase", "refrigerator", "cart", 
               "skis", "pot", "surfboard", "paper", "mouse", "trash can", "cone", "camera", "ball", "bear", "giraffe", 
               "tie", "luggage", "faucet", "hydrant", "snowboard", "oven", "engine", "watch", "face", "street", "ramp", "suitcase"]

#list of vrd prepositions
vrd_preps = ["on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", "park next", "walk next to", 
             "above", "behind", "stand behind", "sit behind", "park behind", "in the front of", "under", "stand under", 
             "sit under", "near", "walk to", "walk", "walk past", "in", "below", "beside", "walk beside", "over", "hold", 
             "by", "beneath", "with", "on the top of", "on the left of", "on the right of", "sit on", "ride", "carry", 
             "look", "stand on", "use", "at", "attach to", "cover", "touch", "watch", "along", "inside", "adjacent to", 
             "across", "contain", "drive", "drive on", "taller than", "eat", "park on", "lying on", "pull", "talk", "lean on", 
             "fly", "face", "play with", "sleep on", "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"]

dict_filtered = {}
for key, value in dict.items():
    preds = []
    for pred in value:
        if(vrd_preps[pred["predicate"]] in ["above", "on", "next to", "under", "in the front of", "behind", "near", "in", "below", "along", "across", "outside of"]):
            preds.append(pred)
    if preds:
        dict_filtered[key] = preds

#load glove
#parse glove text file into a dictionary
file = open("./tensorflow_yolov4_tflite/data/dataset/glove.6B.50d.txt", mode="r", encoding="utf-8")
lines = file.readlines()
word2vec_dict = {}

for line in lines:
    split_line = line.strip("\n").split(' ')
    vec = [float(i) for i in split_line[1:]]
    if(split_line[0] in vrd_objects):
        word2vec_dict[split_line[0]] = vec

relations_dict = {}
for key,value in dict_filtered.items():
    for rel in value:
        relation_dict = {}

        #skip relationships for objects which we have no vector for
        if(vrd_objects[rel["object"]["category"]] == "traffic light"):
            continue
        if(vrd_objects[rel["subject"]["category"]] == "traffic light"):
            continue
        if(vrd_objects[rel["object"]["category"]] == "trash can"):
            continue
        if(vrd_objects[rel["subject"]["category"]] == "trash can"):
            continue

        object1vec = word2vec_dict[vrd_objects[rel["object"]["category"]]]
        object2vec = word2vec_dict[vrd_objects[rel["subject"]["category"]]]
        object1 = Object(vrd_objects[rel["object"]["category"]], rel["object"]["bbox"][2], rel["object"]["bbox"][0], rel["object"]["bbox"][3], rel["object"]["bbox"][1], [])
        object2 = Object(vrd_objects[rel["subject"]["category"]], rel["subject"]["bbox"][2], rel["subject"]["bbox"][0], rel["subject"]["bbox"][3], rel["subject"]["bbox"][1], [])

        #change name of predicate when needed
        if(vrd_preps[rel["predicate"]] == "in the front of"):
            predicate = "in_front_of"
        elif(vrd_preps[rel["predicate"]] == "next to"):
            predicate = "next_to"
        elif(vrd_preps[rel["predicate"]] == "outside of"):
            predicate = "outside_of"
        else:
            predicate = vrd_preps[rel["predicate"]]

        feats, extrafeats = compute_geometrical_features(1920, 1080, object1, object2)

        relation_dict["object1"] = object1vec
        relation_dict["object2"] = object2vec
        relation_dict["predicate"] = predicate
        relation_dict["feats"] = feats
    relations_dict[len(relations_dict)] = relation_dict

#save dictionary to file
file = open('./features_vrd.json', 'w+')
json.dump(relations_dict, file, indent = 2)
file.close()

