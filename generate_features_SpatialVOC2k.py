import json
from sVOC2k_lib_util import Object
from sVOC2k_lib_util import printObjectAttr
from sVOC2k_lib_util import cleanObjLabel_deprecated
from sVOC2k_lib_feat import compute_geometrical_features
from sVOC2k_lib_lang import getEnglishObject
from sVOC2k_lib_lang import fren2eng_preposition

#change as needed
with open("SpatialVOC2k.json", mode="r", encoding="utf-8") as f:
    dict = json.load(f)

#function to create an object instance when given the name from a relation
def getObjectFromName(image, name):
    for o in image['objects'].values():
        if(o['name'] == name):
            return Object(getEnglishObject(cleanObjLabel_deprecated(o['name'])), o['xmin'], o['ymin'], o['xmax'], o['ymax'], [])
    return None

# generate string from list of relations
def string_from_list(list):
    s = '['
    for rel in list:
        s += '\"' + rel + '\", '
    s = s[:-2]+']'
    return s
    

#generate relations list with image dimensions, objects and relation
relations = []
for im in dict.values():
    imSizeX = im['width']
    imSizeY = im['height']
    for r in im['relations']:
        obj1 = getObjectFromName(im, r['object_0'])
        obj2 = getObjectFromName(im, r['object_1'])
        rel_list = r['all']
        rel_list_eng = []
        for rel in rel_list:
            rel_list_eng.append(fren2eng_preposition[rel])
        relations.append([imSizeX, imSizeY, obj1, obj2, rel_list_eng])

#write json file from relations list
file = open("features.json","a")
file.write('{\n')
i = 1
for r in relations:
    feats, extra_feats = compute_geometrical_features(r[0], r[1], r[2], r[3])
    s = '\t\"' + str(i) + '\": {\n'
    file.write(s)
    s = '\t\t\"object_0\": \"' + r[2].label + '\",\n'
    file.write(s)
    s = '\t\t\"object_1\": \"' + r[3].label + '\",\n'
    file.write(s)
    s = '\t\t\"relations": ' + string_from_list(r[4]) + ',\n'
    file.write(s)
    feats_string = str(feats).replace('\'', '\"')
    s = '\t\t\"features": ' + feats_string + '\n'
    file.write(s)
    if(r == relations[-1]):
        file.write("\t}\n")
    else:
        file.write("\t},\n")
    i += 1
file.write('}')
file.close()