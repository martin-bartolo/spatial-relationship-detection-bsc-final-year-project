import json
from absl import app, flags, logging
import string
from sVOC2k_lib_util import Object
from sVOC2k_lib_util import printObjectAttr
from sVOC2k_lib_util import cleanObjLabel_deprecated
from sVOC2k_lib_feat import compute_geometrical_features
from sVOC2k_lib_lang import getEnglishObject
from sVOC2k_lib_lang import fren2eng_preposition

flags.DEFINE_string('preposition', 'above', 'preposition to generate features for (match with filter)')

def main(_argv):
    preposition = flags.FLAGS.preposition

    with open("./result/filtered.json", mode="r", encoding="utf-8") as f:
        dict = json.load(f)

    # generate string from list of relations
    def string_from_list(list):
        s = '['
        for rel in list:
            s += '\"' + rel + '\", '
        s = s[:-2]+']'
        return s
    

    #generate relations list with image dimensions, objects and relation
    relations = []
    print(len(dict.values()))
    for im in dict.values():
        imSizeX = im['width']
        imSizeY = im['height']
        for r in im['relationships']:
            obj1 = Object(r['object1_label'].rstrip(string.digits), r['object1_bbox'][0], r['object1_bbox'][1], r['object1_bbox'][2], r['object1_bbox'][3], [])
            obj2 = Object(r['object2_label'].rstrip(string.digits), r['object2_bbox'][0], r['object2_bbox'][1], r['object2_bbox'][2], r['object2_bbox'][3], [])
            rel_list = [preposition]
            relations.append([imSizeX, imSizeY, obj1, obj2, rel_list])

    #write json file from relations list
    file = open("features_unlabelledvideos.json","a")
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

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass