import sys
import json
from absl import app, flags, logging
import numpy as np
from sVOC2k_lib_lang import englishObjectList
from sVOC2k_lib_lang import eng_preposition_list
import string
import copy

flags.DEFINE_string('input', './features_vrd_filteredobjects.json', 'path to frame folder')
flags.DEFINE_string('prepositions', '', 'prepositions (separated by commas and no spaces) to filter')
flags.DEFINE_string('object', '', 'filter relationships including this object')
flags.DEFINE_string('subject', '', 'filter relationships including this subject')


def main(_argv):

    input_path = flags.FLAGS.input

    #validate object filters
    object_filter = flags.FLAGS.object
    if(object_filter and object_filter not in englishObjectList):
        print("Invalid object filter")
        sys.exit()

    subject_filter = flags.FLAGS.subject
    if(subject_filter and subject_filter not in englishObjectList):
        print("Invalid subject filter")
        sys.exit()

    #validate preposition filters and convert to list
    preposition_filter = flags.FLAGS.prepositions
    overlapping_flag = False
    synonymous_flag = False

    if(preposition_filter == "overlapping"): overlapping_flag = True #filter all overlapping prepositions
    elif(preposition_filter == "synonymous"): synonymous_flag = True #filter all synonymous prepositions
    else:
        preposition_filter_list = []
        if(preposition_filter):
            preposition_filter_list = list(preposition_filter.split(","))
            for preposition in preposition_filter_list:
                if(preposition not in eng_preposition_list):
                    print("Invalid preposition filter")
                    sys.exit()
    
    #list of synonymous prepositions
    synonymous_list = [['above', 'on'], ['opposite', 'in_front_of'], ['below', 'under'], ['next_to', 'near']]

    #load dictionary
    file = open(input_path)
    dict = json.load(file)
    file.close()

    counts = np.zeros(19)
    for key, value in dict.items():
        if value['predicate'] == "above":
            print("HELLO")
        counts[eng_preposition_list.index(value['predicate'])] += 1

    print(counts)

    file.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

