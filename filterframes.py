import sys
import json
from absl import app, flags, logging
import numpy as np
from sVOC2k_lib_lang import englishObjectList
from sVOC2k_lib_lang import eng_preposition_list
import string
import copy

flags.DEFINE_string('input', './result', 'path to frame folder')
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
    file = open(input_path+'/info.json')
    dict = json.load(file)
    file.close()

    for key, value in dict.items():
        updated_value = copy.deepcopy(value)
        for relation in value['relationships']:

            #we remove relation if the object filter does not match
            if(object_filter and (relation['object1_label'].rstrip(string.digits) != object_filter)):
                updated_value['relationships'].remove(relation)
                updated_value['num_relationships'] -= 1
                continue
            #we remove relation if the subject filter does not match
            if(subject_filter and (relation['object2_label'].rstrip(string.digits) != subject_filter)):
                updated_value['relationships'].remove(relation)
                updated_value['num_relationships'] -= 1
                continue

            #filtering by prepositions
            if(preposition_filter):

                #we only keep synonymous prepositions
                if(synonymous_flag):
                    #there cannot be synonymous prepositions if there are not at least 2 prepositions
                    if(len(relation['best_prepositions']) < 2):
                        updated_value['relationships'].remove(relation)
                        updated_value['num_relationships'] -= 1
                        continue
                    else:
                        #we check if any pairs of synonymous prepositions are present and remove the relation if not
                        found_synonymous = False
                        for i in range(len(relation['best_prepositions'])):
                            for j in range(len(relation['best_prepositions'])):
                                if((i != j) and ([relation['best_prepositions'][i], relation['best_prepositions'][j]] in synonymous_list)):
                                    found_synonymous = True
                        if not found_synonymous:
                            updated_value['relationships'].remove(relation)
                            updated_value['num_relationships'] -= 1
                            continue

                #we only keep overlapping prepositions
                elif(overlapping_flag):
                    #there cannot be overlapping prepositions if there are not at least 2 prepositions
                    if(len(relation['best_prepositions']) < 2):
                        updated_value['relationships'].remove(relation)
                        updated_value['num_relationships'] -= 1
                        continue
                    else:
                        #we check that all pairs of prepositions are not synonymous
                        found_synonymous = False
                        for i in range(len(relation['best_prepositions'])):
                            for j in range(len(relation['best_prepositions'])):
                                if((i != j) and ([relation['best_prepositions'][i], relation['best_prepositions'][j]] in synonymous_list)):
                                    found_synonymous = True
                                    break
                            if found_synonymous:
                                break
                        if found_synonymous:
                            updated_value['relationships'].remove(relation)
                            updated_value['num_relationships'] -= 1
                            continue
                
                else:#we remove relation if any of the prepositions mentioned in the filter does not have a confidence of at least 0.5
                    low_confidence_flag = False
                    for preposition in preposition_filter_list:
                        if(list(relation['prep_prediction'].values())[eng_preposition_list.index(preposition)] < 0.5):
                            low_confidence_flag = True
                    if low_confidence_flag:
                        updated_value['relationships'].remove(relation)
                        updated_value['num_relationships'] -= 1
                        continue
        
        dict[key] = updated_value

    #after filtering we go through the dictionary and remove all entries with no relationships
    new_dict = {}
    for key, value in dict.items():
        if value['relationships']:
            new_dict[key]=value

    #count and print total filtered relationships
    num_rels = 0
    for key, value in dict.items():
        num_rels += len(value['relationships'])
    
    print("Found ", num_rels, " relationships")

    #save dictionary to file
    file = open(input_path+'/filtered.json', 'w+')
    json.dump(new_dict, file, indent = 2)
    file.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

