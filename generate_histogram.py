import json
import sys
import random
from absl import app, flags, logging
import numpy as np
import matplotlib.pyplot as plt
from sVOC2k_lib_lang import eng_preposition_list

flags.DEFINE_string('input', './result/filtered.json', 'path to input json file')
flags.DEFINE_string('preposition', '', 'preposition to display confidences for')


def main(_argv):

    input_path = flags.FLAGS.input
    preposition = flags.FLAGS.preposition

    #make sure preposition is valid
    if preposition not in eng_preposition_list:
        print("Invalid preposition")
        sys.exit()

    #load dictionary
    file = open(input_path)
    dict = json.load(file)
    file.close()

    #initialise lists for each confidence bin
    confidences = []
    sorted_confidences = {}
    sorted_confidences_05_06 = []
    sorted_confidences_06_07 = []
    sorted_confidences_07_08 = []
    sorted_confidences_08_09 = []
    sorted_confidences_09_1 = []

    
    for key, value in dict.items():
        for relation in value['relationships']:
            #add confidence values to list to create histogram later
            confidences.append(relation['prep_prediction'][preposition])

            #add relation to correct dictionary
            if(relation['prep_prediction'][preposition] < 0.6):
                rel = {}
                rel['filename'] = key
                rel.update(value)
                rel['relationships'] = [relation]
                sorted_confidences_05_06.append(rel)
                continue
            if(relation['prep_prediction'][preposition] < 0.7):
                rel = {}
                rel['filename'] = key
                rel.update(value)
                rel['relationships'] = [relation]
                sorted_confidences_06_07.append(rel)
                continue
            if(relation['prep_prediction'][preposition] < 0.8):
                rel = {}
                rel['filename'] = key
                rel.update(value)
                rel['relationships'] = [relation]
                sorted_confidences_07_08.append(rel)
                continue
            if(relation['prep_prediction'][preposition] < 0.9):
                rel = {}
                rel['filename'] = key
                rel.update(value)
                rel['relationships'] = [relation]
                sorted_confidences_08_09.append(rel)
                continue
            if(relation['prep_prediction'][preposition] < 1.):
                rel = {}
                rel['filename'] = key
                rel.update(value)
                rel['relationships'] = [relation]
                sorted_confidences_09_1.append(rel)
                continue

    sorted_confidences['confidence_0.5-0.6'] = sorted_confidences_05_06
    sorted_confidences['confidence_0.6-0.7'] = sorted_confidences_06_07
    sorted_confidences['confidence_0.7-0.8'] = sorted_confidences_07_08
    sorted_confidences['confidence_0.8-0.9'] = sorted_confidences_08_09
    sorted_confidences['confidence_0.9-1'] = sorted_confidences_09_1

    #take 2 random relations from each confidence bin
    random_relations = {}
    for key, value in sorted_confidences.items():
        try:
            random_values = random.sample(value, 3)
        except ValueError:
            try:
                random_values = random.sample(value, 2)
            except ValueError:
                try:
                    random_values = random.sample(value, 1)
                except ValueError:
                    print("None found")

        random_relations[key] = random_values
    
    #save dictionary to file
    file = open('./result/random_relations.json', 'w+')
    json.dump(random_relations, file, indent = 2)
    file.close()

    hist = plt.hist(confidences, bins= [.5, .6, .7, .8, .9, 1])
    plt.show()

    
    

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

