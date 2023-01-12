#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 07:53:57 2018

@author: adrian muscat

modified for Python 3
Aug-2020

"""
from sVOC2k_lib_util import cleanObjLabel
from sVOC2k_lib_util import getCsvData2
import numpy as np

######################################################################
# These lists are deprecated
englishObjectList = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                     "car", "cat", "chair", "cow", "diningtable", "dog",
                     "horse",
                     "motorbike", "person", "pottedplant", "sheep", "sofa",
                     "train", "tvmonitor"]
              
frenchObjectList = ["l'avion", "l'oiseau", "le_velo", "le_bateau", "la_bouteille",
              "le_bus", "la_voiture", "le_chat", "la_chaise", "la_vache",
              "la_table", "le_chien", "le_cheval", "la_moto", "la_personne",
              "la_plante", "le_mouton", "le_canape", "le_train",
              "l'ecran"]
######################################################################

              
VOCobjectList = [
                    [ 1, "aeroplane", "l'avion"],
                    [ 2, "bird", "l'oiseau"],
                    [ 3, "bicycle", "le_velo"],
                    [ 4, "boat", "le_bateau"],
                    [ 5, "bottle", "la_bouteille"],
                    [ 6, "bus","le_bus"],
                    [ 7, "car", "la_voiture"],
                    [ 8, "cat", "le_chat"], 
                    [ 9, "chair", "la_chaise"], 
                    [10, "cow", "la_vache"], 
                    [11, "diningtable","la_table"], 
                    [12, "dog", "le_chien"],
                    [13, "horse", "le_cheval"],
                    [14, "motorbike", "la_moto"], 
                    [15, "person", "la_personne"], 
                    [16, "pottedplant","la_plante"], 
                    [17, "sheep", "le_mouton"], 
                    [18, "sofa", "le_canape"],
                    [19, "train", "le_train"], 
                    [20, "tvmonitor", "l'ecran"]
                ]

# define dictionary English to French (objects)
#fren2eng_object={}
#for i,french_object in enumerate(frenchObjectList):
#    fren2eng_object[french_object]=englishObjectList[i]
##
fren2eng_object={}
for i,french_object in enumerate(frenchObjectList):
    fren2eng_object[french_object]=englishObjectList[i]
#
eng2fren_object={}
for i,english_object in enumerate(englishObjectList):
    eng2fren_object[english_object]=frenchObjectList[i]
#
def getFrenchObject(objLabel):
    english=cleanObjLabel(objLabel)
    french = eng2fren_object[english[0]]
    if english[1] is not None:
        french += "_" + english[1]
    return french
#
#
def getEnglishObject(objLabel):
    french=cleanObjLabel(objLabel)
    english = fren2eng_object[french[0]]
    if french[1] is not None:
        english += "_" + french[1]
    return english
#
#

# read french prepositions from dataset
# Define Dictionary from French to English (prepositions)
# translations from collins dictionary
fren2eng_preposition={}
fren2eng_preposition["a_cote"]="next_to"     #(1)    
fren2eng_preposition["a_cote_de"]="next_to"    #(1773) The café’s next door to the station.     
fren2eng_preposition["a_cot_de"]="next_to"
fren2eng_preposition["a_l\'exterieur_de"]="outside_of"#(51)              
fren2eng_preposition["au_dessous_de"]="below"  #(1)also  underneath and below          
fren2eng_preposition["au_dessus_de"]="above"   #(148) above the table            
fren2eng_preposition["au_niveau_de"]="at_the_level_of" #(1155)            
fren2eng_preposition["aucun"]="none"         #(28)  
fren2eng_preposition["autour_de"]="around"  # (42) around the house              
fren2eng_preposition["contre"]="against"    # (730) Don’t put your bike against the wall.      
fren2eng_preposition["dans"]="in"            # (74) It’s in the box, also inside
fren2eng_preposition["derriere"]="behind"       #(1326)
fren2eng_preposition["devant"]="in_front_of"    # (1372) He was sitting in front of me          
fren2eng_preposition["en_face_de"]="opposite"    #(333)          
fren2eng_preposition["en_travers_de"]="across"  # (1)There was a tree lying across the road           
fren2eng_preposition["le_long_de"]="along"       #(85)       
fren2eng_preposition["loin_de"]="far_from"       #(476)   
fren2eng_preposition["loin"]="far_from"
fren2eng_preposition["par_dela"]="beyond"        #(47)        
fren2eng_preposition["pres"]="near"        # (1) I live nearby.                 
fren2eng_preposition["pres_de"]="near"        # (2856) He lives near the post office, Sit down next to me.         
fren2eng_preposition["sous"]="under"        # (533) Put it under the table in the meantime           
fren2eng_preposition["sur"]="on"            # (447) Put it on the table.             

def getVOCembeddings(filename, dialect):
    """
    filename : path to embeddings file
    language : 'english' or 'french'
    """
    obj_idx = {'glove_french':[' ',0], 'w2v_french':[',',1]}
    data = getCsvData2(filename,obj_idx[dialect][0])
    embed_dict={}
    for item in data:        
        embed_dict[item[obj_idx[dialect][1]]] = np.array(item[2:], dtype=float)
    #
    return embed_dict

eng_preposition_list = ["above", "across", "against", "along", "around", "at_the_level_of",
                        "behind", "beyond", "below", "far_from", "in", "in_front_of", "near",
                        "next_to","none","on", "opposite", "outside_of", "under"]

        




















