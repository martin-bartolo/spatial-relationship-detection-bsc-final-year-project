from csv import reader as csvreader
from csv import register_dialect as registerdialect

class Object:
    def __init__(self, label='',xmin=0, ymin=0, xmax=0, ymax=0, prep=[]):
        self.label=label        
        self.xmin=xmin
        self.ymin=ymin
        self.xmax=xmax
        self.ymax=ymax
#
#
def printObjectAttr(obj,message):
    # to output to console object attributes
    print("%10s : label=%s, xmin=%d, xmax=%d, ymin=%d, ymax=%d" %\
        (message, obj.label, obj.xmin, obj.xmax, obj.ymin, obj.ymax))
    return 0
#
#
def getCsvData(fileName):
    # Read data from csv file and return a 2D python list
    # data = [ [1,2,3...], [4,5,6,...], ...]
    data=[]
    with open(fileName, 'rb') as f:
        reader = csvreader(f)
        for row in reader:
            data.append(row)
    return data
#
#
def getCsvData2(fileName,c):
    registerdialect('myDialect',
                     delimiter = c,
                     skipinitialspace=True)
    # Read data from csv file and return a 2D python list
    # data = [ [1,2,3...], [4,5,6,...], ...]
    # c : separator
    data=[]
    with open(fileName, 'rb') as f:
        reader = csvreader(f,dialect='myDialect')
        for row in reader:
            data.append(row)
    return data
#
#
def get_csv_string(data_list):
    data_string=''
    for item in data_list:
        data_string+=item+','
    return data_string[:-1]
#
#
def cleanObjLabel_deprecated(objLabel):
    # Remove any object index at the end 
    # e.g.  'person_1'  returns 'person'
    #       'person_a' returns 'person_a'
    #       'person_1_2'  returns 'person_1'
    x = objLabel.split('_')
    trimLabel=''
    if x[-1].isdigit():
        del(x[-1])
        for i in range(0,len(x)-1):        
            trimLabel +=x[i]
            trimLabel +='_'
        trimLabel +=x[-1]
    else:
        trimLabel=objLabel
    return trimLabel
#
#
def cleanObjLabel(objLabel):
    # Remove any object index at the end 
    # e.g.  'person_1'  returns 'person'
    #       'person_a' returns 'person_a'
    #       'person_1_2'  returns 'person_1'
    x = objLabel.split('_')
    trimLabel=''
    idx = None
    if x[-1].isdigit():
        idx = x[-1]
        del(x[-1])
        for i in range(0,len(x)-1):        
            trimLabel +=x[i]
            trimLabel +='_'
        trimLabel +=x[-1]
    else:
        trimLabel=objLabel
    return trimLabel, idx
