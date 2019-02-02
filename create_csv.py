import csv
#Paths
EVENTS_DATASET_PATH = '../dataset/all.events'
LABELS_DATASET_PATH = '../dataset/all.text'

csv_write = csv.writer(open('input_csv.csv','w'))

def preprocess_features(feature):
    '''
    creates a feature vector and returns it
    '''
    x = []
    attr = [ (f.split(':')[0],f.split(':')[1])  for f in feature.split()]
    for att in attr:
        #ignore mode bucket attribute for now
        if not 'mode-bucket' in att[0]:
            if 'time' in att[0]:
                csv_write.writerows([[att[0],att[1]]])
            elif 'mode' in att[0]:
                
                csv_write.writerows([[att[0],att[1]]])
            else:
                
                csv_write.writerows([[att[0],att[1]]])
    
    return x

feature_file = open(EVENTS_DATASET_PATH,'r')
labels_file = open(LABELS_DATASET_PATH,'r')

for feature,label in zip(feature_file,labels_file):
    try:
        preprocess_features(feature)
        break
    except Exception as e:
        print(e)
        pass