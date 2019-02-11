import csv
events_file_path = '/home/kaustubh/Desktop/nlg2/dataset/all.events'
labels_file_path = '/home/kaustubh/Desktop/nlg2/dataset/all.text'
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
                time_vals = att[1].split('-')
                x.append(int(time_vals[0]))
                x.append(int(time_vals[1]))
            elif 'mode' in att[0]:
                x.append(ordinal_mappings[att[1]])
            else:
                x.append(int(att[1]))
    
    return x
with open('input.csv') as f:
    writer = csv.writer(f)
