import pandas as pd 
from custom_train import CustomTrainer
from collections import OrderedDict
input_data_file_path = '/home/kaustubh/Desktop/nlg2/nlg2/rubics_deploy/stockdata.csv'
label_data_file_path = '/home/kaustubh/Desktop/nlg2/nlg2/rubics_deploy/labels.csv'
df = pd.read_csv(input_data_file_path)
df = df.iloc[:,:-1]
input_data = df.to_dict('record',into=OrderedDict)
df = pd.read_csv(label_data_file_path)
labels = df['Text'].to_list()
metadata = {
    'Date':{
        'data_type':'string' ,
        'variable_type':'categorical'

    },
    'Open':{
        'data_type':'float' ,
        'variable_type':'numeric'

    },
    'High':{
        'data_type':'float' ,
        'variable_type':'numeric'

    },
    'Low':{
        'data_type':'float' ,
        'variable_type':'numeric'

    },
    'Close':{
        'data_type':'float' ,
        'variable_type':'numeric'

    },
    'Change_close':{
        'data_type':'float' ,
        'variable_type':'numeric'

    }
}
obj = CustomTrainer(1,input_data,labels,metadata)
obj.start_training()
obj.save('./mymodel.h5')