import pandas as pd 
from custom_train import CustomTrainer
from collctions import OrderedDict
input_data_file_path = './input.csv'
label_data_file_path = './label.csv'
df = pd.read_csv(input_data_file_path)
input_data = df.to_dict('record',into=OrderedDict)
df = pd.read_csv(label_data_file_path)
labels = df['labels'].to_list()
