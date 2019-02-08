from collections import OrderedDict
from keras import Model
from keras.layers import LSTM,Dense,Bidirectional,Concatenate,Dot,Input,Lambda,RepeatVector,Embedding,Reshape
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.backend as K
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

class CustomTrainer:
    def __init__(
        self,
        input_data,
        labels,
        metadata
    ):
        '''
        Constructor for class CustomTrainer
        #Arguments
        input_data:
            It is list of OrderedDict where each orderedDict represents
            a record or data instance.Each OrderedDict Contains keys as 
            attribute names and its correcponding values
        labels:
            It is a list strigs representing the natural language 
            description of the records in the input_data list
        metadata:
            It is a dictionary which gives information about attribute data types
            and its variable types.Its keys are attribute name and corresponding values are dictionary 
            which will contain two keys which will be 'data_type' and 'variable_type'
        '''
        #check if no of records and labels are same
        assert len(input_data) == len(labels)
        #check if all the records contains same keys
        keys_list = [record.keys() for record in input_data]
        assert len(set(keys_list)) == 1
        assert list(set(keys_list)) == list(metadata.keys())
        self.input_data = input_data
        self.labels = labels
        self.metadata = metadata
        self.categorical_mappings = {}
        self.categorical_mappings['--'] = 0
        self.index = 0
        #make categorical mappings
        for key,val in self.metadata.items():
            if val['variable_type'] == 'categorical':
                unique_values = set([record[key] for record in self.input_data])
                self.categorical_mappings[key] = {}
                for value in unique_values:
                    self.index += 1
                    self.categorical_mappings[key][value] = self.index
        #create training data
        self.train_x = []
        self.train_y = []
        for record,label in list(zip(self.input_data,self.labels)):
            try:
                temp = []
                for key,val in record.items():
                        if key in self.categorical_mappings:
                            temp.append(self.categorical_mappings[key][val])
                        elif self.metadata[key]['data_type'] == 'date':
                            val = val.split('-')
                            if len(val) == 3:
                                temp.append(int(val[0]))
                                temp.append(int(val[1]))
                                temp.append(int(val[2]))
                            else:
                                raise Exception
                        elif self.metadata[key]['variable_type'] == 'numerical':
                            temp.append(val)
                self.train_x.append(temp)
                self.train_y.append(label)
            except Exception:
                    pass
        self.max_input_len = max([len(record) for record in self.train_x])
        self.max_output_len = max([len(record.split()) for record in self.train_y])
        









        
        
        
                    


            
            
    
    

