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
        self.encoder_input = []
        self.decoder_input = []
        self.decoder_true_label = []
        for record,label in zip(self.input_data,self.labels):
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
                self.encoder_input.append(temp)
                self.decoder_input.append('startseq '+label)
                self.decoder_true_label.append(label+' endseq')
            except Exception:
                    pass
        self.max_input_len = max([len(record) for record in self.decoder_input])
        self.max_output_len = max([len(record.split()) for record in self.decoder_input])
        #configs
        self.hidden_dimesion = self.max_input_len * 2
        self.embedding_dimension = 128
        self.batch_size = 128
        self.epochs = 100
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.decoder_input + self.decoder_true_label)
        self.decoder_input = self.tokenizer.texts_to_sequences(self.decoder_input)
        self.decoder_true_label = self.tokenizer.texts_to_sequences(self.decoder_true_label)
        self.encoder_input = pad_sequences(self.encoder_input,maxlen=self.max_input_len,padding='post')
        self.decoder_input = pad_sequences(self.decoder_input,maxlen=self.max_output_len,padding='post')
        self.decoder_true_label = pad_sequences(self.decoder_true_label,maxlen=self.max_output_len,padding='post')
        self.decoder_true_label = to_categorical(self.decoder_true_label)
        self.vocab_size = self.decoder_true_label.shape[2]
        self.model = self.get_model()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.is_trained = False
    
    def start_training(self):
        z = np.zeros((self.encoder_input.shape[0], self.hidden_dimesion))
        self.training_output = self.model.fit(
        [self.encoder_input, self.decoder_input, z, z], self.decoder_true_label,
        batch_size=self.batch_size,
        epochs=self.epochs,
        validation_split=0.1,
        verbose = 1
        )

    @staticmethod
    def softmax_over_time(x):
        assert(K.ndim(x) > 2)
        e = K.exp(x - K.max(x, axis=1, keepdims=True))
        s = K.sum(e, axis=1, keepdims=True)
        return e / s

    def get_model(self):
        '''
        Creates And Returns the model
        '''
        def stack_and_transpose(x):
            x = K.stack(x) 
            return K.permute_dimensions(x, pattern=(1, 0, 2))
        def one_step_attention(h, st_1):
            st_1 = attn_repeat_layer(st_1)
            x = attn_concat_layer([h, st_1])
            x = attn_dense1(x)
            alphas = attn_dense2(x)
            context = attn_dot([alphas, h])
            return context
        encoder_input_layer = Input(shape=(self.max_input_len,))
        reshape_layer = Reshape((self.max_input_len,1))
        reshaped_input = reshape_layer(encoder_input_layer)
        encoder_layer = Bidirectional(LSTM(self.hidden_dimesion,return_sequences=True))
        encoder_outputs = encoder_layer(reshaped_input)
        decoder_input_layer = Input(shape=(self.max_output_len,))
        decoder_embedding = Embedding(self.vocab_size,self.embedding_dimension)
        decoder_inputs_x = decoder_embedding(decoder_input_layer)
        attn_repeat_layer = RepeatVector(self.max_input_len)
        attn_concat_layer = Concatenate(axis=-1)
        attn_dense1 = Dense(10, activation='tanh')
        attn_dense2 = Dense(1, activation=self.softmax_over_time)
        attn_dot = Dot(axes=1) 
        decoder_lstm = LSTM(self.hidden_dimesion, return_state=True)
        decoder_dense = Dense(self.vocab_size, activation='softmax')
        initial_s = Input(shape=(self.hidden_dimesion,), name='s0')
        initial_c = Input(shape=(self.hidden_dimesion,), name='c0')
        context_last_word_concat_layer = Concatenate(axis=2)
        s = initial_s
        c = initial_c
        outputs = []
        t = 0
        while True:
            context = one_step_attention(encoder_outputs, s)
            selector = Lambda(lambda x: x[:, t:t+1])
            xt = selector(decoder_inputs_x)
            decoder_lstm_input = context_last_word_concat_layer([context, xt])
            o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])
            decoder_outputs = decoder_dense(o)
            outputs.append(decoder_outputs)
            t+=1
            if t >= self.max_output_len:
                break
        stacker = Lambda(stack_and_transpose)
        outputs = stacker(outputs)

        model = Model(
        inputs=[
            encoder_input_layer,
            decoder_input_layer,
            initial_s, 
            initial_c,
        ],
        outputs=outputs
        )
        return model
        
        

    









                    
                    
            
                        


                
                
        
    

