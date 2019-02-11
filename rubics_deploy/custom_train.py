from collections import OrderedDict
from keras import Model
from keras.layers import LSTM,Dense,Bidirectional,Concatenate,Dot,Input,Lambda,RepeatVector,Embedding,Reshape
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.models import load_model
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

class CustomTrainer:
    def __init__(
        self,
        project_id,
        input_data,
        labels,
        metadata
    ):
        '''
        Constructor for class CustomTrainer
        #Arguments
        project_id:
            unique ID assigned for this model project
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
        self.project_id = project_id
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
                temp = self.preprocess_input_record(record)
                self.encoder_input.append(temp)
                self.decoder_input.append('startseq '+label)
                self.decoder_true_label.append(label+' endseq')
            except Exception:
                    pass
        self.max_input_len = max([len(record) for record in self.decoder_input])
        self.max_output_len = max([len(record.split()) for record in self.decoder_input])
        #configs
        #==================================
        self.hidden_dimesion = self.max_input_len * 2
        self.embedding_dimension = 128
        self.batch_size = 128
        self.epochs = 500
        #==================================
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.decoder_input + self.decoder_true_label)
        self.decoder_input = self.tokenizer.texts_to_sequences(self.decoder_input)
        self.decoder_true_label = self.tokenizer.texts_to_sequences(self.decoder_true_label)
        self.encoder_input = pad_sequences(self.encoder_input,maxlen=self.max_input_len,padding='post')
        self.decoder_input = pad_sequences(self.decoder_input,maxlen=self.max_output_len,padding='post')
        self.decoder_true_label = pad_sequences(self.decoder_true_label,maxlen=self.max_output_len,padding='post')
        self.decoder_true_label = to_categorical(self.decoder_true_label)
        self.vocab_size = self.decoder_true_label.shape[2]
        self.model,self.enc_model,self.dec_model = self.get_model() #create models
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.is_trained = False
    
    def preprocess_input_record(self,record):
        '''
        Record is a OrderedDict Representing one input
        data instance
        '''
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
        return temp

    def start_training(self):
        z = np.zeros((self.encoder_input.shape[0], self.hidden_dimesion))
        cb = EarlyStopping(monitor='val_loss',mode='min',patience=10,min_delta=0.01)
        self.training_output = self.model.fit(
        [self.encoder_input, self.decoder_input, z, z], self.decoder_true_label,
        batch_size=self.batch_size,
        epochs=self.epochs,
        validation_split=0.1,
        verbose = 1,
        callbacks=[cb]
        )
        self.enc_model.save(str(self.project_id)+"_enc_model.h5")
        self.dec_model.save(str(self.project_id)+"_dec_model.h5")
        self.is_trained = True


    def predict(self,input_data):
        '''
        Input data represents a OrderDict on which we need to run
        the model and get the prediction
        '''
        if not self.is_trained:
            raise Exception("This Model is not Trained Yet")
        self.enc_model = load_model(str(self.project_id)+"_enc_model.h5",custom_objects = {'softmax_over_time':self.softmax_over_time,'t':0})
        self.dec_model = load_model(str(self.project_id)+"_dec_model.h5",custom_objects = {'softmax_over_time':self.softmax_over_time,'t':0})
        self.enc_model._make_predict_function()
        self.dec_model._make_predict_function()
        test_input = self.preprocess_input_record(input_data)
        test_input = pad_sequences([test_input],maxlen=self.max_input_len,padding='post')
        pred_output = self.decode_sequence(test_input)
        return pred_output

    def save(self,filepath):
        with open(filepath,'wb') as f:
            pickle.dump(self,f)

    def decode_sequence(self,input_seq):
        word_to_index = self.tokenizer.word_index
        index_to_word = {v:k for k,v in word_to_index.items()}
        input_seq = input_seq.reshape(1,self.max_input_len)
        enc_out = self.enc_model.predict(input_seq)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = word_to_index['startseq']
        eos = word_to_index['endseq']
        s = np.zeros((1, self.hidden_dimesion))
        c = np.zeros((1, self.hidden_dimesion))
        output_sentence = []
        for _ in range(self.max_input_len):
            o, s, c = self.dec_model.predict([target_seq, enc_out, s, c])  
            idx = np.argmax(o.flatten())
            if eos == idx:
                break
            word = ''
            if idx > 0:
                word = index_to_word[idx]
                output_sentence.append(word)
            target_seq[0, 0] = idx

        return ' '.join(output_sentence)

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
        enc_model = Model(encoder_input_layer, encoder_outputs)
        encoder_outputs_as_input = Input(shape=(self.max_input_len, self.hidden_dimesion * 2,))
        decoder_inputs_single = Input(shape=(1,))
        decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
        context = one_step_attention(encoder_outputs_as_input, initial_s)
        decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])
        o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
        decoder_outputs = decoder_dense(o)
        dec_model = Model(
        inputs=[
            decoder_inputs_single,
            encoder_outputs_as_input,
            initial_s, 
            initial_c
        ],
        outputs=[decoder_outputs, s, c]
        )
        return model,enc_model,dec_model
                
                

            









                    
                    
            
                        


                
                
        
    

