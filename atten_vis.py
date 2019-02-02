import pickle
from keras.models import load_model
import numpy as np
import keras.backend as K
from nltk.translate.bleu_score import corpus_bleu
from config import *
import csv
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

def softmax_over_time(x):
  assert(K.ndim(x) > 2)
  e = K.exp(x - K.max(x, axis=1, keepdims=True))
  s = K.sum(e, axis=1, keepdims=True)
  return e / s



enc_model = load_model('cpu/enc_model.h5',custom_objects = {'softmax_over_time':softmax_over_time,'t':0})
dec_model = load_model('cpu/dec_model.h5',custom_objects = {'softmax_over_time':softmax_over_time,'t':0})
token = pickle.load(open('cpu/token_v2.pkl','rb'))
word_to_index = token.word_index
from keras.models import Model



layer_name = 'dense_2'
intermediate_layer_model = Model(inputs=dec_model.input,
                                 outputs=dec_model.get_layer(layer_name).output)

    

ordinal_mappings = { 
        '':1, '--':2, 'Chc':3, 
        'Def':4, 'E':5, 'ENE':6, 'ESE':7, 
        'Lkly':8, 'N':9, 'NE':10, 'NNE':11, 'NNW':12, 
        'NW':13, 'S':14, 'SChc':15, 'SE':16, 'SSE':17, 
        'SSW':18, 'SW':19, 'W':20, 
        'WNW':21, 'WSW':22
    }

def parse_csv(path):
    f = open(path,'r')
    reader = csv.reader(f)
    rows = [row for row in reader]
    f.close()
    return rows



def preprocess_features(feature):
    '''
    creates a feature vector and returns it
    '''
    x = []
    input_attr = []
    for att in feature:
        #ignore mode bucket attribute for now
        if not 'mode-bucket' in att[0]:
            if 'time' in att[0]:
                time_vals = att[1].split('-')
                x.append(int(time_vals[0]))
                x.append(int(time_vals[1]))
                input_attr.append(att[0]+" 1")
                input_attr.append(att[0]+" 2")
            elif 'mode' in att[0]:
                x.append(ordinal_mappings[att[1]])
                input_attr.append(att[0])
            else:
                x.append(int(att[1]))
                input_attr.append(att[0])
    
    return x,input_attr








def decode_sequence(input_seq):
    index_to_word = {v:k for k,v in word_to_index.items()}
    input_seq = input_seq.reshape(1,MAX_LEN_INPUT)
    enc_out = enc_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word_to_index['startseq']
    eos = word_to_index['endseq']
    s = np.zeros((1, HIDDEN_DIMENSION))
    c = np.zeros((1, HIDDEN_DIMENSION))
    output_sentence = []
    att_wt = []
    for _ in range(MAX_LEN_OUTPUT):
        o, s, c = dec_model.predict([target_seq, enc_out, s, c])
        intermediate_output = intermediate_layer_model.predict([target_seq, enc_out, s, c])
        intermediate_output = intermediate_output.reshape(1,MAX_LEN_INPUT)
        att_wt.append(intermediate_output)
        idx = np.argmax(o.flatten())
        if eos == idx:
            break
        word = ''
        if idx > 0:
            word = index_to_word[idx]
            output_sentence.append(word)
        target_seq[0, 0] = idx

    return ' '.join(output_sentence),att_wt

def get_prediction_on_csv(path):
    rows = parse_csv(path)
    test_input,input_attr = preprocess_features(rows)
    print(test_input)
    test_input = pad_sequences([test_input],maxlen=MAX_LEN_INPUT,padding='post')
    pred_output,att_wt = decode_sequence(test_input)
    att_wt = K.stack(att_wt,axis = 1)
    att_wt = K.reshape(att_wt,(att_wt.shape[1],att_wt.shape[2]))
    print(att_wt.shape)
    att_wt = K.eval(att_wt)
    print(pred_output)
    plt.matshow(att_wt)
    plt.xticks(np.arange(att_wt.shape[1]),input_attr,rotation = 'vertical')
    plt.yticks(np.arange(att_wt.shape[0]),pred_output.split())
    plt.show()
    return pred_output

get_prediction_on_csv('input_csv.csv')