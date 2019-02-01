from keras import Model
from keras.layers import LSTM,Dense,Bidirectional,Concatenate,Dot,Input,Lambda,RepeatVector,Embedding,Reshape
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras.backend as K
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from config import *



if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM

#Paths
EVENTS_DATASET_PATH = './nlg/dataset/all.events'
LABELS_DATASET_PATH = './nlg/dataset/all.text'


#Configs



#Ordinal Mappings Required dict
ordinal_mappings = { 
        '':1, '--':2, 'Chc':3, 
        'Def':4, 'E':5, 'ENE':6, 'ESE':7, 
        'Lkly':8, 'N':9, 'NE':10, 'NNE':11, 'NNW':12, 
        'NW':13, 'S':14, 'SChc':15, 'SE':16, 'SSE':17, 
        'SSW':18, 'SW':19, 'W':20, 
        'WNW':21, 'WSW':22
    }

def save_file(obj_to_save,filepath):
    '''
    saves object to file
    '''
    f = open(filepath,'wb')
    pickle.dump(obj_to_save,f)
    f.close()
    print('Saved At ',filepath)
    
    
def softmax_over_time(x):
  assert(K.ndim(x) > 2)
  e = K.exp(x - K.max(x, axis=1, keepdims=True))
  s = K.sum(e, axis=1, keepdims=True)
  return e / s


if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM


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

encoder_input,decoder_input,decoder_true_label =[],[],[]

feature_file = open(EVENTS_DATASET_PATH,'r')
labels_file = open(LABELS_DATASET_PATH,'r')

for feature,label in zip(feature_file,labels_file):
    try:
        encoder_input.append(preprocess_features(feature))
        lable = label.replace('%',' percent ')
        decoder_input.append('startseq '+label)
        decoder_true_label.append(label+' endseq')
    except Exception as e:
        print(e)
        pass


assert len(encoder_input) == len(decoder_input)

#randomly shuffling the data
zipped_data = list(zip(encoder_input,decoder_input,decoder_true_label))
random.shuffle(zipped_data) #shuffle data
encoder_input,decoder_input,decoder_true_label = zip(*zipped_data) #unzip the data


TRAIN_DATASET_SIZE = int(len(encoder_input) * 0.8)

max_len_input = max([len(x) for x in encoder_input])
max_len_output = max([len(x.split()) for x in decoder_input])

print(max_len_input)
print(max_len_output)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(decoder_input + decoder_true_label)
decoder_input_np = tokenizer.texts_to_sequences(decoder_input)
decoder_true_label_np = tokenizer.texts_to_sequences(decoder_true_label)

save_file(tokenizer,"token_v2.pkl")

encoder_input_np = pad_sequences(encoder_input,maxlen=max_len_input,padding='post')
decoder_input_np = pad_sequences(decoder_input_np,maxlen=max_len_output,padding='post')
decoder_true_label_np = pad_sequences(decoder_true_label_np,maxlen=max_len_output,padding='post')

#Training Data
encoder_input_np_train = encoder_input_np[:TRAIN_DATASET_SIZE]
decoder_input_np_train = decoder_input_np[:TRAIN_DATASET_SIZE]
decoder_true_label_np_train = decoder_true_label_np[:TRAIN_DATASET_SIZE]
                                                    
#Testing Data                                  
encoder_input_np_test = encoder_input_np[TRAIN_DATASET_SIZE:]
decoder_input_np_test = decoder_input_np[TRAIN_DATASET_SIZE:]
decoder_true_label_np_test = decoder_true_label_np[TRAIN_DATASET_SIZE:]

save_file((encoder_input_np_test,decoder_input_np_test,decoder_true_label_np_test),'testdata_v2.pkl')



decoder_true_label_np_train = to_categorical(decoder_true_label_np_train)

print(decoder_true_label_np_train.shape)


VOCAB_SIZE = decoder_true_label_np_train.shape[2]

print("Vocab size is ",VOCAB_SIZE)
print("Building Model..")

#start building model
encoder_input_layer = Input(shape=(max_len_input,))
reshape_layer = Reshape((max_len_input,1))
reshaped_input = reshape_layer(encoder_input_layer)
encoder_layer = Bidirectional(LSTM(HIDDEN_DIMENSION,return_sequences=True))
encoder_outputs = encoder_layer(reshaped_input)
decoder_input_layer = Input(shape=(max_len_output,))
decoder_embedding = Embedding(VOCAB_SIZE,EMBEDDING_DIMENSION)
decoder_inputs_x = decoder_embedding(decoder_input_layer)
attn_repeat_layer = RepeatVector(max_len_input)
attn_concat_layer = Concatenate(axis=-1)
attn_dense1 = Dense(10, activation='tanh')
attn_dense2 = Dense(1, activation=softmax_over_time)
attn_dot = Dot(axes=1) 

def one_step_attention(h, st_1):
  st_1 = attn_repeat_layer(st_1)
  x = attn_concat_layer([h, st_1])
  x = attn_dense1(x)
  alphas = attn_dense2(x)
  context = attn_dot([alphas, h])
  return context

decoder_lstm = LSTM(HIDDEN_DIMENSION, return_state=True)
decoder_dense = Dense(VOCAB_SIZE, activation='softmax')

initial_s = Input(shape=(HIDDEN_DIMENSION,), name='s0')
initial_c = Input(shape=(HIDDEN_DIMENSION,), name='c0')
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
  if t >= max_len_output:
    break
  
  

def stack_and_transpose(x):
  x = K.stack(x) 
  return K.permute_dimensions(x, pattern=(1, 0, 2))


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
print(model.summary())

print("Compiling Model..")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training Model..")
z = np.zeros((TRAIN_DATASET_SIZE, HIDDEN_DIMENSION))
r = model.fit(
  [encoder_input_np_train, decoder_input_np_train, z, z], decoder_true_label_np_train,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=0.1,
  verbose = 1
)


pred_encoder_model = Model(encoder_input_layer, encoder_outputs)

pred_encoder_model.save('enc_model.h5')
encoder_outputs_as_input = Input(shape=(max_len_input, HIDDEN_DIMENSION * 2,))
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)


context = one_step_attention(encoder_outputs_as_input, initial_s)
decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])
o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
decoder_outputs = decoder_dense(o)
decoder_model = Model(
  inputs=[
    decoder_inputs_single,
    encoder_outputs_as_input,
    initial_s, 
    initial_c
  ],
  outputs=[decoder_outputs, s, c]
)

decoder_model.save('dec_model.h5')

# plot loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig("loss_tracking.png",bbox_inches='tight')
plt.show()

#plot accuracy
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.savefig("acc_tracking.png",bbox_inches='tight')
plt.show()

model.save('training_model.h5')
