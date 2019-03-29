import pickle
from keras.models import load_model
import numpy as np
import keras.backend as K
from nltk.translate.bleu_score import corpus_bleu
from config import *


def calculate_bleu(predicted,actual):
    '''
    calculates bleu scores for test set
    '''
    hypothesis = []
    references = []
    for pred,act in zip(predicted,actual):
        act = [word for word in act.split() if word.isalnum()]
        references.append([act])
        hypothesis.append(pred.split())
    print(references[:2])
    print(hypothesis[:2])
    print('BLEU-1: %f' % corpus_bleu(references,hypothesis, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(references,hypothesis,weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(references,hypothesis, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(references,hypothesis,weights=(0.25, 0.25, 0.25, 0.25)))



def softmax_over_time(x):
  assert(K.ndim(x) > 2)
  e = K.exp(x - K.max(x, axis=1, keepdims=True))
  s = K.sum(e, axis=1, keepdims=True)
  return e / s

enc_model = load_model('cpu/enc_model.h5',custom_objects = {'softmax_over_time':softmax_over_time,'t':0})
dec_model = load_model('cpu/dec_model.h5',custom_objects = {'softmax_over_time':softmax_over_time,'t':0})



token = pickle.load(open('cpu/token_v2.pkl','rb'))
testdata = pickle.load(open('cpu/testdata_v2.pkl','rb'))
encoder_input,decoder_input,decoder_true_input = testdata
word_to_index = token.word_index
index_to_word = {v:k for k,v in word_to_index.items()}



def decode_sequence(input_seq):
  input_seq = input_seq.reshape(1,MAX_LEN_INPUT)
  enc_out = enc_model.predict(input_seq)
  target_seq = np.zeros((1, 1))
  target_seq[0, 0] = word_to_index['startseq']
  eos = word_to_index['endseq']
  s = np.zeros((1, HIDDEN_DIMENSION))
  c = np.zeros((1, HIDDEN_DIMENSION))
  output_sentence = []
  for _ in range(MAX_LEN_OUTPUT):
    o, s, c = dec_model.predict([target_seq, enc_out, s, c])  
    idx = np.argmax(o.flatten())
    if eos == idx:
      break
    word = ''
    if idx > 0:
      word = index_to_word[idx]
      output_sentence.append(word)
    target_seq[0, 0] = idx

  return ' '.join(output_sentence)

preds = []
references = []
num_samples = encoder_input.shape[0]
for i in range(5):
  test_input = encoder_input[i]
  true_output = decoder_true_input[i]
  pred_output = decode_sequence(test_input)
  #print("================================================================")
  print("Test Input :",test_input)
  #print("------------------------------")
  true_output_text = " ".join([index_to_word[x] for x in true_output if x > 0])
  true_output_text = " ".join(true_output_text.split()[:-1])
  print("True Output :",true_output_text)
  #print("-----------------------------------")
  print("Predicted Output:",pred_output)
  #print(i)
  preds.append(pred_output)
  references.append(true_output_text)
calculate_bleu(preds,references)