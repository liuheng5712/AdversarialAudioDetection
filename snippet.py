# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:18:24 2019

@author: hengl
"""

"""
This script is meant to examine how much the inputs and hidden states are contributing to the final output
In percentage sense
"""
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
from shutil import copyfile
import scipy.io.wavfile as wav
import struct
import time
import os
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")
import DeepSpeech
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None
from util.text import ctc_label_dense_to_sparse
from tf_logits import get_logits
import scipy.io as sio
import inspect
toks = " abcdefghijklmnopqrstuvwxyz'-"
samples = []
###############################################################################
prefix = 'adver'
for i in range(100):
    if i < 10:
        samples.append('sample-00000' + str(i) + prefix + '.wav')
    else:
        samples.append('sample-0000' + str(i) + prefix + '.wav')
###############################################################################
"""
Here we build up the workflow to inspect inside the DeepSpeech when feed in an audio
first we build up the graph
"""

# The recurrent layer quantization bits starts at 2^4 and ends at 2^7
sess = tf.Session()
counter = 0
quant_BiRNN = [2, 3]
for q in quant_BiRNN:
    quant = [-1, -1, -1, q, -1, -1]
#    if prefix == 'adver':
#        os.system('touch results/adver/' + str(quant[3]) + '.txt')
#        write_path = 'results/adver/' + str(quant[3]) + '.txt'
#    elif prefix == 'benign':
#        os.system('touch results/benign/' + str(quant[3]) + '.txt')
#        write_path = 'results/benign/' + str(quant[3]) + '.txt'
#    f = open(write_path, 'w')
    
    for sample in samples[:10]:
        print('processing audio: ' + sample)    
        path = 'Datasets/Mozilla_subset/'
        fs, audio = wav.read(path + sample)
        #audio = audio.copy()
        audios = []
        lengths = []
        audios.append(list(audio)) 
        lengths.append(len(audio))
        audios = tf.convert_to_tensor(audios)
        
        with tf.variable_scope(tf.get_variable_scope().name, reuse = tf.AUTO_REUSE):
            logits, _, _, _ = get_logits(audios, (np.array(lengths)-1)//320, quant)
            decoded, _ = tf.nn.ctc_beam_search_decoder(logits, (np.array(lengths)-1)//320, merge_repeated=False, beam_width=100)
            if counter == 0:
                vars_name = ['b1:0', 'h1:0', 'b2:0', 'h2:0', 'b3:0', 'h3:0', 'bidirectional_rnn/fw/basic_lstm_cell/kernel:0', 'bidirectional_rnn/fw/basic_lstm_cell/bias:0', 'bidirectional_rnn/bw/basic_lstm_cell/kernel:0', 'bidirectional_rnn/bw/basic_lstm_cell/bias:0', 'b5:0', 'h5:0', 'b6:0', 'h6:0']
                saver = tf.train.Saver([v for v in tf.global_variables() if v.name in vars_name])
                saver.restore(sess, "models/session_dump")
                counter = counter + 1
        
        """print out the decoded sentence"""
        r_out = sess.run(decoded)
        lst = [(r_out, 1)]
        for out, logits in lst:
            chars = out[0].values
            res = np.zeros(out[0].dense_shape)+len(toks)-1
            for ii in range(len(out[0].values)):
                x,y = out[0].indices[ii]
                res[x,y] = out[0].values[ii]
            # Here we print the strings that are recognized.
            res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
            transcription = "\n".join(res)   
            print(transcription)
            #print >> f, transcription # python3: print(transcription, file = f)
    #f.close()


