import os
import numpy as np 
from asrtoolkit import wer
import asr.align as align
import scipy.io as sio


carlini_WER = -1. * np.ones((9, 9, 100))
carlini_CER = -1. * np.ones((9, 9, 100))
benign_WER = -1. * np.ones((9, 9, 100))
benign_CER = -1. * np.ones((9, 9, 100))
carlini = open('../Mozilla_results/genericcarlini.txt', 'r')
carlini = carlini.read().splitlines()
benign = open('../Mozilla_results/benign.txt', 'r')
benign = benign.read().splitlines()

for FCN_quant in range(1, 10):
    print('processing FCN: ', FCN_quant)
    for biRNN_quant in range(1, 10):
        print('processing BiRNN:', biRNN_quant)
        carlini_quant = open('../Mozilla_results/genericcarlini/FCN=' + str(FCN_quant) + '/' + str(biRNN_quant) + '.txt', 'r')
        carlini_quant = carlini_quant.read().splitlines()
        benign_quant = open('../Mozilla_results/benign/FCN=' + str(FCN_quant) + '/' + str(biRNN_quant) + '.txt', 'r')
        benign_quant = benign_quant.read().splitlines()
        
        ##########################################################
        # collect score for 100 audios and average
        benign_CER_temp = []
        carlini_CER_temp = []
        # calculating the Character Error Rate (CER)
        for index in range(100):
            # process the transcripts in the carlini (adversarial)
            reference = carlini[index]
            hypothesis = carlini_quant[index]
            # if the hypothesis and the reference are not empty
            if len(hypothesis) != 0 and len(reference) != 0:
                carlini_CER_temp.append(wer.cer(reference, hypothesis))
                carlini_CER[FCN_quant-1][biRNN_quant-1][index] = wer.cer(reference, hypothesis) 
            else:
                carlini_CER_temp.append(-1)
            # process the transcripts in the benign
            reference = benign[index]
            hypothesis = benign_quant[index]
            # if the hypothesis and the reference are not empty
            if len(hypothesis) != 0 and len(reference) != 0:
                benign_CER_temp.append(wer.cer(reference, hypothesis))   
                benign_CER[FCN_quant-1][biRNN_quant-1][index] = wer.cer(reference, hypothesis)
            else:
                benign_CER_temp.append(-1)
                
       
        ##########################################################
        # collect score for 100 audios and average
        carlini_WER_temp = []
        benign_WER_temp = []
        # calculating the Word Error Rate (WER)
        for index in range(100):
            # process the transcripts in the carlini (adversarial)
            reference = carlini[index].split()
            hypothesis = carlini_quant[index].split()
            temp = align.calculate_wer(reference, hypothesis)
            if np.isnan(temp):
                carlini_WER_temp.append(-1)
            else:
                carlini_WER_temp.append(temp)
                carlini_WER[FCN_quant-1][biRNN_quant-1][index] = temp
            # process the transcripts in the benign
            reference = benign[index].split()
            hypothesis = benign_quant[index].split()
            temp = align.calculate_wer(reference[:len(hypothesis)], hypothesis)
            if np.isnan(temp):
                benign_WER_temp.append(-1)
            else:
                benign_WER_temp.append(temp)
                benign_WER[FCN_quant-1][biRNN_quant-1][index] = temp
            

    
sio.savemat('ER.mat', {'genericcarlini_WER': carlini_WER, 'genericcarlini_CER': carlini_CER, 'benign_WER': benign_WER, 'benign_CER': benign_CER})











