import scipy.io as sio
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


sample_num = 100 # the sample amount for each class
Mozilla_benign_CER = sio.loadmat('/home/hengl/matlab/bin/scripts/ActQuantAdver/Mozilla_results/ErrRate.mat')['benign_CER']/100.
Mozilla_carlini_CER = sio.loadmat('/home/hengl/matlab/bin/scripts/ActQuantAdver/Mozilla_results/ErrRate.mat')['genericcarlini_CER']/100.
#Libri_benign_CER = sio.loadmat('/home/hengl/matlab/bin/scripts/ActQuantAdver/Libri_results/ErrRate.mat')['benign_CER']/100.
#Libri_carlini_CER = sio.loadmat('/home/hengl/matlab/bin/scripts/ActQuantAdver/Libri_results/ErrRate.mat')['heng_CER']/100.
benign_CER = Mozilla_benign_CER
carlini_CER = Mozilla_carlini_CER

# this is  the train test split
threshold_levels = np.linspace(0, 1, 101).tolist()
threshold_levels.reverse()
ratio = (sample_num * np.linspace(0.1, 0.8, 15)).astype(int).tolist()
for idx in ratio:
    train_index = range(idx/2)
    val_index = range(idx/2, idx)
    test_index = range(idx, sample_num)
    
    accu = 0.
    precision = 0.
    recall = 0.
    auc = 0.
    fpr_final = []
    tpr_final = []
    rounds = 50
    for r in range(rounds):
        # shuffle the audios and break the corresponding relationship between benign and adversarial audios
        index = np.random.permutation(sample_num)
        benign_CER = benign_CER[:, :, index]
        index = np.random.permutation(sample_num)
        carlini_CER = carlini_CER[:, :, index]        
        # now we do train-validation to choose which quant bits to use
        best_accu = 0.0
        index1 = -1
        index2 = -1
        final_threshold = -1
        for FCN in range(9):
            for biRNN in range(9):    
            		# We only use the character error rate because it has more details, the threshold can be determined by min/max
            		# Use training data to get a threshold
            		threshold = (np.min(carlini_CER[:, :, train_index], axis = 2)[FCN][biRNN] + np.max(benign_CER[:, :, train_index], axis = 2)[FCN][biRNN])/2
            		# Validate the threshold on validation dataset
            		benign_audios = benign_CER[:, :, val_index][FCN, biRNN]
            		carlini_audios = carlini_CER[:, :, val_index][FCN, biRNN]
            		true_positive = np.sum(benign_audios < threshold).astype(np.float)
            		true_negative = np.sum(carlini_audios > threshold).astype(np.float)
            		accuracy = (true_positive + true_negative)/(len(benign_audios) + len(carlini_audios))
            		# use the quant bits with the best accuracy
            		if best_accu < accuracy:
            			best_accu = accuracy
            			index1 = FCN
            			index2 = biRNN
            			final_threshold = threshold
               
        # we apply the validated threshold on testing dataset      
        threshold = final_threshold
        benign_audios = benign_CER[:, :, test_index][index1][index2]
        carlini_audios = carlini_CER[:, :, test_index][index1][index2]
        true_positive = np.sum(benign_audios < threshold).astype(np.float)
        true_negative = np.sum(carlini_audios > threshold).astype(np.float)
        # calculate the accuracy overall for both benign and adversarial audios
        accu = np.round((true_positive + true_negative)/(2 * sample_num - 2 * idx), 3) + accu
        # precision = tp/(tp + fp)
        precision = np.round(true_positive/(true_positive + (sample_num - idx - true_positive)), 3) + precision
        # recall = tp/(tp + fn)
        recall = np.round(true_positive/(true_positive + (sample_num - idx - true_negative)), 3) + recall
        
        #calculate the AUC using sklearn
        num = len(benign_audios.tolist())
        # 1 for benign class and 0 for adversarial class
        y = [1] * num + [0] * num
        # here we calculate the scores by mapping instead of truncating
        # multiplied by -1 because smaller error rate implied higher confidence score of benign audio
        pred = [-1*i for i in benign_audios.tolist() + carlini_audios.tolist()]
        mini = min(pred)
        maxi = max(pred)
        # normalize 
        pred = [(i - mini)/(maxi - mini) for i in pred]
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label = 1, drop_intermediate = False)
        fpr_final.append(fpr)
        tpr_final.append(tpr)
        auc = metrics.auc(fpr, tpr) + auc  
        
        if idx == 50 and r == 1:
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=20)
            plt.ylabel('True Positive Rate', fontsize=20)
            plt.title('Receiver operating characteristic', fontsize=20)
            plt.savefig('rocgenericcarlini.pdf', quality = 100, format = 'pdf')
            plt.show()

    #print(str(idx/100.) + '&' + str(np.round(accu/50, 4)) + '&' + str(np.round(precision/50, 4)) + '&' + str(np.round(recall/50, 4)) + '&' + str(np.round(auc/50, 4)) + '\\\\')
        
        
        
   
        

