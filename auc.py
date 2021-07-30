import time
import tensorflow as tf
from utils import *

from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pylab as plt
import numpy as np

outs = np.loadtxt(r'D:\global_loocv_93.01.txt', delimiter=',')
path_md_origin = r'C:\Users\joyce\Desktop\exp\Bipartite-Local-Models-and-hubness-aware-regression-master\Bipartite-Local-Models-and-hubness-aware-regression-master\DATA\5.temp-result\m_d.txt'
m_d_origin = np.loadtxt(path_md_origin, delimiter=',')
outs_2 = np.loadtxt(r'D:\global_loocv_91.63.txt', delimiter=',')
outs_3 = np.loadtxt(r'D:\global_loocv_92.62.txt', delimiter=',')
outs_4 = np.loadtxt(r'D:\global_loocv_92.64.txt', delimiter=',')
outs_5 = np.loadtxt(r'D:\global_loocv_91.63.txt', delimiter=',')

        #############画图部分
fpr, tpr, threshold = metrics.roc_curve(list([int(i) for i in m_d_origin.flatten()]), list(outs.T.flatten()))
roc_auc = metrics.auc(fpr, tpr)

fpr_2, tpr_2, threshold = metrics.roc_curve(list([int(i) for i in m_d_origin.flatten()]), list(outs_2.T.flatten()))
roc_auc_2 = metrics.auc(fpr_2, tpr_2)

fpr_3, tpr_3, threshold = metrics.roc_curve(list([int(i) for i in m_d_origin.flatten()]), list(outs_3.T.flatten()))
roc_auc_3 = metrics.auc(fpr_3, tpr_3)

fpr_4, tpr_4, threshold = metrics.roc_curve(list([int(i) for i in m_d_origin.flatten()]), list(outs_4.T.flatten()))
roc_auc_4 = metrics.auc(fpr_4, tpr_4)

fpr_5, tpr_5, threshold = metrics.roc_curve(list([int(i) for i in m_d_origin.flatten()]), list(outs_5.T.flatten()))
roc_auc_5 = metrics.auc(fpr_5, tpr_5)

#plt.title('Global LOOCV')
plt.title('5-fold cross validation')
plt.plot(fpr, tpr, 'r', label='RWR= %0.4f' % roc_auc)
plt.plot(fpr_2, tpr_2, 'b', linestyle='--', label='no RWR= %0.4f' % roc_auc_2)
#plt.plot(fpr_3, tpr_3, 'y',alpha=0.4,  label='no enhanced layer= %0.4f' % roc_auc_3)
#plt.plot(fpr_4, tpr_4, 'g',alpha=0.4,  label='no enhanced layer= %0.4f' % roc_auc_4)
#plt.plot(fpr_5, tpr_5, 'b',alpha=0.4,  label='no enhanced layer= %0.4f' % roc_auc_5)

plt.plot([0, 1], [0, 1], color='#000000', linestyle='--')
plt.legend(loc='lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()