from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from utils import *
from models import GAutoencoder
from metrics import *
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pylab as plt
import numpy as np


class Training():
  def __init__(self):
     self.model ='GAutoencoder'
     self.learning_rate = 0.0001
     self.dropout = 0.1
     #  5e-4
     self.weight_decay = 1e-5
     #100
     self.early_stopping = 200
     self.max_degree = 3
     self.latent_factor = 256
     #200
     self.epochs =500
        
  def train(self,train_arr, test_arr):
        # Settings
    
    # Load data
    adj, features, size_u, size_v, logits_train, logits_test, train_mask, test_mask, labels = load_data(train_arr, test_arr) 
    # Some preprocessing
    if self.model == 'GAutoencoder':
        model_func = GAutoencoder
    else:
        raise ValueError('Invalid argument for model: ' + str(self.model))
    
    
    # Define placeholders
    placeholders = {
        'adjacency_matrix': tf.compat.v1.placeholder(tf.float32, shape=adj.shape),
        'Feature_matrix': tf.compat.v1.placeholder(tf.float32, shape=features.shape),
        'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, logits_train.shape[1])),
        'labels_mask': tf.compat.v1.placeholder(tf.int32),
        'negative_mask': tf.compat.v1.placeholder(tf.int32)
    }
    
    # Create model
        #25
    self.latent_factor_num=256
    model = model_func(placeholders, size_u, size_v, self.latent_factor_num)
    
    # Initialize session
    sess = tf.compat.v1.Session()
    
    
    # Init variables
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Define model evaluation function
    def evaluate(adj, features, labels, mask, negative_mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(adj, features,labels, mask, negative_mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)
  
    # Train model
    for epoch in range(self.epochs):
        t = time.time()
        # Construct feed dictionary
        negative_mask, label_neg = generate_mask(labels, len(train_arr))
        feed_dict1 = construct_feed_dict(adj, features, logits_train, train_mask, negative_mask, placeholders)
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict1)

        print("Epoch:", '%04d' % (epoch + 1), 
              "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), 
              "time=", "{:.5f}".format(time.time() - t))
     
    print("Optimization Finished!")
     
    # Testing
    test_cost, test_acc, test_duration = evaluate(adj, features, logits_test, test_mask, negative_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
 
    # Obtaining predicted probability scores
    feed_dict_val = construct_feed_dict(adj, features, logits_test, test_mask, negative_mask, placeholders)
    outs = sess.run(model.outputs, feed_dict=feed_dict_val)
    outs = outs.reshape((383,495))
    print(outs)
    np.savetxt("D:\global_loocv_9.txt", outs, delimiter=',', fmt='%f')


    #hid = sess.run(model.hid, feed_dict=feed_dict_val)
    path_md_origin = r'C:\Users\joyce\Desktop\exp\Bipartite-Local-Models-and-hubness-aware-regression-master\Bipartite-Local-Models-and-hubness-aware-regression-master\DATA\5.temp-result\m_d.txt'
    m_d_origin = np.loadtxt(path_md_origin, delimiter=',')
    print(outs.T.flatten())
    print(m_d_origin.flatten())

        #############画图部分
    fpr, tpr, threshold = metrics.roc_curve(list([int(i) for i in m_d_origin.flatten()]), list(outs.T.flatten()))
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label='Val AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

  
    
