import sys,os
from importlib import reload
import numpy as np
from train import Training
reload(sys)


# sys.setdefaultencoding('utf-8')
#################################################

# # # # # PATH # # # # #
full_path=os.path.realpath(__file__)
# print(os.path.basename(__file__))
eop=full_path.rfind(__file__)
eop=full_path.rfind(os.path.basename(__file__))
main_path=full_path[0:eop]
folder_path=full_path[0:eop]+u'data'

if __name__ == "__main__":
  # path='C:/Users/joyce/Desktop/GCNMDA-master/data/MDAD/adj.txt'
  # # path_train_data=folder_path+u'\MDAD\\adj.txt'
  # # path_test_data=folder_path+u'\MDAD\\adj.txt'
  # # test_data = np.loadtxt(path_test_data)
  # # train_data = np.loadtxt(path_train_data)
  # test_data=np.loadtxt(path)
  # train_data = np.loadtxt(path)
  # print(test_data)



  # train_data=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
  # test_data=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
  train_data=np.arange(5430)
  test_data = np.arange(5430)
  gcn = Training()
  gcn.train(train_data,test_data)
 