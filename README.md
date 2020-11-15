# python-Jupyter-Notebook-
## I would like to import STL-10 datasets using smillar method as it is done as for CIFAR10 datasets below 
## My goal to be able to learn how to import STL-10 in simllar way as cifar10 and to be examine the dataset and analys the data   
import cifar10 
import matplotlib.pyplot as plt 
import numpy as np  

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.model_selection import cross_val_score 
from sklearn.decomposition import PCA  

from sklearn.neighbors import KNeighborsClassifier  

## Importing Datsets 

cifar10.data_path = "data/CIFAR-10/" 

cifar10.maybe_download_and_extract()  

class_names = cifar10.load_class_names() class_names  

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()
