import pickle
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
cons=['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z']

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='name of the file where is saved the model to test')
args = parser.parse_args()
file=args.file

test_file = open("test.txt", "rb")
test_lis = pickle.load(test_file)
test_file.close()

test=np.array(test_lis)
y_test=test[:,-1]
X_test=test[:,:-1]


clf=pickle.load((open(file,'rb')))
y_pred=clf.predict(X_test)
print('Accuracy', accuracy_score(y_test, y_pred))
print('Precision', list(zip(cons,list(precision_score(y_pred,y_test, average=None)))))
print('recall', list(zip(cons,list(recall_score(y_pred,y_test, average=None)))))
print('f score', list(zip(cons,list(f1_score(y_pred,y_test, average=None)))))