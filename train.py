import pickle
import argparse
import numpy as np

train_file = open("train.txt", "rb")
train_lis = pickle.load(train_file)
train_file.close()


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='name of the file where to save the model')
parser.add_argument('--model', type=int, required=True, help='0 for MultinomialNB, 1 for SVC')
args = parser.parse_args()

train=np.array(train_lis)


file=args.file
X=train[:,:-1]
y=train[:,-1]
from  sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
if args.model == 0:
    clf = MultinomialNB()
    clf.fit(X, y)
else:
    clf =SVC(kernel='linear')
    clf.fit(X, y)
    
with open(file, 'wb') as f:
    pickle.dump(clf,f )   
 
      
