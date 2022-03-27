import numpy as np
import argparse
import gzip
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='the path of a zip file')
parser.add_argument('--n_samples', type=int, required=True, help='Number of samples to use')
parser.add_argument('--prop_test', type=float, required=True, help='proportion of the samples to keep for testing')
args = parser.parse_args()
print("n_samples=" + str(args.n_samples))

cons=['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z']
n_samples=args.n_samples
path=args.path
prop_test=args.prop_test

def helper(sentence, position):
    a=np.zeros(5,dtype=object)
    l=len(sentence)
    for j in range(4):
        if (position+4)<l:
            a[j]=sentence[position+j]
        else: 
            return None
    k=4
    while (sentence[position+k].lower() not in cons) & (position+k<l-1):
        k+=1
    if sentence[position+k].lower() in cons:
        a[4]=sentence[position+k].lower()
    if a[4]==0:
        return None
    else:
        return a

def mat_4_letters_first_cons(texte, n_samples=20):
    matrix=np.empty((1,5))
    n=0
    for sentence in texte:
      
        for position in range(len(sentence)-5):
            h = helper(sentence, position)
            if type(h)== np.ndarray:
                matrix=np.vstack((matrix,h))
                n+=1
            if n==n_samples: break
        if n==n_samples: break 
    return matrix[1:,:]

def sample_lines(path,lines):
    f=gzip.open(path,'rb')
    return random.sample(list(f),lines)

#function which transforms the list of sentences in a list of strings and get rid of the b' and the \n'
def list_of_lists_of_signs(s):
    return [''.join(i for i in str(sent) if not i.isdigit()).rstrip('\'\"').removesuffix('\\n').removeprefix('b\'').removeprefix('b\"').lstrip('.') for sent in s]

sample=sample_lines(path,round(n_samples/5))
texte=list_of_lists_of_signs(sample)
matrix=mat_4_letters_first_cons(texte, n_samples)
y=matrix[:,4]
X_0=matrix[:,:4]

enc = OneHotEncoder()
m=enc.fit_transform(X_0).toarray()


def split_samples(X_0,y, prop_test):
   
    return train_test_split(X_0, y, test_size=prop_test)

train_X, test_X,train_y, test_y=split_samples(m,y,prop_test)

list_train=list(np.concatenate((train_X,train_y[:,None]),axis=1))
list_test=list(np.concatenate((test_X,test_y[:,None]),axis=1))

enc_file = open('enc.pkl', 'wb')
pickle.dump(enc, enc_file)
enc_file.close()
train_file = open('train.txt', 'wb')
pickle.dump(list_train, train_file)
train_file.close()
test_file = open('test.txt', 'wb')
pickle.dump(list_test, test_file)
test_file.close()