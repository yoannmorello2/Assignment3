import pickle
import argparse
import numpy as np
cons=['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z']
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

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', type=str, required=True, help='sentence')
parser.add_argument('--file', type=str, required=True, help='Name of the file containing the model')
args = parser.parse_args()
sentence=args.sentence
file=args.file

enc_file = open('enc.pkl', 'rb')
enc=pickle.load(enc_file)
enc_file.close()
clf=pickle.load((open(file,'rb')))

sentences=[''.join(i for i in sentence if not i.isdigit()).rstrip('\'\"').removesuffix('\\n').removeprefix('b\'').removeprefix('b\"').lstrip('.')]
print(sentences)
X_0=mat_4_letters_first_cons(sentences, 10000)[:,:4]
X=enc.transform(X_0).toarray()
M=clf.predict_proba(X)
perp=2**(-np.sum(np.log2(M)*M)/(M.shape[0]))
print('perplexity=',perp)