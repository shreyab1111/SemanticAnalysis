from __future__ import print_function, division
from builtins import range
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer=WordNetLemmatizer()

titles=[line.rstrip() for line in open('all_book_titles.txt')]
stopwords= set(w.rstrip() for w in open('stopwords.txt'))
stopwords=stopwords.union({'introduction','edition','series','application','approach','card','access','package','plus',
                           'etext','brief','vol','guide','fundamental','essential','printed','third','second','fourth',})

def my_tokenizer(s):
    s=s.lower() #no capital letters
    tokens=nltk.tokenize.word_tokenize(s) #works like split() in JAVA
    tokens=[t for t in tokens if len(t)>2] #removing a, an, to.. such small words
    tokens=[wordnet_lemmatizer.lemmatize(t) for t in tokens] #lemmetizing the words to their original form
    tokens=[t for t in tokens if t not in stopwords] #removing
    tokens=[t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens
    
word_index_map={}
current_index=0
all_tokens=[]
all_titles=[]
index_word_map=[]

error_count=0
for title in titles:
    try:
        title=title.encode('ascii','ignore').decode('utf-8')# this will throw exception if bad characters
        all_titles.append(title)
        tokens=my_tokenizer(title)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token]=current_index
                current_index+=1
                index_word_map.append(token)
    except Exception as e:
        print(e)
        print(title)
        error_count += 1
        pass       
print("Number of errors parsing file:", error_count, "number of lines in file:", len(titles))
if error_count == len(titles):
    print("There is no data to do anything with! Quitting...")
#exit() 

#take the tokens and convert them into a bunch of numbers acc to their proportions
#this is unsupervised learning
def tokens_to_vector(tokens):
    X=np.zeros(len(word_index_map))
    for t in tokens:
        i=word_index_map[t]
        X[i]=1 #just indicator variable not sum
    return X
    
N= len(all_tokens)
D=len(word_index_map)
X = np.zeros((D, N)) # terms will go along rows, documents along columns
i=0
for tokens in all_tokens:
    X[:,i]=tokens_to_vector(tokens)
    i+=1
    
def main():
    svd=TruncatedSVD()
    Z=svd.fit_transform(X)
    plt.scatter(Z[:,0], Z[:,1])
    for i in range(D):
        plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))
    plt.show()

if __name__ == '__main__':
    main()    
