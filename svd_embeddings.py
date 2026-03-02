#@Author: Akash Manna, IIIT Hyderabad
#@Date: 02/03/2026


import os
import sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

nltk.download('brown')
nltk.download('stopwords')
#Step-1: First load brown coupus from nltk
from nltk.corpus import brown

brown_corpus= [x.lower() for x in brown.words()]
punctuation = ['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~','``',"''",'--']
brown_corpus= [x.lower() for x in brown.words()]
pun_stop = punctuation + stopwords.words('english')

brown_corpus= filter_words1 = [x for x in brown_corpus if x not in pun_stop]
brown_corpus= list(filter(lambda x: x.isalpha() and len(x) > 1, filter_words1)) # remove numbers and single letter words

#Step-2: Implement co-occurance matrix
words_freq = defaultdict(int)
for line in tqdm(brown_corpus):
    line = line.split()
    for word in line:
        words_freq[word]+=1

brown_corpus_filtered = []
for sentence in tqdm(brown_corpus):
    brown_corpus_filtered.append(' '.join([word for word in sentence.split() if words_freq[word]>10]))

vectorizer = CountVectorizer()
vectorized_mat = vectorizer.fit_transform(brown_corpus_filtered)
token_list = vectorizer.get_feature_names_out()

print("Count Matrix Shape: ", vectorized_mat.shape)

co_occ_mat = vectorized_mat.T*vectorized_mat

# Occurence of same word one after other is almost never. hence set diagonal to zero.
co_occ_mat.setdiag(0)
print("Shape of Co-Occurence Matrix", co_occ_mat.shape)


#Step-3: Apply SVD on Cooccurance matrix
svd = TruncatedSVD(n_components=350) #Setting top 350 features
svd_mat = svd.fit_transform(co_occ_mat)
svd = TruncatedSVD(n_components=350)
svd_mat = svd.fit_transform(co_occ_mat)

print("Explained Variance Ratio: ", svd.explained_variance_ratio_)
print("Eigen Values: ", svd.singular_values_)

print("The shape SVD matrix obtained using top k eigen-vectors: ", svd_mat.shape)

#Saving the svd_matrix
