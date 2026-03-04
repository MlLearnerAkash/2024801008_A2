#@Author: Akash Manna, IIIT Hyderabad
#@Date: 02/03/2026


import os
import sys
import numpy as np
import torch
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

# brown_corpus= filter_words1 = [x for x in brown_corpus if x not in pun_stop]
brown_sents = [
    ' '.join([w.lower() for w in sent if w.isalpha() and w.lower() not in pun_stop])
    for sent in brown.sents()
]
# brown_corpus= list(filter(lambda x: x.isalpha() and len(x) > 1, filter_words1)) # remove numbers and single letter words

#Step-2: Implement co-occurance matrix
words_freq = defaultdict(int)
for line in tqdm(brown_sents):
    line = line.split()
    for word in line:
        words_freq[word]+=1

brown_corpus_filtered = [
    ' '.join([w for w in sent.split() if words_freq[w] > 10])
    for sent in brown_sents
]

vectorizer = CountVectorizer()
vectorized_mat = vectorizer.fit_transform([s for s in brown_corpus_filtered if s.strip()])
token_list = vectorizer.get_feature_names_out()

print("Count Matrix Shape: ", vectorized_mat.shape)

co_occ_mat = vectorized_mat.T*vectorized_mat

# Occurence of same word one after other is almost never. hence set diagonal to zero.
co_occ_mat.setdiag(0)
print("Shape of Co-Occurence Matrix", co_occ_mat.shape)


#Step-3: Apply SVD on Cooccurance matrix
svd = TruncatedSVD(n_components=350) #Setting top 350 features
svd_mat = svd.fit_transform(co_occ_mat)

print("Explained Variance Ratio: ", svd.explained_variance_ratio_)
print("Eigen Values: ", svd.singular_values_)

print("The shape SVD matrix obtained using top k eigen-vectors: ", svd_mat.shape)


#Saving the svd_matrix
os.makedirs('embeddings', exist_ok=True)

# Save as a dict: word -> embedding tensor
svd_tensor = torch.tensor(svd_mat, dtype=torch.float32)   # (vocab_size, 350)
token_list_py = list(token_list)                           # list of vocab words

torch.save({
    'embeddings': svd_tensor,   # shape (V, 350)
    'vocab': token_list_py      # list of V words (index i → token_list_py[i])
}, 'embeddings/svd.pt')

print(f"Saved SVD embeddings to embeddings/svd_embeddings.pt | Shape: {svd_tensor.shape}")
