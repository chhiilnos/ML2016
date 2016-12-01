import sys
import re
import code
import sys
import nltk 
import numpy as np
import csv
import pickle
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

titles=[]
docs=[]

#read in titles and docs
file = open(sys.argv[1]+'title_StackOverflow.txt', 'r')
for line in file.readlines():
    titles.append(line)
file.close()

file = open(sys.argv[1]+'docs.txt', 'r')
for line in file.readlines():
    docs.append(line)
file.close()

print('read titles_StackOverflow.txt  and docs.txt done')


#define a funtion to tokenize and stem text
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

print('define tokenize_and_stem done')


#parameters of truncated SVD
n_components=21
max_df=0.8
max_features=15000
min_df=1
num_clusters =36
n_init=40

print('define parameters done')


#build tfidf_vectorizer and calculate tfidf_matrix of titles
tfidf_vectorizer = TfidfVectorizer( max_df=max_df, max_features=max_features , min_df=min_df, stop_words='english', use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,1) )
tfidf_matrix = tfidf_vectorizer.fit(titles+docs) 
tfidf_matrix = tfidf_vectorizer.transform(titles) 

print('tfidf_matrix done')


#now extract features from tfidf_matrix by LSA (aka truncated SVD+l2-normalizer)
svd = TruncatedSVD( n_components = n_components )
normalizer = Normalizer( copy = False )
lsa = make_pipeline( svd , normalizer )
X = tfidf_matrix.toarray()
X = lsa.fit_transform(X)

print('LSA done')


#perform k means on the extracted features of titles
km = KMeans( n_clusters = num_clusters , n_init = n_init )
km.fit(X)
label = km.labels_

print('k means done')


#now read in pairs of title indices form check_index.csv
x_ID = []
y_ID = []
f = open( sys.argv[1]+'check_index.csv','r' )
for row in csv.reader(f) :
        x_ID.append(row[1])
        y_ID.append(row[2])
f.close()

print('read check_index.csv done')


#ultimately output the result
f = open( sys.argv[1] + sys.argv[2],'w' )
w = csv.writer( f )
f.write('ID,Ans\n')
for i in range (5000000) :
        if( label[int(x_ID[i+1])] == label[int(y_ID[i+1])] ) :
                w.writerow([i,1])
        else :
                w.writerow([i,0])
f.close()

print('output result done')                                                                                                                                                                                                                                                  
