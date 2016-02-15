import string
import collections
 
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import numpy as np #from numpy package
import sklearn.cluster  # from sklearn package
# import distance #from distance package
import fnmatch
import glob,os
import slate
# import site; 
# print site.getsitepackages()
def filemaker(name_of_file):
	with open(name_of_file) as f:
		doc = slate.PDF(f)	
	data  = np.asarray(doc)
	return data

matches = []
os.chdir("/home/madbitloman/Documents")
for file in glob.glob("*.pdf"):
    matches.append(file)

for f_names in matches:
	print f_names
	out=filemaker(f_names)
	# print out   

def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    text = text.translate(None, string.punctuation)
    tokens = word_tokenize(text)
 
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
 
    return tokens
 
 
def cluster_texts(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 stop_words=stopwords.words('english'),
                                 max_df=0.5,
                                 min_df=0.1,
                                 lowercase=True)
 
    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
 
    clustering = collections.defaultdict(list)
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
 
    return clustering	

clusters=cluster_texts(out,7)    