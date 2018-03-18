import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import re

import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

import nltk
from nltk.corpus import stopwords

from sklearn.datasets import fetch_20newsgroups
from progressbar import ProgressBar

def text_process(text):

    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''

    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


yelp = pd.read_csv('women.csv')
#print(yelp.shape)
X = yelp['Review_Text']
y = yelp['Recommended']

#print(X.head)

pbar = ProgressBar() #for showing the progress bar :)

#text="Love this dress!  it's sooo pretty.  i happened to find it in a store, and i'm glad i did bc i never would have ordered it online bc it's petite.  i bought a petite and am 5'8.  i love the length on me- hits just a little below the knee.  would definitely be a true midi on someone who is truly petite."
for i in pbar(range(len(X))) :
	X[i] = str(X[i])
	X[i] = re.sub(r'\d+', '', X[i]) #remove numbers(yeah it will make the comments vague but still the algorithm will get some idea of what the lady wants to say as the words surrounding the numbers will tell the algo it the lady wants to convey(height-xx inch and the dress was shorrt- the algo understood that height was there and and the dress was short))
	#print(X[i])
	#print(type(X[i]))
 
#print(X[5])
#print(type(X[5]))
#X[5] = re.sub(r'\d+', '', X[5])
#print(X[5])
#bow_transformer = CountVectorizer(analyzer=text_process).fit(output)


