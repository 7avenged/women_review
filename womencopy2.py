import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC

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
print(y[9])
print(np.shape(X))
print(np.shape(y))
pbar = ProgressBar() #for showing the progress bar :)

print(type(y))

#for i in pbar(range(len(X))) :
#	X[i] = str(X[i])
#	X[i] = re.sub(r'\d+', '', X[i]) #remove numbers(yeah it will make the comments vague but still the algorithm will get some idea of what the lady wants to say as the words surrounding the numbers will tell the algo it the lady wants to convey(height-xx inch and the dress was shorrt- the algo understood that height was there and and the dress was short))

#df3 = pd.DataFrame({"Review_Text" :X})  #write the cleaned training data to  new file for loading 
#df3.to_csv("output113ed.csv", index=False)



bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
X = bow_transformer.transform(X)
#print(type(X))
#print(len(X))

#df4 = pd.DataFrame({"Review_Text" :X},index=range(len(X)+1))#pd.Series(range(1,len(X)+1)))  #write the cleaned training data to  new file for loading 
#df4.to_csv("output114.csv", index=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) #split into training and test data
#print(type(X))
#print(len(X))
#df5 = pd.DataFrame({"X_train" :X_train},index=range(len(X)+1))#range(l)pd.Series(range(1,len(X)+1)))  #write the cleaned training data to  new file for loading 
#df5.to_csv("X_train.csv", index=True)
#df6 = pd.DataFrame({"X_test" :X_test},index=range(len(X)+1))#pd.Series(range(1,len(X)+1)))  #write the cleaned training data to  new file for loading 
#df6.to_csv("X_test.csv", index=True)
#df7 = pd.DataFrame({"y_train" :y_train},index=range(len(X)+1))#pd.Series(range(1,len(X)+1)))  #write the cleaned training data to  new file for loading 
#df7.to_csv("y_train.csv", index=True)
#df8 = pd.DataFrame({"y_test" :y_test},index=range(len(X)+1))#pd.Series(range(1,len(X)+1)))  #write the cleaned training data to  new file for loading 
#df8.to_csv("y_test.csv", index=True)

nb = BernoulliNB()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)


clf = SVC()
clf.fit(X_train, y_train) 
preds1 = clf.predict(X_test)

#print(confusion_matrix(y_test, preds))
#print('\n')
#print('F1 score for binary :')
#print(metrics.f1_score(y_test, preds) )
#print('Recall score:')
#print(metrics.recall_score(y_test, preds) )
print('Accuracy score:')
print(accuracy_score(y_test, preds))

print('precision score :')
print(precision_score(y_test, preds, average='micro') )

print('Accuracy score SVC:')
print(accuracy_score(y_test, preds1))

print('precision score SVC:')
print(precision_score(y_test, preds1, average='micro') )


#df8 = pd.DataFrame({"predition" :y_test})
#df8.to_csv("accuracy.csv", index=True)

#print(classification_report(y_test, preds))
