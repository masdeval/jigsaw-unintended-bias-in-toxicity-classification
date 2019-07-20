import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import cross_validate
import pandas as pd
import os
import preprocess_twitter as stanfordPreprocessing
from sklearn.metrics import roc_auc_score
import sklearn
from sklearn.linear_model import  LogisticRegressionCV
from sklearn.linear_model import  SGDClassifier
import gc
import pickle
import gensim.downloader as api
import random
from sklearn.externals import joblib
from collections import defaultdict
import dill
import copy
import json


groups = ['black','christian','female','homosexual_gay_or_lesbian','jewish','male','muslim',
          'psychiatric_or_mental_illness','white']


embedding = api.load('glove-wiki-gigaword-200')

def buildVector(tokens, word2vec, size=150, replacementIdentity=False,isInSomeGroup = False, isToxic = False, weights = None):
    vec = np.zeros(size)
    count = 0.

    for word in tokens:
        try:
            if replacementIdentity: # this is to create feature for train
                if isToxic and isInSomeGroup:
                    if word in weights:
                        aux = word2vec[word]
                        try:
                            aux = aux + weights[word]['add_pleasent'] * pleasentVector - weights[word]['sub_unpleasent'] * unpleasentVector
                        except: None

                        try:
                            aux = aux - weights[word]['sub_pleasent'] * pleasentVector + weights[word]['add_unpleasent'] * unpleasentVector
                        except: None

                        vec += aux
                    else:
                        vec += word2vec[word]

                else:
                    vec += word2vec[word]
            else: # this is to create feature for test
                vec += word2vec[word]

            count += 1.
        except KeyError: # handling the case where the token is not present
            #print("\nWord not found : " + word)
            continue

    return vec

def createFeatures(data,embedding,size,tfidf=None, replaceIdentity = False, model = None):
  features = []
  # Creating a representation for the whole tweet
  for index,sample in (data.iterrows()):

     words = gensim.utils.simple_preprocess(sample['comment_text'])
     # words = stanfordPreprocessing.tokenize(word).split()

     if tfidf is not None:
      #With tfidf
      features.append(buildVectorTFIDF(words, embedding, tfidfVectorizer, tfidf.getrow(i).toarray(), size))
     else:
       if replaceIdentity:
           # train
           try:
              weights_json = dill.load(open('weights_json', 'rb'))
           except Exception:
              print('Problem loading the weights!')
              raise

           # apply the weights only in examples that belong to groups
           #features.append(buildVector(words,embedding,size = size, replaceIdentity = replaceIdentity, isInSomeGroup = isInSomeGroup(sample), isToxic=sample['target']>0.5, weights = weights_json))

           # apply the weights to any toxic example
           #features.append(buildVector(words, embedding, size=size, replaceIdentity=replaceIdentity,
           #                            isInSomeGroup=True, isToxic=sample['target'] > 0.5,
           #                            weights=weights_json))

           # do not use weights
           #features.append(buildVector(words,embedding,size = size))

       else:
           # test
           features.append(buildVector(words, embedding, size=size))

  #joblib.dump(features, open('word2vec_features_train.save', 'wb'))
  #del features_train
  #gc.collect()
  return features

def filter_frame_v2(frame, keyword=None, length=None):
    if keyword:
        frame = frame[frame[keyword] > 0.5]
    if length:
        frame = frame[frame['length'] <= length]
    return frame

#### Train
# X_train = pd.read_csv('balanced_train.csv', sep=',')
# Y_train = pd.read_csv('balanced_train_Y.csv', sep=',', usecols=['toxic'])

# Select only groups
# data = pd.concat([X_train, Y_train], axis=1)
# data_aux = None
# for term in groups:
#     frame = filter_frame_v2(data, term)
#     frame['group'] = 1 # Creating target variable group member
#     data_aux = pd.concat([data_aux,frame],axis=0)
#
# #select the non group examples
# non_group = data[~data.index.isin(data_aux.index)].sample(frac=.5)
# non_group['group'] = 0
# data = pd.concat([data_aux,non_group],axis=0)
# data.reset_index(inplace=True)
# data_y = data['group']
#
# del X_train
# del data_aux
# del non_group
# gc.collect()

#Training
# model = SGDClassifier(loss='log',penalty='l2',n_jobs=2)
# for index, sample in data.iterrows():
#     words = gensim.utils.simple_preprocess(sample['comment_text'])
#     features = buildVector(words, embedding, size=200)
#     y = data_y.iloc[index]
#     y = np.array([int(y)])
#     model.partial_fit(features.reshape(1, -1), y, classes=[1, 0])
# pickle.dump(model, open('group_predictor_model.save', 'wb'), protocol=2)


#### Test
data = pd.read_csv('balanced_test.csv', sep=',')
#Y_test = pd.read_csv('balanced_test_Y.csv', sep=',', usecols=['toxic'])

loaded_model = pickle.load(open('group_predictor_model.save', 'rb'))
data_aux = None
for term in groups:
    frame = filter_frame_v2(data, term)
    frame['group'] = 1 # Creating target variable group member
    data_aux = pd.concat([data_aux,frame],axis=0)
#select the non group examples
non_group = data[~data.index.isin(data_aux.index)]
non_group['group'] = 0
data = pd.concat([data_aux,non_group],axis=0)
data.reset_index(inplace=True)
data_y = data['group']

features_test = createFeatures(data, embedding, size=200)
pred = loaded_model.predict_proba(features_test)[:, 1]
pickle.dump(pred, open('group_predictor_prediction.save', 'wb'), protocol=2)
auc = roc_auc_score(data_y, pred)
print('Overall Test ROC AUC: %.3f' %auc)
